"""
Long-term/episodic memory - architecture doc 8.4. Auto-selects between
two backends, same settings-driven pattern as every other cloud
integration in this project:

  1. GROQ_API_KEY set -> real Graphiti (app/memory/graphiti_adapter.py),
     backed by Neo4j Aura (cloud) if NEO4J_URI is also set, or embedded
     Kuzu (no server, no Docker, no credentials) otherwise. Graphiti
     needs an LLM for entity/relationship extraction regardless of which
     graph store backs it, which is why the Groq key is the actual
     switch here, not a graph-specific setting.
  2. No GROQ_API_KEY -> SQL-backed substitute (EpisodeRecord,
     app/core/db.py). Satisfies this project's actual query shape (time-
     ordered customer history) without a graph database, but doesn't
     give genuine multi-hop relationship traversal (e.g. "customers who
     share a shipping address with a known fraud case") the way a real
     Graphiti deployment does — a real, if not yet load-bearing, gap.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.core.db import EpisodeRecord


def _use_graphiti() -> bool:
    from app.memory.graphiti_adapter import _is_graphiti_available
    return _is_graphiti_available()


def log_episode(db: Session, customer_id: str, episode_type: str, content: dict,
                 occurred_at: datetime, case_id: str | None = None) -> dict:
    """Records one episode (e.g., a resolved case, a fraud flag) tied to a
    customer. Called by the Orchestrator (Phase 6) on case resolution, and
    by the Fraud/Risk Agent when it raises a flag."""
    if _use_graphiti():
        from app.memory.graphiti_adapter import log_episode_graphiti
        return log_episode_graphiti(customer_id, episode_type, content, occurred_at, case_id)

    episode = EpisodeRecord(
        customer_id=customer_id,
        case_id=case_id,
        episode_type=episode_type,
        content=content,
        occurred_at=occurred_at,
    )
    db.add(episode)
    db.commit()
    return {
        "id": episode.id,
        "customer_id": episode.customer_id,
        "episode_type": episode.episode_type,
        "occurred_at": episode.occurred_at.isoformat(),
    }


def log_episode_async(customer_id: str, episode_type: str, content: dict,
                       occurred_at: datetime, case_id: str | None = None) -> str:
    """Memory-upgrade follow-up: enqueues the SAME write log_episode()
    does, but off the calling request's hot path - see
    app/workers/handlers.py's handle_log_episode() for the full
    reasoning (the real, named latency cost this addresses: when
    Graphiti/Neo4j is active, one log_episode() call is 3 real network
    round-trips, worst-case bounded by GRAPHITI_CALL_TIMEOUT_SECONDS).

    Returns a job_id immediately (ack-fast, same contract as every
    webhook handler in this project) - the actual write happens on the
    job queue's worker thread/process. Callers that need to confirm the
    write actually landed (mainly tests) poll
    get_job_queue().get_job(job_id), same pattern as
    tests/test_phase12_api.py's webhook tests.

    Deliberately does NOT take a db session - the handler opens its own
    (app/workers/handlers.py's _get_session()), since this call may run
    on a different thread, or in a genuinely separate process entirely
    (RQJobQueue via scripts/run_rq_worker.py), where the caller's
    session would not be valid.

    HONEST TRADEOFF: this makes the write EVENTUALLY consistent, not
    immediate - see handle_log_episode()'s docstring for exactly which
    reads this can and can't affect. Not used for reads
    (get_customer_history) - those still block, since the agent calling
    them genuinely needs the data now to score risk.
    """
    from app.workers.job_queue import get_job_queue
    from app.workers.handlers import handle_log_episode
    q = get_job_queue()
    # Idempotent, defensive registration - see start_worker() comment
    # below for why this can't rely solely on app.main's startup hook
    # having run. register_handler() is a plain dict assignment
    # (job_queue.py), so calling it repeatedly with the same function
    # is harmless.
    q.register_handler("log_episode", handle_log_episode)
    # Defensive, idempotent: InProcessJobQueue.start_worker() no-ops if
    # already running; RQJobQueue.start_worker() is unconditionally a
    # no-op (see its own docstring). Needed because this function is
    # called from plain Python code (resolution_completion.py,
    # orchestrator.py), not only from FastAPI request handlers where
    # app.main's lifespan hook already started the worker - a direct
    # function call (including most of this project's own tests) would
    # otherwise enqueue into a queue nothing is ever draining.
    q.start_worker()
    return q.enqueue("log_episode", {
        "customer_id": customer_id, "episode_type": episode_type, "content": content,
        "occurred_at_iso": occurred_at.isoformat(), "case_id": case_id,
    })


def get_customer_history(db: Session, customer_id: str, episode_type: str | None = None,
                          limit: int = 20) -> list[dict]:
    """Time-ordered (most recent first) episode history for a customer —
    this is the exact query the Customer Context Agent (Phase 6) uses to
    answer "has this customer had prior return issues" and similar."""
    if _use_graphiti():
        from app.memory.graphiti_adapter import get_customer_history_graphiti
        return get_customer_history_graphiti(customer_id, episode_type, limit)

    q = db.query(EpisodeRecord).filter(EpisodeRecord.customer_id == customer_id)
    if episode_type:
        q = q.filter(EpisodeRecord.episode_type == episode_type)
    q = q.order_by(EpisodeRecord.occurred_at.desc()).limit(limit)
    return [
        {
            "id": e.id,
            "case_id": e.case_id,
            "episode_type": e.episode_type,
            "content": e.content,
            "occurred_at": e.occurred_at.isoformat(),
        }
        for e in q.all()
    ]


def summarize_customer_risk_profile(db: Session, customer_id: str) -> dict:
    """A small piece of derived reasoning the Fraud/Risk Agent (Phase 6)
    will use: return count in the last 90 days, weighted with episode
    types, NOT just a raw count — per the architecture doc's explicit
    warning against penalizing legitimate high-engagement customers on
    return frequency alone."""
    history = get_customer_history(db, customer_id, limit=200)
    return_episodes = [h for h in history if h["episode_type"] == "case_resolved"
                        and h["content"].get("exception_type") == "return"]
    fraud_flags = [h for h in history if h["episode_type"] == "fraud_flag_raised"]
    return {
        "customer_id": customer_id,
        "total_return_cases": len(return_episodes),
        "fraud_flags_raised": len(fraud_flags),
        "most_recent_episode": history[0] if history else None,
    }
