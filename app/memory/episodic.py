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
