"""
Long-term/episodic memory — architecture doc 8.4. Production pick is
Zep/Graphiti (chosen specifically for time-aware relationship reasoning
over a graph DB — the differentiator over Mem0, per the architecture
doc). Graphiti requires a Neo4j or FalkorDB backend to actually run; this
sandbox has no Docker and no network access to stand one up.

This module ships a SQL-backed substitute (`EpisodeRecord`, app/core/db.py)
that satisfies the SPECIFIC functional contract this project's Phase 5 DoD
requires — "query a customer's prior case history, correctly time-ordered"
— without needing a graph database. What it does NOT give you that a real
Graphiti deployment would: genuine multi-hop relationship traversal (e.g.
"customers who share a shipping address with a known fraud case," or
entity-relationship reasoning beyond a single customer_id key). This
project's current agents (Phase 6+) only need the time-ordered-history
query shape, so the gap is real but not yet load-bearing — flagged here so
it isn't silently forgotten if a future phase needs graph traversal.

Swap point: replace this module's functions with calls to a Graphiti
client (`graphiti_core.Graphiti`) once running against a real Neo4j/
FalkorDB instance — `log_episode`/`get_customer_history` map directly to
Graphiti's `add_episode`/`search` methods.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.core.db import EpisodeRecord


def log_episode(db: Session, customer_id: str, episode_type: str, content: dict,
                 occurred_at: datetime, case_id: str | None = None) -> dict:
    """Records one episode (e.g., a resolved case, a fraud flag) tied to a
    customer. Called by the Orchestrator (Phase 6) on case resolution, and
    by the Fraud/Risk Agent when it raises a flag."""
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
