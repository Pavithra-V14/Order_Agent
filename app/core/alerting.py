"""
Alerting - architecture doc Layer 11 / 8.8: basic alerting for circuit
breaker trips, idempotency collisions, and guardrail Tier-1 blocks. No
network access to any webhook (Slack or otherwise) from this sandbox, so
alerts are logged + persisted to AlertRecord. Swap point: send_alert()'s
body gets a requests.post(slack_webhook_url, ...) call added once a real
webhook URL is configured - callers don't change.
"""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from app.core.db import AlertRecord

logger = logging.getLogger("alerts")


def send_alert(db: Session, event_type: str, detail: dict) -> AlertRecord:
    """event_type: 'circuit_breaker_trip' | 'idempotency_collision' | 'tier1_block' | 'log_episode_failure' | 'auto_pipeline_failure' | 'tier3_judge_flagged'."""
    logger.warning("ALERT [%s]: %s", event_type, detail)
    record = AlertRecord(event_type=event_type, detail=detail)
    db.add(record)
    db.commit()
    return record


def get_recent_alerts(db: Session, event_type: str = None, limit: int = 50) -> list:
    q = db.query(AlertRecord)
    if event_type:
        q = q.filter(AlertRecord.event_type == event_type)
    q = q.order_by(AlertRecord.created_at.desc()).limit(limit)
    return [
        {"id": a.id, "event_type": a.event_type, "detail": a.detail, "created_at": a.created_at.isoformat()}
        for a in q.all()
    ]
