"""
Notification tool — per the build checklist's explicit note: "can be a
stub that logs instead of sending, for portfolio purposes." Kept behind
the same interface a real email/SMS provider wrapper would use so it's a
one-function swap later, not a redesign.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger("notifications")

_sent_log: list[dict] = []  # in-memory, inspectable in tests


def send_notification(customer_id: str, channel: str, subject: str, body: str) -> dict:
    """channel: 'email' | 'sms'. Real implementation would call
    SendGrid/Twilio/etc; this logs and records to an in-memory list so
    tests can assert on what would have been sent."""
    record = {
        "customer_id": customer_id,
        "channel": channel,
        "subject": subject,
        "body": body,
        "sent_at": datetime.now(timezone.utc).isoformat(),
    }
    logger.info("NOTIFICATION [%s -> %s]: %s", channel, customer_id, subject)
    _sent_log.append(record)
    return record


def get_sent_notifications(customer_id: str | None = None) -> list[dict]:
    if customer_id:
        return [n for n in _sent_log if n["customer_id"] == customer_id]
    return list(_sent_log)


def reset_sent_log() -> None:
    _sent_log.clear()
