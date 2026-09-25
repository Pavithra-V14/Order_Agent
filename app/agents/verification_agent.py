"""
Verification Agent - architecture doc Part 3 topology: confirms a
resolution actually completed before the case is marked resolved. Never
trusts the Execution Agent's return value alone - re-checks against
independent state (the idempotency record itself, or the carrier's
tracking status), which is what catches a scenario where execution
THINKS it succeeded but the downstream system's state doesn't actually
reflect it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from sqlalchemy.orm import Session

from app.core.db import IdempotencyRecord
from app.tools import carrier as carrier_tool


class VerificationStatus(str, Enum):
    VERIFIED = "verified"
    NOT_YET_VERIFIABLE = "not_yet_verifiable"
    VERIFICATION_FAILED = "verification_failed"


@dataclass
class VerificationResult:
    status: VerificationStatus
    detail: str = ""


def verify_refund(db: Session, idempotency_key: str, refund_id: str | None = None) -> VerificationResult:
    """Independent check: does a completed idempotency record for this
    EXACT key exist with a 'succeeded' result? This is deliberately NOT
    just re-reading the Execution Agent's in-memory return value - it
    re-derives the answer from the durable record.

    The real Stripe gateway relies on Stripe's own idempotency and writes
    no local record, so when there is none and a refund_id is known, the
    refund is re-fetched from the provider instead."""
    record = db.get(IdempotencyRecord, idempotency_key)
    if record is None and refund_id:
        from app.tools.payment import get_payment_gateway
        refund = get_payment_gateway().get_refund(refund_id)
        if refund is not None:
            if refund["status"] == "succeeded":
                return VerificationResult(status=VerificationStatus.VERIFIED,
                                           detail=f"Refund {refund_id} confirmed by the payment provider.")
            if refund["status"] in ("pending", "requires_action"):
                return VerificationResult(status=VerificationStatus.NOT_YET_VERIFIABLE,
                                           detail=f"Refund {refund_id} is {refund['status']} at the provider.")
            return VerificationResult(status=VerificationStatus.VERIFICATION_FAILED,
                                       detail=f"Provider reports refund {refund_id} as {refund['status']!r}.")
    if record is None:
        return VerificationResult(status=VerificationStatus.VERIFICATION_FAILED,
                                   detail=f"No idempotency record found for key {idempotency_key!r} - "
                                          f"execution may not have actually completed.")
    if record.result.get("status") != "succeeded":
        return VerificationResult(status=VerificationStatus.VERIFICATION_FAILED,
                                   detail=f"Idempotency record exists but status is "
                                          f"{record.result.get('status')!r}, not 'succeeded'.")
    return VerificationResult(status=VerificationStatus.VERIFIED,
                               detail=f"Refund {record.result.get('id')} confirmed via idempotency record.")


def verify_reship(db: Session, idempotency_key: str) -> VerificationResult:
    """A generated label isn't fulfillment - verification checks the
    carrier's OWN tracking status, not just that a label object exists."""
    record = db.get(IdempotencyRecord, idempotency_key)
    if record is None:
        return VerificationResult(status=VerificationStatus.VERIFICATION_FAILED,
                                   detail=f"No idempotency record found for key {idempotency_key!r}.")
    tracking_number = record.result.get("tracking_number")
    if not tracking_number:
        return VerificationResult(status=VerificationStatus.VERIFICATION_FAILED,
                                   detail="Label record has no tracking_number.")

    gateway = carrier_tool.get_carrier_gateway()
    tracking = gateway.get_tracking_status(tracking_number)
    if tracking["status"] in ("label_created",):
        return VerificationResult(status=VerificationStatus.NOT_YET_VERIFIABLE,
                                   detail=f"Label {tracking_number} created but not yet scanned by carrier.")
    if tracking["status"] == "unknown":
        return VerificationResult(status=VerificationStatus.VERIFICATION_FAILED,
                                   detail=f"Tracking number {tracking_number} not recognized by carrier.")
    return VerificationResult(status=VerificationStatus.VERIFIED,
                               detail=f"Carrier confirms status: {tracking['status']}")
