"""
OMS tool — architecture doc Part 6/8.4. Real read/write operations against
`MockOrderRecord` (app/core/db.py) — this project has no access to an
actual OMS, so a self-hosted equivalent stands in, but the operations
(lookup by order_id, status update) are the same shape a real OMS's API
would expose, and the Diagnosis Agent (Phase 6) will query this exactly
as it would query a live OMS.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.db import MockOrderRecord


def get_order(db: Session, order_id: str) -> dict | None:
    order = db.get(MockOrderRecord, order_id)
    if order is None:
        return None
    return {
        "order_id": order.order_id,
        "customer_id": order.customer_id,
        "channel": order.channel,
        "status": order.status,
        "payment_intent_id": order.payment_intent_id,
        "total_amount_usd": order.total_amount_usd,
        "payment_method_breakdown": order.payment_method_breakdown,
        "purchase_date": order.purchase_date.isoformat(),
        "line_items": order.line_items,
    }


def update_order_status(db: Session, order_id: str, new_status: str) -> dict:
    """Not idempotency-wrapped: a status update is naturally idempotent
    (setting status to the same value twice has the same effect as once),
    unlike a refund or label generation, which each create a NEW side
    effect on every call. Per architecture doc Layer 5, idempotency keys
    are for actions that aren't naturally safe to repeat — this isn't one."""
    order = db.get(MockOrderRecord, order_id)
    if order is None:
        raise ValueError(f"No such order: {order_id}")
    order.status = new_status
    db.commit()
    return get_order(db, order_id)


def create_order(db: Session, order_id: str, customer_id: str, channel: str, status: str,
                  total_amount_usd: float, purchase_date, line_items: list[dict],
                  payment_method_breakdown: dict | None = None, payment_intent_id: str | None = None) -> dict:
    """Test/seeding helper — a real OMS wouldn't expose order creation to
    this system (orders originate upstream); this exists so Phase 4/6
    tests can set up realistic order state without a real OMS to write against."""
    order = MockOrderRecord(
        order_id=order_id,
        customer_id=customer_id,
        channel=channel,
        status=status,
        payment_intent_id=payment_intent_id,
        total_amount_usd=total_amount_usd,
        purchase_date=purchase_date,
        line_items=line_items,
        payment_method_breakdown=payment_method_breakdown,
    )
    db.add(order)
    db.commit()
    return get_order(db, order_id)
