"""
Job handlers - the actual work that happens off the request thread, per
architecture doc 8.1's "ack fast, process async" webhook contract. Each
handler opens its own DB session (runs on the worker thread) and is
registered against a job_type in app.workers.job_queue.
"""
from __future__ import annotations

import uuid

from app.core.db import ExceptionCase, CaseState, AuditLogEntry
from app.tools import oms, wms as wms_tool, carrier as carrier_tool

_EXCEPTION_TRIGGERING_STATUSES = {"payment_failed", "delivery_exception", "return_requested"}


def _get_session():
    """Imported at CALL time, not module-import time — app.core.db is
    reloaded per-test-file in this project's test fixtures (each phase
    isolates its own SQLite DB), and a module-level `from app.core.db
    import SessionLocal` would bind to whatever engine existed the first
    time this module was ever imported in a pytest session, silently
    going stale on every later reload. Confirmed as a real bug during
    Phase 12 testing (a webhook job failed with "No such order" because
    it was querying an abandoned engine) — this function-level import is
    the fix, matching the pattern already used for other reload-sensitive
    singletons (e.g. app.rag.vectorstore's client)."""
    from app.core.db import SessionLocal
    return SessionLocal()


def _create_case_if_needed(db, order_id, exception_type):
    order = oms.get_order(db, order_id)
    if order is None:
        return None
    case = ExceptionCase(
        id=str(uuid.uuid4()), order_id=order_id, customer_id=order["customer_id"],
        channel=order["channel"], exception_type=exception_type, state=CaseState.DETECTED,
    )
    db.add(case)
    db.add(AuditLogEntry(case_id=case.id, actor="system", action="state_transition",
                          detail={"from": None, "to": "detected", "reason": f"webhook: {exception_type}"}))
    db.commit()
    return case.id


def handle_oms_webhook(payload: dict) -> dict:
    """payload: {order_id, new_status}"""
    db = _get_session()
    try:
        order_id = payload["order_id"]
        new_status = payload["new_status"]
        oms.update_order_status(db, order_id, new_status)

        case_id = None
        if new_status in _EXCEPTION_TRIGGERING_STATUSES:
            case_id = _create_case_if_needed(
                db, order_id, exception_type="payment" if "payment" in new_status else "return"
            )

        return {"order_id": order_id, "new_status": new_status, "case_created": case_id}
    finally:
        db.close()


def handle_inventory_webhook(payload: dict) -> dict:
    """payload: {sku, warehouse, new_on_hand_qty, new_sellable_qty}"""
    db = _get_session()
    try:
        return wms_tool.handle_inventory_update_webhook(
            db, sku=payload["sku"], warehouse=payload["warehouse"],
            new_on_hand_qty=payload["new_on_hand_qty"], new_sellable_qty=payload["new_sellable_qty"],
        )
    finally:
        db.close()


def handle_carrier_webhook(payload: dict) -> dict:
    """payload: {tracking_number, new_status, order_id (optional)}"""
    result = carrier_tool.handle_carrier_status_webhook(
        tracking_number=payload["tracking_number"], new_status=payload["new_status"],
    )
    if payload.get("order_id") and payload["new_status"] in ("delivery_exception", "lost"):
        db = _get_session()
        try:
            case_id = _create_case_if_needed(db, payload["order_id"], exception_type="carrier")
            result["case_created"] = case_id
        finally:
            db.close()
    return result
