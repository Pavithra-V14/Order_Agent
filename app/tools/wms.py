"""
WMS tool — architecture doc edge cases 3.1/3.2 (phantom stock, damaged/
quarantined stock) and Layer 5 (idempotent transfers, multi-warehouse
race conditions). Real read/write operations against `MockInventoryRecord`.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.db import MockInventoryRecord
from app.tools.idempotency import with_idempotency


def get_stock(db: Session, sku: str, warehouse: str | None = None) -> list[dict]:
    """Returns sellable_qty explicitly separate from on_hand_qty — the
    Inventory Agent (Phase 6) must query sellable_qty for fulfillment
    decisions, never on_hand_qty alone (edge case 3.2: quantity can be
    correct on-hand but not actually fulfillable if damaged/quarantined)."""
    q = db.query(MockInventoryRecord).filter(MockInventoryRecord.sku == sku)
    if warehouse:
        q = q.filter(MockInventoryRecord.warehouse == warehouse)
    return [
        {
            "sku": r.sku,
            "warehouse": r.warehouse,
            "on_hand_qty": r.on_hand_qty,
            "sellable_qty": r.sellable_qty,
        }
        for r in q.all()
    ]


def seed_stock(db: Session, sku: str, warehouse: str, on_hand_qty: float, sellable_qty: float) -> dict:
    record = MockInventoryRecord(
        sku=sku, warehouse=warehouse, on_hand_qty=on_hand_qty, sellable_qty=sellable_qty
    )
    db.add(record)
    db.commit()
    return {"sku": sku, "warehouse": warehouse, "on_hand_qty": on_hand_qty, "sellable_qty": sellable_qty}


def transfer_stock(db: Session, sku: str, from_warehouse: str, to_warehouse: str,
                    qty: float, idempotency_key: str) -> dict:
    """Idempotent, real quantity-decrementing transfer. A retried call
    with the same idempotency_key returns the first call's result without
    decrementing stock a second time — this is what actually prevents the
    'two concurrent cases claim the last unit' race condition, combined
    with the row-level lock a real Postgres deployment would add via
    SELECT ... FOR UPDATE (SQLite doesn't support row locking the same
    way; documented here rather than silently pretending SQLite gives you
    the same guarantee production Postgres would)."""

    def _execute() -> dict:
        from_stock = db.query(MockInventoryRecord).filter(
            MockInventoryRecord.sku == sku, MockInventoryRecord.warehouse == from_warehouse
        ).first()
        if from_stock is None or from_stock.sellable_qty < qty:
            raise ValueError(
                f"Insufficient sellable stock for {sku} at {from_warehouse}: "
                f"requested {qty}, available {from_stock.sellable_qty if from_stock else 0}"
            )
        to_stock = db.query(MockInventoryRecord).filter(
            MockInventoryRecord.sku == sku, MockInventoryRecord.warehouse == to_warehouse
        ).first()
        if to_stock is None:
            to_stock = MockInventoryRecord(sku=sku, warehouse=to_warehouse, on_hand_qty=0, sellable_qty=0)
            db.add(to_stock)

        from_stock.on_hand_qty -= qty
        from_stock.sellable_qty -= qty
        to_stock.on_hand_qty += qty
        to_stock.sellable_qty += qty
        # NOT committed here — with_idempotency() commits this staged
        # mutation together with the idempotency record in one atomic
        # transaction (see that function's docstring for why this matters).

        return {
            "sku": sku, "from_warehouse": from_warehouse, "to_warehouse": to_warehouse,
            "qty": qty, "from_remaining_sellable": from_stock.sellable_qty,
        }

    result, was_replayed = with_idempotency(
        db=db,
        idempotency_key=idempotency_key,
        tool_name="wms_transfer",
        request_args={"sku": sku, "from_warehouse": from_warehouse, "to_warehouse": to_warehouse, "qty": qty},
        execute_fn=_execute,
    )
    result["_was_replayed"] = was_replayed
    return result


def handle_inventory_update_webhook(db: Session, sku: str, warehouse: str,
                                     new_on_hand_qty: float, new_sellable_qty: float) -> dict:
    """Simulates the inbound inventory-update webhook handler (Phase 12
    will wire this to a real FastAPI POST /webhooks/wms route). Updates
    the source-of-truth DB record AND invalidates the tool-response cache
    in the SAME call — this is the concrete mechanism behind architecture
    doc 8.9's "an inbound webhook actively invalidates the relevant cache
    key rather than waiting out the TTL," and what prevents the
    phantom-stock (edge case 3.1) and marketplace-lag (edge case 3.4)
    failure modes: a diagnosis running concurrently with this webhook
    will see the new numbers on its NEXT cached read, not up to 45
    seconds later."""
    record = db.query(MockInventoryRecord).filter(
        MockInventoryRecord.sku == sku, MockInventoryRecord.warehouse == warehouse
    ).first()
    if record is None:
        record = MockInventoryRecord(sku=sku, warehouse=warehouse, on_hand_qty=0, sellable_qty=0)
        db.add(record)
    record.on_hand_qty = new_on_hand_qty
    record.sellable_qty = new_sellable_qty
    db.commit()

    from app.cache.tool_cache import invalidate_stock_cache
    invalidated_count = invalidate_stock_cache(sku, warehouse)

    return {
        "sku": sku, "warehouse": warehouse,
        "new_on_hand_qty": new_on_hand_qty, "new_sellable_qty": new_sellable_qty,
        "cache_keys_invalidated": invalidated_count,
    }
