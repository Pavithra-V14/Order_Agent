"""
Phase 4 DoD: every tool independently callable with a passing unit test;
the idempotency test proves a retried write does NOT double-execute.
"""
import os
import tempfile
from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    """Fresh SQLite file per test module run + fresh Base.metadata create,
    so Phase 4 tests don't collide with Phase 0/3's dev DB or each other."""
    tmp_path = os.path.join(tempfile.gettempdir(), "test_phase4.db")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}"

    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)  # rebuild engine/SessionLocal against the new DATABASE_URL
    db_module.init_db()

    yield db_module

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


@pytest.fixture(autouse=True)
def reset_fake_gateways():
    from app.tools.payment import reset_fake_gateway
    from app.tools.carrier import reset_fake_carrier
    from app.tools.notification import reset_sent_log
    reset_fake_gateway()
    reset_fake_carrier()
    reset_sent_log()
    yield


# ---------------------------------------------------------------------------
# THE critical test: idempotency actually prevents double-execution
# ---------------------------------------------------------------------------
def test_duplicate_refund_call_does_not_issue_second_refund(isolated_db):
    """Simulates the exact failure mode named in the architecture doc's
    Part 0 (\"≥3 failure modes\"): a retried refund call with the same
    idempotency_key must NOT result in two refunds."""
    from app.tools.payment import get_payment_gateway

    db = isolated_db.SessionLocal()
    gateway = get_payment_gateway()
    gateway.seed_transaction("pi_test_123", amount_usd=100.0)

    key = "case-abc-refund-attempt-1"
    result1 = gateway.issue_refund(db, "pi_test_123", 42.0, idempotency_key=key)
    result2 = gateway.issue_refund(db, "pi_test_123", 42.0, idempotency_key=key)

    assert gateway.refund_call_count == 1, (
        f"Expected exactly ONE real refund execution despite two calls with "
        f"the same idempotency_key; got {gateway.refund_call_count}"
    )
    assert result1["id"] == result2["id"], "replayed call must return the SAME refund id"
    assert result1["_was_replayed"] is False
    assert result2["_was_replayed"] is True
    db.close()


def test_different_idempotency_keys_do_issue_separate_refunds(isolated_db):
    """Sanity check the inverse: this ISN'T a blanket dedup on
    (payment_intent_id, amount) — different keys mean genuinely different
    requests (e.g. two separate legitimate partial refunds)."""
    from app.tools.payment import get_payment_gateway

    db = isolated_db.SessionLocal()
    gateway = get_payment_gateway()
    gateway.seed_transaction("pi_test_456", amount_usd=100.0)

    r1 = gateway.issue_refund(db, "pi_test_456", 20.0, idempotency_key="key-1")
    r2 = gateway.issue_refund(db, "pi_test_456", 20.0, idempotency_key="key-2")

    assert gateway.refund_call_count == 2
    assert r1["id"] != r2["id"]
    db.close()


def test_reusing_key_with_different_args_raises(isolated_db):
    """A retried call MUST carry the same args as the original — reusing a
    key with different arguments is a bug in the caller, and should be
    loud, not silently return a mismatched cached result."""
    from app.tools.payment import get_payment_gateway
    from app.tools.idempotency import IdempotencyKeyReusedWithDifferentArgs

    db = isolated_db.SessionLocal()
    gateway = get_payment_gateway()
    gateway.seed_transaction("pi_test_789", amount_usd=100.0)

    gateway.issue_refund(db, "pi_test_789", 10.0, idempotency_key="reused-key")
    with pytest.raises(IdempotencyKeyReusedWithDifferentArgs):
        gateway.issue_refund(db, "pi_test_789", 25.0, idempotency_key="reused-key")
    db.close()


def test_duplicate_stock_transfer_does_not_double_decrement(isolated_db):
    """The multi-warehouse race condition, applied to a simple retried
    call: stock must only move once, not twice, for the same idempotency_key."""
    from app.tools.wms import seed_stock, transfer_stock, get_stock

    db = isolated_db.SessionLocal()
    seed_stock(db, sku="SKU-1", warehouse="WH-A", on_hand_qty=10, sellable_qty=10)

    key = "transfer-attempt-1"
    transfer_stock(db, "SKU-1", "WH-A", "WH-B", qty=3, idempotency_key=key)
    transfer_stock(db, "SKU-1", "WH-A", "WH-B", qty=3, idempotency_key=key)  # retry

    wh_a_stock = get_stock(db, "SKU-1", "WH-A")[0]
    assert wh_a_stock["sellable_qty"] == 7, (
        f"Expected WH-A to lose exactly 3 units once, not twice; got sellable_qty={wh_a_stock['sellable_qty']}"
    )
    db.close()


def test_transfer_fails_cleanly_on_insufficient_sellable_stock(isolated_db):
    from app.tools.wms import seed_stock, transfer_stock

    db = isolated_db.SessionLocal()
    seed_stock(db, sku="SKU-2", warehouse="WH-A", on_hand_qty=5, sellable_qty=2)  # 3 damaged/quarantined

    with pytest.raises(ValueError):
        # requesting 4 units when only 2 are SELLABLE (not 5 on-hand) must fail
        transfer_stock(db, "SKU-2", "WH-A", "WH-B", qty=4, idempotency_key="k1")
    db.close()


# ---------------------------------------------------------------------------
# Functional tests per tool (Phase 4 DoD: every tool independently testable)
# ---------------------------------------------------------------------------
def test_oms_create_and_get_order(isolated_db):
    from app.tools.oms import create_order, get_order

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-1", customer_id="CUST-1", channel="direct", status="paid",
        total_amount_usd=99.99, purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
        line_items=[{"sku": "SKU-1", "category": "apparel", "qty": 1, "price": 99.99}],
    )
    order = get_order(db, "ORD-1")
    assert order["customer_id"] == "CUST-1"
    assert order["status"] == "paid"
    db.close()


def test_oms_update_status_is_naturally_idempotent(isolated_db):
    from app.tools.oms import create_order, update_order_status

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-2", customer_id="CUST-2", channel="direct", status="paid",
        total_amount_usd=50.0, purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc), line_items=[],
    )
    update_order_status(db, "ORD-2", "refunded")
    result = update_order_status(db, "ORD-2", "refunded")  # calling twice is safe
    assert result["status"] == "refunded"
    db.close()


def test_carrier_label_generation_is_idempotent(isolated_db):
    from app.tools.carrier import get_carrier_gateway

    db = isolated_db.SessionLocal()
    gateway = get_carrier_gateway()
    key = "label-attempt-1"
    r1 = gateway.generate_return_label(db, "ORD-3", idempotency_key=key)
    r2 = gateway.generate_return_label(db, "ORD-3", idempotency_key=key)
    assert gateway.label_call_count == 1
    assert r1["tracking_number"] == r2["tracking_number"]
    db.close()


def test_notification_send_and_query():
    from app.tools.notification import send_notification, get_sent_notifications

    send_notification("CUST-1", "email", "Your refund is on the way", "body text")
    sent = get_sent_notifications("CUST-1")
    assert len(sent) == 1
    assert sent[0]["subject"] == "Your refund is on the way"


def test_mcp_server_registers_all_nine_tools():
    import asyncio
    from app.tools.mcp_server import server

    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {
        "get_order", "update_order_status", "get_stock", "transfer_stock",
        "get_transaction_status", "issue_refund", "get_tracking_status",
        "generate_return_label", "send_customer_notification",
    }
