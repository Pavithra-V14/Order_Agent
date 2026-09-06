"""
Phase 14: mocked-tool-response tests (each tool returning malformed/
empty/error responses - confirm graceful handling, not a crash) and a
full chaos-test pass across every external dependency, not just payment
(Phase 8 covered payment specifically; this extends the same pattern to
carrier and WMS).
"""
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), "test_phase14_errors.db")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}"

    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)
    db_module.init_db()

    yield db_module

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


@pytest.fixture(autouse=True)
def reset_all():
    from app.tools.payment import reset_fake_gateway
    from app.tools.carrier import reset_fake_carrier
    from app.core.circuit_breaker import reset_all_breakers
    reset_fake_gateway()
    reset_fake_carrier()
    reset_all_breakers()
    yield


def test_oms_get_order_nonexistent_returns_none_not_crash(isolated_db):
    from app.tools.oms import get_order
    db = isolated_db.SessionLocal()
    result = get_order(db, "ORD-DOES-NOT-EXIST")
    assert result is None
    db.close()


def test_diagnosis_handles_order_not_found_gracefully(isolated_db):
    """If the order genuinely doesn't exist, the diagnosis loop must not
    crash - it should surface this as a finding, not raise unhandled."""
    from app.agents.llm_client import FakeLLMClient
    from app.agents.diagnosis_agent import run_diagnosis

    db = isolated_db.SessionLocal()
    result = run_diagnosis(db, FakeLLMClient(), order_id="ORD-MISSING-1", max_steps=5)
    assert result.terminated_reason in ("concluded", "max_steps_reached")
    assert result.findings.get("order", {}).get("error") == "order not found"
    db.close()


def test_wms_transfer_with_nonexistent_sku_raises_clean_valueerror(isolated_db):
    """A transfer request for a SKU with no stock record at all must
    raise a clean ValueError, not an unhandled AttributeError."""
    from app.tools.wms import transfer_stock
    db = isolated_db.SessionLocal()
    with pytest.raises(ValueError):
        transfer_stock(db, sku="SKU-NEVER-SEEDED", from_warehouse="WH-A", to_warehouse="WH-B",
                        qty=1, idempotency_key="k1")
    db.close()


def test_payment_get_transaction_status_nonexistent_raises_clean_error(isolated_db):
    from app.tools.payment import get_payment_gateway
    gateway = get_payment_gateway()
    with pytest.raises(ValueError):
        gateway.get_transaction_status("pi_never_seeded")


def test_carrier_get_tracking_unknown_number_returns_unknown_status_not_crash():
    """An unrecognized tracking number returns status='unknown' rather
    than raising, matching a real carrier API's 404-shaped response."""
    from app.tools.carrier import get_carrier_gateway
    gateway = get_carrier_gateway()
    result = gateway.get_tracking_status("TRK-NEVER-SEEDED")
    assert result["status"] == "unknown"


def test_resolution_workflow_handles_empty_diagnosis_causes_list(isolated_db):
    """An empty root_causes list must not crash the resolution workflow."""
    from app.agents.resolution_policy_workflow import propose_resolution_decision
    decision = propose_resolution_decision(
        diagnosis_root_causes=[], inventory_result={}, order_amount_usd=25.0,
    )
    assert decision is not None
    assert decision.action.value == "refund"


def test_tier2_rejects_completely_empty_decision_dict():
    from app.guardrails.tier2_structural import validate_structure
    result = validate_structure({})
    assert result.passed is False
    assert len(result.errors) > 0


def test_chaos_carrier_permanent_failure(isolated_db):
    """Carrier label generation with a permanently failing carrier API:
    circuit trips, no label ever gets created."""
    from app.tools.carrier import get_carrier_gateway
    from app.core.circuit_breaker import get_circuit_breaker, CircuitOpenError

    db = isolated_db.SessionLocal()
    gateway = get_carrier_gateway()

    original_generate = gateway.generate_return_label

    def failing_generate(db_arg, order_id, idempotency_key):
        raise TimeoutError("Simulated carrier API timeout (chaos test)")

    gateway.generate_return_label = failing_generate

    breaker = get_circuit_breaker("carrier_chaos_test", failure_threshold=3, reset_timeout_seconds=30.0)

    last_status = None
    for i in range(5):
        try:
            breaker.call(lambda i=i: gateway.generate_return_label(db, "ORD-CHAOS-CARRIER", f"key-{i}"))
        except CircuitOpenError:
            last_status = "circuit_open"
            break
        except TimeoutError:
            last_status = "timeout"
            continue

    assert last_status == "circuit_open", "circuit should have tripped and started failing fast"
    assert breaker.call_attempts == 3, f"expected exactly 3 real attempts before tripping, got {breaker.call_attempts}"

    gateway.generate_return_label = original_generate
    db.close()


def test_chaos_wms_transfer_failure_leaves_no_partial_state(isolated_db):
    """A WMS transfer that fails mid-transaction must not leave stock in
    a half-transferred state - proves the atomicity fix from Phase 4
    (single-transaction commit) under a forced failure, not just the
    happy path."""
    from app.tools.wms import seed_stock, get_stock
    from unittest.mock import patch

    db = isolated_db.SessionLocal()
    seed_stock(db, sku="SKU-CHAOS-WMS", warehouse="WH-A", on_hand_qty=10, sellable_qty=10)

    with patch("app.tools.wms.with_idempotency") as mock_idem:
        mock_idem.side_effect = RuntimeError("Simulated commit failure mid-transaction")
        from app.tools.wms import transfer_stock
        with pytest.raises(RuntimeError):
            transfer_stock(db, "SKU-CHAOS-WMS", "WH-A", "WH-B", qty=4, idempotency_key="chaos-key-1")

    db.rollback()
    stock = get_stock(db, "SKU-CHAOS-WMS", "WH-A")
    assert stock[0]["sellable_qty"] == 10, (
        f"Expected stock UNCHANGED after a simulated mid-transaction failure, "
        f"got sellable_qty={stock[0]['sellable_qty']}"
    )
    db.close()
