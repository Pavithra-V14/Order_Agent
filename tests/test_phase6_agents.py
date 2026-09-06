"""
Phase 6 DoD: "Given a synthetic multi-cause exception (payment decline +
concurrent OOS), the Diagnosis Agent correctly identifies both causes in
its output, and the runaway-loop test terminates within the configured
ceiling."
"""
import os
import tempfile
from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), "test_phase6.db")
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
def reset_gateways():
    from app.tools.payment import reset_fake_gateway
    from app.tools.carrier import reset_fake_carrier
    reset_fake_gateway()
    reset_fake_carrier()
    yield


def test_multi_cause_diagnosis_identifies_both_payment_and_inventory(isolated_db):
    """THE core Phase 6 acceptance test: an order with BOTH a payment
    decline AND a concurrent out-of-stock condition must surface BOTH root
    causes in the diagnosis output, not just one."""
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.tools.payment import get_payment_gateway
    from app.agents.llm_client import FakeLLMClient
    from app.agents.diagnosis_agent import run_diagnosis

    db = isolated_db.SessionLocal()

    # Order with a failed payment status
    create_order(
        db, order_id="ORD-MULTI-1", customer_id="CUST-1", channel="direct",
        status="payment_failed", total_amount_usd=150.0,
        purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
        line_items=[{"sku": "SKU-OOS", "category": "electronics", "qty": 2, "price": 75.0}],
    )
    # Payment gateway confirms the decline
    gateway = get_payment_gateway()
    gateway.seed_transaction("pi_multi_1", amount_usd=150.0, status="declined")

    # Inventory is ALSO short: only 1 sellable unit exists across warehouses, but 2 were ordered
    seed_stock(db, sku="SKU-OOS", warehouse="WH-A", on_hand_qty=1, sellable_qty=0)
    seed_stock(db, sku="SKU-OOS", warehouse="WH-B", on_hand_qty=1, sellable_qty=1)

    llm = FakeLLMClient()
    result = run_diagnosis(db, llm, order_id="ORD-MULTI-1", payment_intent_id="pi_multi_1")

    assert result.terminated_reason == "concluded"
    causes_text = " | ".join(result.root_causes)
    assert "payment_issue" in causes_text, f"Expected a payment-related root cause, got: {result.root_causes}"
    assert "inventory_issue" in causes_text or "SKU-OOS" in causes_text, (
        f"Expected an inventory-related root cause for the OOS SKU, got: {result.root_causes}"
    )
    assert len(result.root_causes) >= 2, (
        f"Expected at least 2 distinct root causes for this multi-cause scenario, got: {result.root_causes}"
    )
    db.close()


def test_single_cause_case_does_not_over_report(isolated_db):
    """Sanity check the inverse: a normal, single-issue order should NOT
    spuriously report multiple causes — proves the multi-cause test above
    is detecting real signal, not just always returning a long list."""
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.tools.payment import get_payment_gateway
    from app.agents.llm_client import FakeLLMClient
    from app.agents.diagnosis_agent import run_diagnosis

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-SINGLE-1", customer_id="CUST-2", channel="direct",
        status="paid", total_amount_usd=50.0,
        purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
        line_items=[{"sku": "SKU-OK", "category": "apparel", "qty": 1, "price": 50.0}],
    )
    gateway = get_payment_gateway()
    gateway.seed_transaction("pi_single_1", amount_usd=50.0, status="succeeded")
    seed_stock(db, sku="SKU-OK", warehouse="WH-A", on_hand_qty=10, sellable_qty=10)

    llm = FakeLLMClient()
    result = run_diagnosis(db, llm, order_id="ORD-SINGLE-1", payment_intent_id="pi_single_1")

    assert result.terminated_reason == "concluded"
    assert result.root_causes == ["no_anomaly_detected: all checked systems report normal state"]
    db.close()


def test_runaway_diagnosis_terminates_at_step_ceiling(isolated_db):
    """A deliberately non-converging planner (always requests a check with
    no data source, never concludes) must be stopped by max_steps, not
    loop forever."""
    from app.agents.llm_client import BaseLLMClient, DiagnosisStepPlan
    from app.agents.diagnosis_agent import run_diagnosis
    from app.tools.oms import create_order

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-RUNAWAY-1", customer_id="CUST-3", channel="direct", status="paid",
        total_amount_usd=10.0, purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc), line_items=[],
    )

    class NeverConcludesLLM(BaseLLMClient):
        """Deliberately broken planner: always asks for a check that has
        no data source and can never be satisfied, simulating a
        pathological/buggy planning model."""
        def plan_next_diagnosis_step(self, case_context, findings_so_far):
            return DiagnosisStepPlan(action="check_nonexistent_system",
                                      reasoning="(deliberately non-converging for this test)")

        def assess_fraud_risk(self, case_context, customer_risk_profile):
            return {"risk_score": 0.0, "flag": False, "reasons": []}

    result = run_diagnosis(db, NeverConcludesLLM(), order_id="ORD-RUNAWAY-1", max_steps=5)

    assert result.terminated_reason == "max_steps_reached"
    assert len(result.steps_taken) == 5, f"Expected exactly max_steps=5 steps taken, got {len(result.steps_taken)}"
    db.close()


def test_runaway_diagnosis_terminates_at_wall_clock_timeout(isolated_db):
    """The other half of the ceiling: even with a huge max_steps, a slow
    planner must be cut off by wall-clock time, not run indefinitely."""
    import time
    from app.agents.llm_client import BaseLLMClient, DiagnosisStepPlan
    from app.agents.diagnosis_agent import run_diagnosis
    from app.tools.oms import create_order

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-SLOW-1", customer_id="CUST-4", channel="direct", status="paid",
        total_amount_usd=10.0, purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc), line_items=[],
    )

    class SlowLLM(BaseLLMClient):
        def plan_next_diagnosis_step(self, case_context, findings_so_far):
            time.sleep(0.05)
            return DiagnosisStepPlan(action="check_nonexistent_system", reasoning="(slow, for timeout test)")

        def assess_fraud_risk(self, case_context, customer_risk_profile):
            return {"risk_score": 0.0, "flag": False, "reasons": []}

    result = run_diagnosis(db, SlowLLM(), order_id="ORD-SLOW-1", max_steps=10_000,
                            wall_clock_timeout_seconds=0.15)

    assert result.terminated_reason == "timeout_reached"
    assert len(result.steps_taken) < 10_000  # proves the timeout fired, not the step ceiling
    db.close()


def test_orchestrator_runs_full_diagnosis_phase_and_persists_state():
    """End-to-end: the LangGraph orchestrator runs start -> [diagnosis,
    fraud, inventory, customer_context] in parallel -> aggregate, and the
    ExceptionCase row reflects the results afterward — proving state
    persistence, not just in-memory graph state."""
    import os, tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), "test_phase6_orch.db")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}"

    from app.core.config import get_settings
    get_settings.cache_clear()
    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)
    db_module.init_db()

    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.tools.payment import get_payment_gateway, reset_fake_gateway
    from app.core.db import ExceptionCase, CaseState

    reset_fake_gateway()
    db = db_module.SessionLocal()
    create_order(
        db, order_id="ORD-E2E-1", customer_id="CUST-E2E", channel="direct", status="paid",
        total_amount_usd=80.0, purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
        line_items=[{"sku": "SKU-E2E", "category": "apparel", "qty": 1, "price": 80.0}],
    )
    get_payment_gateway().seed_transaction("pi_e2e_1", amount_usd=80.0, status="succeeded")
    seed_stock(db, sku="SKU-E2E", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    case = ExceptionCase(id="case-e2e-1", order_id="ORD-E2E-1", customer_id="CUST-E2E",
                          channel="direct", exception_type="payment", state=CaseState.DETECTED)
    db.add(case)
    db.commit()
    db.close()

    from app.agents.orchestrator import run_orchestrator_for_case
    final_state = run_orchestrator_for_case(
        case_id="case-e2e-1", order_id="ORD-E2E-1", customer_id="CUST-E2E",
        payment_intent_id="pi_e2e_1",
    )

    assert final_state["diagnosis_terminated_reason"] == "concluded"
    assert "fraud_result" in final_state
    assert "inventory_result" in final_state
    assert "customer_context_result" in final_state

    db2 = db_module.SessionLocal()
    persisted_case = db2.get(ExceptionCase, "case-e2e-1")
    assert persisted_case.state == CaseState.DIAGNOSING
    assert persisted_case.diagnosis is not None
    assert persisted_case.diagnosis["terminated_reason"] == "concluded"
    assert persisted_case.fraud_risk_score is not None
    db2.close()

    if os.path.exists(tmp_path):
        os.remove(tmp_path)
