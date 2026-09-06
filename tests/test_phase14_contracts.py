"""
Phase 14: contract tests between agents. Each test asserts that one
agent's OUTPUT shape is exactly what the NEXT agent in the pipeline
expects as INPUT - catching the class of bug where two independently-
correct agents disagree about a field name or type.
"""
import os
import tempfile
from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), "test_phase14_contracts.db")
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


def test_diagnosis_result_root_causes_contract_matches_resolution_workflow_input(isolated_db):
    """Contract: DiagnosisResult.root_causes is a list[str] of the exact
    shape that resolution_policy_workflow.propose_resolution_decision()
    pattern-matches against via substring checks. If Diagnosis Agent's
    output format ever changes, this test catches the mismatch
    immediately rather than silently making every resolution decision
    fall through to the default branch."""
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.tools.payment import get_payment_gateway
    from app.agents.llm_client import FakeLLMClient
    from app.agents.diagnosis_agent import run_diagnosis
    from app.agents.resolution_policy_workflow import propose_resolution_decision

    db = isolated_db.SessionLocal()
    create_order(db, order_id="ORD-CONTRACT-1", customer_id="CUST-1", channel="direct",
                 status="payment_failed", total_amount_usd=50.0,
                 purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
                 line_items=[{"sku": "SKU-1", "category": "apparel", "qty": 1, "price": 50.0}])
    get_payment_gateway().seed_transaction("pi_contract_1", amount_usd=50.0, status="declined")
    seed_stock(db, sku="SKU-1", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    diagnosis_result = run_diagnosis(db, FakeLLMClient(), order_id="ORD-CONTRACT-1",
                                      payment_intent_id="pi_contract_1")

    assert isinstance(diagnosis_result.root_causes, list)
    for cause in diagnosis_result.root_causes:
        assert isinstance(cause, str), f"resolution_policy_workflow expects str root causes, got {type(cause)}"
        assert any(cause.startswith(prefix) for prefix in
                   ("payment_issue:", "inventory_issue:", "carrier_issue:", "no_anomaly_detected:",
                    "diagnosis_incomplete:", "diagnosis_timeout:")), (
            f"root cause {cause!r} doesn't match any prefix propose_resolution_decision() checks for"
        )

    decision = propose_resolution_decision(
        diagnosis_root_causes=diagnosis_result.root_causes,
        inventory_result={"any_shortfall": False},
        order_amount_usd=50.0,
    )
    assert decision.action.value == "refund"
    db.close()


def test_inventory_agent_output_contract_matches_resolution_workflow_input():
    """Contract: run_inventory_agent()'s output has 'any_shortfall' (bool)
    at the top level - propose_resolution_decision() reads exactly this
    key."""
    from app.agents.workflow_agents import run_inventory_agent
    from app.agents.resolution_policy_workflow import propose_resolution_decision

    tmp_path = os.path.join(tempfile.gettempdir(), "test_phase14_inv_contract.db")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}"
    from app.core.config import get_settings
    get_settings.cache_clear()
    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)
    db_module.init_db()

    from app.tools.wms import seed_stock
    db = db_module.SessionLocal()
    seed_stock(db, sku="SKU-CONTRACT-2", warehouse="WH-A", on_hand_qty=1, sellable_qty=1)

    inventory_result = run_inventory_agent(db, line_items=[{"sku": "SKU-CONTRACT-2", "qty": 3}])
    assert "any_shortfall" in inventory_result
    assert isinstance(inventory_result["any_shortfall"], bool)

    decision = propose_resolution_decision(
        diagnosis_root_causes=["inventory_issue: SKU-CONTRACT-2 has insufficient sellable stock"],
        inventory_result=inventory_result,
        order_amount_usd=30.0,
    )
    assert decision.action.value == "partial_credit"
    db.close()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_resolution_result_contract_matches_execution_agent_input(isolated_db):
    """Contract: ResolutionResult.decision is a ResolutionDecision object
    (not a dict) - execution_agent.execute_resolution() calls
    decision.action, decision.amount_usd as attributes, not dict keys."""
    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow
    from app.agents.execution_agent import execute_resolution
    from app.tools.payment import get_payment_gateway
    from app.guardrails.schema import ResolutionDecision

    db = isolated_db.SessionLocal()
    get_payment_gateway().seed_transaction("pi_contract_3", amount_usd=30.0)

    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: transaction status is 'declined'"],
        inventory_result={"any_shortfall": False}, order_amount_usd=30.0,
        fraud_flag_present=False, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0, retrieved_policy_doc_id="RET-POLICY-2025-A",
        retrieved_policy_version="1",
    )
    assert isinstance(result.decision, ResolutionDecision), (
        "execute_resolution() accesses decision.action/.amount_usd as attributes - "
        "a dict here would fail with AttributeError, not a clean error"
    )

    exec_result = execute_resolution(db, case_id="case-contract-3", decision=result.decision,
                                      payment_intent_id="pi_contract_3")
    assert exec_result.status.value == "executed"
    db.close()
