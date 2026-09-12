"""
Test for a real gap found during a direct audit:
app/agents/learning_loop.py's retrieve_similar_past_resolutions() -
dynamic few-shot retrieval, per architecture doc 8.5 - was fully
implemented but never actually called from anywhere. Since the real
Resolution-Policy Workflow is rule-based, not LLM-driven, the most
directly valuable wiring is surfacing similar past cases as CONTEXT for
a human reviewer during escalation (the stated goal: "improves
consistency"), not altering the decision itself.
"""
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_few_shot_{os.getpid()}_{id(object())}.db")
    import app.core.db as db_module
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    fresh_engine = create_engine(f"sqlite:///{tmp_path}", connect_args={"check_same_thread": False})
    db_module.Base.metadata.create_all(bind=fresh_engine)
    db_module.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=fresh_engine)
    db_module.engine = fresh_engine

    yield db_module

    fresh_engine.dispose()
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except PermissionError:
        pass


def test_resolution_result_surfaces_genuinely_similar_past_cases(isolated_db):
    """THE regression test: seeds real past resolution outcomes, then
    proves a new resolution run actually retrieves and surfaces the
    genuinely similar ones — not an empty list, and not just any past
    case regardless of similarity."""
    from app.agents.learning_loop import record_resolution_outcome
    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow

    db = isolated_db.SessionLocal()

    # A genuinely similar past case (same kind of root cause) ...
    record_resolution_outcome(
        db, case_id="case-past-similar-1", cluster_key="payment_direct",
        case_feature_summary="root_causes=['payment_issue: transaction declined'], fraud_flag=False",
        agent_proposed_resolution={"action": "deny", "amount_usd": 0.0},
        human_final_resolution={"action": "deny", "amount_usd": 0.0},
    )
    # ... and a genuinely DIFFERENT one (inventory, not payment) that
    # should rank lower for a payment-issue query.
    record_resolution_outcome(
        db, case_id="case-past-different-1", cluster_key="inventory_direct",
        case_feature_summary="root_causes=['inventory_issue: insufficient stock'], fraud_flag=False",
        agent_proposed_resolution={"action": "partial_credit", "amount_usd": 20.0},
        human_final_resolution={"action": "partial_credit", "amount_usd": 20.0},
    )
    db.commit()

    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: transaction declined"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=45.0, fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90, auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A", retrieved_policy_version="1",
        payment_status="requires_payment_method",
        db=db, case_id="case-new-payment-issue",
    )

    assert result.similar_past_cases, "must retrieve at least one similar past case, not an empty list"
    retrieved_ids = [c["case_id"] for c in result.similar_past_cases]
    assert "case-past-similar-1" in retrieved_ids, (
        "the genuinely similar past payment-issue case must be retrieved"
    )
    # The most similar case should rank first (highest similarity).
    top_case = result.similar_past_cases[0]
    assert top_case["case_id"] == "case-past-similar-1"
    assert top_case["human_final_resolution"]["action"] == "deny"
    db.close()


def test_resolution_result_has_no_similar_cases_when_no_history_exists():
    """With no past resolution history at all, this must return an
    empty list cleanly — not crash, and not fabricate results."""
    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow
    from app.core.db import SessionLocal, init_db

    init_db()
    db = SessionLocal()

    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: transaction declined"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=45.0, fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90, auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A", retrieved_policy_version="1",
        payment_status="requires_payment_method",
        db=db, case_id="case-no-history-test",
    )

    assert result.similar_past_cases == []
    db.close()


def test_few_shot_retrieval_never_alters_the_actual_decision():
    """THE regression test for the deliberate design constraint: even
    with genuinely similar past cases resolved differently than the
    current rule-based decision would propose, the decision itself
    must be UNCHANGED — few-shot retrieval is context only, never a
    silent override."""
    from app.agents.learning_loop import record_resolution_outcome
    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow
    from app.core.db import SessionLocal, init_db

    init_db()
    db = SessionLocal()
    record_resolution_outcome(
        db, case_id="case-past-refund-instead", cluster_key="payment_direct",
        case_feature_summary="root_causes=['payment_issue: transaction declined'], fraud_flag=False",
        agent_proposed_resolution={"action": "refund", "amount_usd": 999.0},
        human_final_resolution={"action": "refund", "amount_usd": 999.0},
    )
    db.commit()

    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: transaction declined"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=45.0, fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90, auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A", retrieved_policy_version="1",
        payment_status="requires_payment_method",  # correctly leads to DENY, per the real business rule
        db=db, case_id="case-decision-unaltered-test",
    )

    # The real rule (a never-charged payment can't be refunded) must
    # still apply, regardless of what a similar past case did.
    assert result.decision.action.value == "deny"
    db.close()


def test_similar_past_cases_reach_the_escalations_api(isolated_db):
    """THE regression test proving this isn't just computed and
    discarded — it must actually reach case.resolution_decision, where
    the existing Escalations API/UI already reads from, giving a human
    reviewer real visibility with zero new endpoints needed."""
    from datetime import datetime, timezone
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.tools.payment import get_payment_gateway
    from app.core.db import ExceptionCase, CaseState
    from app.agents.learning_loop import record_resolution_outcome
    from app.agents.orchestrator import run_full_case_pipeline

    db = isolated_db.SessionLocal()

    record_resolution_outcome(
        db, case_id="case-past-for-api-test", cluster_key="payment_direct",
        case_feature_summary="root_causes=['payment_issue: payment method required or declined'], fraud_flag=False",
        agent_proposed_resolution={"action": "deny", "amount_usd": 0.0},
        human_final_resolution={"action": "deny", "amount_usd": 0.0},
    )
    db.commit()

    create_order(
        db, order_id="ORD-FEWSHOT-API-TEST", customer_id="CUST-FEWSHOT-API-TEST", channel="direct",
        status="payment_failed", total_amount_usd=45.0, purchase_date=datetime.now(timezone.utc),
        line_items=[{"sku": "SKU-FEWSHOT-API-TEST", "category": "apparel", "qty": 1, "price": 45.0}],
        payment_intent_id="pi_fewshot_api_test",
    )
    get_payment_gateway().seed_transaction("pi_fewshot_api_test", amount_usd=45.0, status="declined")
    seed_stock(db, sku="SKU-FEWSHOT-API-TEST", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    case = isolated_db.ExceptionCase(
        id="case-fewshot-api-test", order_id="ORD-FEWSHOT-API-TEST", customer_id="CUST-FEWSHOT-API-TEST",
        channel="direct", exception_type="payment", state=isolated_db.CaseState.DETECTED,
    )
    db.add(case)
    db.commit()

    run_full_case_pipeline(
        db, case_id="case-fewshot-api-test", order_id="ORD-FEWSHOT-API-TEST", customer_id="CUST-FEWSHOT-API-TEST",
        order_amount_usd=45.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0, payment_intent_id="pi_fewshot_api_test",
        retrieved_policy_doc_id="RET-POLICY-2025-A", retrieved_policy_version="1",
    )

    refreshed_case = db.get(ExceptionCase, "case-fewshot-api-test")
    assert refreshed_case.resolution_decision is not None
    assert "similar_past_cases" in refreshed_case.resolution_decision, (
        "similar_past_cases must be embedded in the same resolution_decision JSON the Escalations "
        "API already exposes — proving a human reviewer can actually see this, not just that it "
        "was computed internally and discarded"
    )
    db.close()
