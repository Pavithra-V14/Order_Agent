"""
Regression test for a real bug found while manually verifying Stage 3's
similar_past_cases in the actual case-detail UI: orchestrator.py's
aggregate step adds similar_past_cases onto case.resolution_decision,
but app/agents/resolution_completion.py's complete_resolution() then
REASSIGNED the whole resolution_decision dict from final_decision alone
- which has no knowledge of similar_past_cases - silently discarding it
for EVERY resolved case, auto-executed or otherwise. Confirmed by
directly instrumenting both call sites (not assumed) before fixing.
"""
import os
import tempfile
from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_similar_cases_survive_{os.getpid()}_{id(object())}.db")
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


def test_complete_resolution_preserves_similar_past_cases_already_on_the_case(isolated_db):
    """Unit-level reproduction, isolated from the rest of the real
    pipeline: seed a case whose resolution_decision ALREADY has
    similar_past_cases (simulating what orchestrator.py's aggregate
    step does), then call complete_resolution() directly and confirm
    that key survives the reassignment, not just the fields
    final_decision itself carries."""
    from app.core.db import ExceptionCase, CaseState
    from app.tools.oms import create_order
    from app.tools.payment import get_payment_gateway
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy
    from app.agents.resolution_completion import complete_resolution

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-SIMILAR-SURVIVE", customer_id="CUST-SIMILAR-SURVIVE", channel="direct",
        status="paid", total_amount_usd=30.0, purchase_date=datetime.now(timezone.utc),
        line_items=[{"sku": "SKU-1", "category": "apparel", "qty": 1, "price": 30.0}],
        payment_intent_id="pi_similar_survive_test",
    )
    get_payment_gateway().seed_transaction("pi_similar_survive_test", amount_usd=30.0, status="succeeded")

    fake_similar_past_cases = [
        {"case_id": "case-history-0", "case_feature_summary": "root_causes=[...], fraud_flag=False",
         "human_final_resolution": {"action": "refund", "amount_usd": 30.0}, "similarity": 1.0},
    ]
    case = ExceptionCase(
        id="case-similar-survive", order_id="ORD-SIMILAR-SURVIVE", customer_id="CUST-SIMILAR-SURVIVE",
        channel="direct", exception_type="return", state=CaseState.DETECTED,
        resolution_decision={"similar_past_cases": fake_similar_past_cases},  # pre-populated, as orchestrator.py does
    )
    db.add(case)
    db.commit()

    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=30.0, confidence=0.9,
        reasoning="Test reasoning long enough to pass schema validation.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )

    complete_resolution(
        db, case=case, proposed_decision=decision, final_decision=decision,
        decided_by="system:test", action_label="test_action", payment_intent_id="pi_similar_survive_test",
    )

    updated_case = db.get(ExceptionCase, "case-similar-survive")
    assert updated_case.resolution_decision["similar_past_cases"] == fake_similar_past_cases, (
        "complete_resolution() must preserve similar_past_cases already present on "
        "case.resolution_decision, not discard it when reassigning the decision fields"
    )
    # The decision's own real fields must ALSO be present - this isn't
    # just "never overwrite," it's "merge the two correctly."
    assert updated_case.resolution_decision["action"] == "refund"
    assert updated_case.resolution_decision["amount_usd"] == 30.0


def test_complete_resolution_does_not_add_similar_past_cases_key_when_none_existed(isolated_db):
    """The other half of the contract: a case with NO similar_past_cases
    (e.g. no resolution history existed yet) must not have the key
    fabricated out of nowhere."""
    from app.core.db import ExceptionCase, CaseState
    from app.tools.oms import create_order
    from app.tools.payment import get_payment_gateway
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy
    from app.agents.resolution_completion import complete_resolution

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-NO-SIMILAR", customer_id="CUST-NO-SIMILAR", channel="direct",
        status="paid", total_amount_usd=30.0, purchase_date=datetime.now(timezone.utc),
        line_items=[{"sku": "SKU-1", "category": "apparel", "qty": 1, "price": 30.0}],
        payment_intent_id="pi_no_similar_test",
    )
    get_payment_gateway().seed_transaction("pi_no_similar_test", amount_usd=30.0, status="succeeded")

    case = ExceptionCase(
        id="case-no-similar", order_id="ORD-NO-SIMILAR", customer_id="CUST-NO-SIMILAR",
        channel="direct", exception_type="return", state=CaseState.DETECTED,
    )
    db.add(case)
    db.commit()

    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=30.0, confidence=0.9,
        reasoning="Test reasoning long enough to pass schema validation.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )

    complete_resolution(
        db, case=case, proposed_decision=decision, final_decision=decision,
        decided_by="system:test", action_label="test_action", payment_intent_id="pi_no_similar_test",
    )

    updated_case = db.get(ExceptionCase, "case-no-similar")
    assert "similar_past_cases" not in updated_case.resolution_decision
