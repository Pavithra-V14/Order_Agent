"""
Test for a real production bug: a real Groq LLM called check_carrier 7
TIMES in a row despite already having a clear "returned" status from
the first call, never reaching "conclude" and burning the entire step
ceiling. This proves the defensive backstop in run_diagnosis() actually
skips the wasted real API call when a planner re-requests already-
fetched data, regardless of why it asked again.
"""
from datetime import datetime, timezone

import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_redundant_check_{os.getpid()}_{id(object())}.db")
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


def test_redundant_carrier_check_is_skipped_not_re_executed(isolated_db):
    """Simulates exactly the reported bug: a planner that keeps
    requesting check_carrier even after that data already exists in
    findings. Proves the actual carrier gateway is only ever called
    ONCE, not repeatedly, regardless of how many times the planner asks."""
    from app.core.db import SessionLocal
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.tools.carrier import get_carrier_gateway
    from app.agents.diagnosis_agent import run_diagnosis
    from app.agents.llm_client import BaseLLMClient, DiagnosisStepPlan

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-REDUNDANT-CHECK", customer_id="CUST-REDUNDANT-CHECK", channel="direct",
        status="shipped", total_amount_usd=45.0,
        purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
        line_items=[{"sku": "SKU-REDUNDANT-CHECK", "category": "apparel", "qty": 1, "price": 45.0}],
    )
    seed_stock(db, sku="SKU-REDUNDANT-CHECK", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    tracking_number = "TRACK-REDUNDANT-CHECK"
    get_carrier_gateway().seed_tracking(tracking_number, "returned")

    call_count = {"carrier_calls": 0}
    real_get_tracking_status = get_carrier_gateway().get_tracking_status

    def counting_get_tracking_status(*args, **kwargs):
        call_count["carrier_calls"] += 1
        return real_get_tracking_status(*args, **kwargs)

    get_carrier_gateway().get_tracking_status = counting_get_tracking_status

    class StubbornLLMClient(BaseLLMClient):
        """Simulates the real reported bug: keeps re-requesting
        check_carrier even after that data already exists, never
        concluding on its own — exactly what a real Groq model did."""
        def plan_next_diagnosis_step(self, case_context, findings_so_far):
            if "order" not in findings_so_far:
                return DiagnosisStepPlan(action="check_order", reasoning="need order first")
            if "inventory" not in findings_so_far:
                return DiagnosisStepPlan(action="check_inventory", reasoning="need inventory")
            # Always asks for carrier again, no matter what — the exact
            # stubborn/looping behavior that caused the real bug.
            return DiagnosisStepPlan(action="check_carrier", reasoning="checking carrier again")

        def assess_fraud_risk(self, case_context, customer_risk_profile):
            return {"risk_score": 0.0, "flag": False, "reasoning": "not used in this test"}

    try:
        result = run_diagnosis(
            db, StubbornLLMClient(), order_id="ORD-REDUNDANT-CHECK",
            tracking_number=tracking_number, max_steps=8,
        )
    finally:
        get_carrier_gateway().get_tracking_status = real_get_tracking_status

    assert call_count["carrier_calls"] == 1, (
        f"the real carrier gateway must only be called ONCE even though the planner requested "
        f"check_carrier repeatedly — got {call_count['carrier_calls']} real calls, meaning the "
        f"redundant-check guard did not actually prevent the wasted repeated API calls"
    )
    assert result.terminated_reason == "max_steps_reached"
    assert "carrier" in result.findings
    db.close()
