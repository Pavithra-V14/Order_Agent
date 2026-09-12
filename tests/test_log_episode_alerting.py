"""
Test for a real production gap: a log_episode() failure (e.g. a real
Pydantic validation error inside Graphiti's own internal LLM call) was
previously only ever a warning line in a log file - correctly non-fatal
to the case, but invisible to anyone not actively watching logs. This
proves it now produces a real, queryable AlertRecord too, using the
same alerting mechanism already in place for circuit breaker trips.
"""
import os
import tempfile
from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_log_episode_alert_{os.getpid()}_{id(object())}.db")
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


def test_log_episode_failure_produces_a_real_alert_not_just_a_log_line(isolated_db):
    from app.tools.oms import create_order
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy
    from app.agents.resolution_completion import complete_resolution
    from app.core.alerting import get_recent_alerts

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-LOG-EPISODE-ALERT-TEST", customer_id="CUST-LOG-EPISODE-ALERT-TEST", channel="direct",
        status="paid", total_amount_usd=45.0,
        purchase_date=datetime.now(timezone.utc),
        line_items=[{"sku": "SKU-LOG-EPISODE-ALERT-TEST", "category": "apparel", "qty": 1, "price": 45.0}],
        payment_intent_id="pi_log_episode_alert_test",
    )
    from app.tools.payment import get_payment_gateway
    get_payment_gateway().seed_transaction("pi_log_episode_alert_test", amount_usd=45.0, status="succeeded")

    case = isolated_db.ExceptionCase(
        id="case-log-episode-alert-test", order_id="ORD-LOG-EPISODE-ALERT-TEST",
        customer_id="CUST-LOG-EPISODE-ALERT-TEST", channel="direct",
        exception_type="return", state=isolated_db.CaseState.DETECTED,
    )
    db.add(case)
    db.commit()

    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=45.0, confidence=0.9,
        reasoning="Test reasoning long enough to pass schema validation.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )

    import app.agents.resolution_completion as rc_module

    def always_fails(*args, **kwargs):
        raise ValueError("simulated real Pydantic validation error: Field required: summaries")

    original_log_episode = rc_module.log_episode
    rc_module.log_episode = always_fails
    try:
        result = complete_resolution(
            db, case=case, proposed_decision=decision, final_decision=decision,
            decided_by="system:test", action_label="test_action",
            payment_intent_id="pi_log_episode_alert_test",
        )
    finally:
        rc_module.log_episode = original_log_episode

    # The case must still resolve correctly - non-fatal, matching the
    # existing, correct behavior.
    assert result["outcome"] == "resolved"

    # But it must ALSO be visible as a real, queryable alert now - not
    # just a warning line in a log file.
    alerts = get_recent_alerts(db, event_type="log_episode_failure")
    assert len(alerts) == 1, "a real AlertRecord must exist for this failure, not just a log line"
    assert alerts[0]["detail"]["case_id"] == "case-log-episode-alert-test"
    assert "summaries" in alerts[0]["detail"]["error"]
    db.close()
