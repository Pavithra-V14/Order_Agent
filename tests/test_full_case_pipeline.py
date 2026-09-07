"""
Tests for app.agents.orchestrator.run_full_case_pipeline - the
previously-missing function that actually completes an auto-execute
routing decision, and the first real code path that calls log_episode().
"""
import os
import tempfile
from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_full_pipeline_{os.getpid()}_{id(object())}.db")
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


@pytest.fixture(autouse=True)
def reset_all():
    os.environ.pop("GROQ_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()
    from app.tools.payment import reset_fake_gateway
    from app.core.circuit_breaker import reset_all_breakers
    reset_fake_gateway()
    reset_all_breakers()
    yield


def test_auto_execute_case_actually_gets_resolved(isolated_db):
    """THE core proof: a genuine auto-execute routing decision now
    actually executes and resolves the case - previously, NOTHING did
    this outside of scripts/full_pipeline_demo.py's workaround."""
    from app.core.db import SessionLocal, ExceptionCase, CaseState
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.tools.payment import get_payment_gateway
    from app.agents.orchestrator import run_full_case_pipeline

    db = SessionLocal()
    create_order(db, order_id="ORD-FULLPIPE-1", customer_id="CUST-FULLPIPE-1", channel="direct",
                 status="payment_failed", total_amount_usd=30.0,
                 purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
                 line_items=[{"sku": "SKU-FULLPIPE-1", "category": "apparel", "qty": 1, "price": 30.0}],
                 payment_intent_id="pi_fullpipe_1")
    get_payment_gateway().seed_transaction("pi_fullpipe_1", amount_usd=30.0, status="declined")
    seed_stock(db, sku="SKU-FULLPIPE-1", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    case = ExceptionCase(id="case-fullpipe-1", order_id="ORD-FULLPIPE-1", customer_id="CUST-FULLPIPE-1",
                          channel="direct", exception_type="payment", state=CaseState.DETECTED)
    db.add(case)
    db.commit()

    result = run_full_case_pipeline(
        db, case_id="case-fullpipe-1", order_id="ORD-FULLPIPE-1", customer_id="CUST-FULLPIPE-1",
        order_amount_usd=30.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0, payment_intent_id="pi_fullpipe_1",
        retrieved_policy_doc_id="RET-POLICY-2025-A", retrieved_policy_version="1",
    )

    assert result["routing"] == "auto_execute"
    assert result["completion"]["outcome"] == "resolved"

    db2 = SessionLocal()
    final_case = db2.get(ExceptionCase, "case-fullpipe-1")
    assert final_case.state == CaseState.RESOLVED
    assert final_case.execution_result["status"] == "executed"
    db2.close()
    db.close()


def test_auto_execute_completion_writes_a_customer_history_episode(isolated_db):
    """THE regression test for the second gap found alongside the first:
    log_episode() must actually get called when a case auto-resolves -
    previously nothing in the real application ever called it at all."""
    from app.core.db import SessionLocal, ExceptionCase, CaseState
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.tools.payment import get_payment_gateway
    from app.agents.orchestrator import run_full_case_pipeline
    from app.memory.episodic import get_customer_history

    db = SessionLocal()
    create_order(db, order_id="ORD-FULLPIPE-2", customer_id="CUST-FULLPIPE-2", channel="direct",
                 status="payment_failed", total_amount_usd=25.0,
                 purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
                 line_items=[{"sku": "SKU-FULLPIPE-2", "category": "apparel", "qty": 1, "price": 25.0}],
                 payment_intent_id="pi_fullpipe_2")
    get_payment_gateway().seed_transaction("pi_fullpipe_2", amount_usd=25.0, status="declined")
    seed_stock(db, sku="SKU-FULLPIPE-2", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    case = ExceptionCase(id="case-fullpipe-2", order_id="ORD-FULLPIPE-2", customer_id="CUST-FULLPIPE-2",
                          channel="direct", exception_type="payment", state=CaseState.DETECTED)
    db.add(case)
    db.commit()

    history_before = get_customer_history(db, "CUST-FULLPIPE-2")
    assert history_before == []

    run_full_case_pipeline(
        db, case_id="case-fullpipe-2", order_id="ORD-FULLPIPE-2", customer_id="CUST-FULLPIPE-2",
        order_amount_usd=25.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0, payment_intent_id="pi_fullpipe_2",
        retrieved_policy_doc_id="RET-POLICY-2025-A", retrieved_policy_version="1",
    )

    history_after = get_customer_history(db, "CUST-FULLPIPE-2")
    assert len(history_after) == 1, (
        "log_episode() must have been called during auto-execute completion"
    )
    assert history_after[0]["episode_type"] == "case_resolved"
    assert history_after[0]["content"]["exception_type"] == "payment"
    db.close()


def test_escalated_case_is_not_completed(isolated_db):
    """The mirror case: when routing genuinely escalates, the case must
    NOT be marked resolved and log_episode must NOT fire yet."""
    from app.core.db import SessionLocal, ExceptionCase, CaseState
    from app.tools.oms import create_order
    from app.agents.orchestrator import run_full_case_pipeline
    from app.memory.episodic import get_customer_history

    db = SessionLocal()
    create_order(db, order_id="ORD-FULLPIPE-3", customer_id="CUST-FULLPIPE-3", channel="direct",
                 status="payment_failed", total_amount_usd=5000.0,
                 purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
                 line_items=[{"sku": "SKU-FULLPIPE-3", "category": "electronics", "qty": 1, "price": 5000.0}],
                 payment_intent_id="pi_fullpipe_3")
    from app.tools.payment import get_payment_gateway
    get_payment_gateway().seed_transaction("pi_fullpipe_3", amount_usd=5000.0, status="declined")

    case = ExceptionCase(id="case-fullpipe-3", order_id="ORD-FULLPIPE-3", customer_id="CUST-FULLPIPE-3",
                          channel="direct", exception_type="payment", state=CaseState.DETECTED)
    db.add(case)
    db.commit()

    result = run_full_case_pipeline(
        db, case_id="case-fullpipe-3", order_id="ORD-FULLPIPE-3", customer_id="CUST-FULLPIPE-3",
        order_amount_usd=5000.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0, payment_intent_id="pi_fullpipe_3",
        retrieved_policy_doc_id="RET-POLICY-2025-A", retrieved_policy_version="1",
    )

    assert result["routing"] in ("escalate", "blocked")
    assert result["completion"] is None

    db2 = SessionLocal()
    final_case = db2.get(ExceptionCase, "case-fullpipe-3")
    assert final_case.state != CaseState.RESOLVED
    db2.close()

    assert get_customer_history(db, "CUST-FULLPIPE-3") == [], (
        "log_episode must NOT fire for a case that hasn't actually resolved yet"
    )
    db.close()
