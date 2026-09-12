"""
Test for a real gap found during a direct audit: app/memory/summary_buffer.py's
SummaryBuffer (architecture doc 8.4's short-term, per-case working
memory) was fully implemented and tested on its own mechanical
behavior, but never actually wired into the real diagnosis loop -
get_or_create_buffer() had zero real call sites anywhere in the
application. This proves run_diagnosis() now genuinely populates it and
returns its final state as part of DiagnosisResult.
"""
import os
import tempfile
from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_summary_buffer_{os.getpid()}_{id(object())}.db")
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


def test_diagnosis_populates_and_returns_a_real_summary_buffer(isolated_db):
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.agents.diagnosis_agent import run_diagnosis
    from app.agents.llm_client import FakeLLMClient
    from app.memory.summary_buffer import _buffers

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-SUMMARY-BUFFER-TEST", customer_id="CUST-SUMMARY-BUFFER-TEST", channel="direct",
        status="paid", total_amount_usd=45.0, purchase_date=datetime.now(timezone.utc),
        line_items=[{"sku": "SKU-SUMMARY-BUFFER-TEST", "category": "apparel", "qty": 1, "price": 45.0}],
    )
    seed_stock(db, sku="SKU-SUMMARY-BUFFER-TEST", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    result = run_diagnosis(
        db, FakeLLMClient(), order_id="ORD-SUMMARY-BUFFER-TEST",
        case_id="case-summary-buffer-test",
    )

    # THE actual proof: case_summary must be genuinely populated, not
    # an empty default — meaning the buffer was actually written to
    # during the real diagnosis loop, not just instantiated and ignored.
    assert result.case_summary
    assert result.case_summary["case_id"] == "case-summary-buffer-test"
    assert len(result.case_summary["recent_items"]) > 0, (
        "the buffer's recent_items must contain real recorded steps from this diagnosis run"
    )
    # Also verify the module-level registry genuinely has this case's
    # buffer, not just a return value constructed separately.
    assert "case-summary-buffer-test" in _buffers
    db.close()


def test_summary_buffer_resets_between_separate_diagnosis_runs_for_the_same_case(isolated_db):
    """A reopened case re-running diagnosis must not inherit a stale
    summary from a previous, unrelated run — proves the reset-at-start
    behavior actually works."""
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.agents.diagnosis_agent import run_diagnosis
    from app.agents.llm_client import FakeLLMClient

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-SUMMARY-RESET-TEST", customer_id="CUST-SUMMARY-RESET-TEST", channel="direct",
        status="paid", total_amount_usd=45.0, purchase_date=datetime.now(timezone.utc),
        line_items=[{"sku": "SKU-SUMMARY-RESET-TEST", "category": "apparel", "qty": 1, "price": 45.0}],
    )
    seed_stock(db, sku="SKU-SUMMARY-RESET-TEST", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    result1 = run_diagnosis(db, FakeLLMClient(), order_id="ORD-SUMMARY-RESET-TEST", case_id="case-summary-reset-test")
    first_run_item_count = len(result1.case_summary["recent_items"])

    result2 = run_diagnosis(db, FakeLLMClient(), order_id="ORD-SUMMARY-RESET-TEST", case_id="case-summary-reset-test")
    second_run_item_count = len(result2.case_summary["recent_items"])

    assert second_run_item_count == first_run_item_count, (
        "a second diagnosis run for the SAME case must produce the same number of recorded items as "
        "the first, not an accumulating count — proving the buffer resets rather than carrying over "
        "stale items from the previous run"
    )
    db.close()


def test_case_summary_is_genuinely_queryable_via_the_audit_log(isolated_db):
    """THE regression test proving case_summary isn't just computed and
    discarded — it must actually reach the persisted audit log, where a
    real person reviewing a case (or the Audit Log page) can see it."""
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.core.db import ExceptionCase, CaseState, AuditLogEntry
    from app.agents.orchestrator import run_orchestrator_for_case
    from app.agents.llm_client import FakeLLMClient

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-SUMMARY-AUDIT-TEST", customer_id="CUST-SUMMARY-AUDIT-TEST", channel="direct",
        status="paid", total_amount_usd=45.0, purchase_date=datetime.now(timezone.utc),
        line_items=[{"sku": "SKU-SUMMARY-AUDIT-TEST", "category": "apparel", "qty": 1, "price": 45.0}],
    )
    seed_stock(db, sku="SKU-SUMMARY-AUDIT-TEST", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)
    case = ExceptionCase(id="case-summary-audit-test", order_id="ORD-SUMMARY-AUDIT-TEST",
                          customer_id="CUST-SUMMARY-AUDIT-TEST", channel="direct",
                          exception_type="payment", state=CaseState.DETECTED)
    db.add(case)
    db.commit()

    run_orchestrator_for_case(
        case_id="case-summary-audit-test", order_id="ORD-SUMMARY-AUDIT-TEST",
        customer_id="CUST-SUMMARY-AUDIT-TEST", llm=FakeLLMClient(),
    )

    entry = db.query(AuditLogEntry).filter(
        AuditLogEntry.case_id == "case-summary-audit-test", AuditLogEntry.action == "diagnosis_complete",
    ).first()
    assert entry is not None
    assert "case_summary" in entry.detail
    assert entry.detail["case_summary"]["recent_items"]
    db.close()
