"""
Manual-check follow-up: GET /api/v1/cases/{case_id} previously exposed
fraud_risk_score/fraud_flag but not the underlying reasons (only ever
visible via a second round-trip to /audit-log), and never
working_memory_summary at all (not exposed anywhere). This closes both
gaps.

Found while writing these tests: app/core/db.py had working_memory_summary
declared TWICE in ExceptionCase's class body (with two different,
partially-contradictory docstrings) - harmless at runtime (the second
declaration simply overwrote the first, same table/column either way),
but real evidence this exact area was touched by two separate,
uncoordinated changes. Fixed by merging into one accurate declaration
before writing these tests - not left as found.
"""
import os
import tempfile
from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_case_response_fields_{os.getpid()}_{id(object())}.db")
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


def _make_admin_key(db_module):
    from app.core.db import ApiKeyRecord
    from app.core.auth import hash_api_key, generate_api_key
    raw_key = generate_api_key()
    db = db_module.SessionLocal()
    db.add(ApiKeyRecord(key_hash=hash_api_key(raw_key), name="test-admin", role="admin"))
    db.commit()
    return raw_key


def test_case_response_includes_fraud_reasons(isolated_db):
    from app.core.db import ExceptionCase, CaseState
    db = isolated_db.SessionLocal()
    db.add(ExceptionCase(
        id="case-fraud-reasons-api", order_id="ORD-1", customer_id="CUST-1",
        channel="direct", exception_type="fraud", state=CaseState.RESOLVED,
        fraud_risk_score=0.9, fraud_flag="flagged",
        fraud_reasons=["prior fraud flag", "same-day address change"],
    ))
    db.commit()

    raw_key = _make_admin_key(isolated_db)
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/cases/case-fraud-reasons-api", headers={"X-API-Key": raw_key})

    assert resp.status_code == 200
    body = resp.json()
    assert body["fraud_reasons"] == ["prior fraud flag", "same-day address change"]


def test_case_response_includes_working_memory_summary(isolated_db):
    from app.core.db import ExceptionCase, CaseState
    db = isolated_db.SessionLocal()
    db.add(ExceptionCase(
        id="case-working-memory-api", order_id="ORD-2", customer_id="CUST-2",
        channel="direct", exception_type="return", state=CaseState.DIAGNOSING,
        working_memory_summary={
            "case_id": "case-working-memory-api", "running_summary": "folded summary text",
            "recent_items": [{"agent": "diagnosis", "summary": "checked payment"}],
        },
    ))
    db.commit()

    raw_key = _make_admin_key(isolated_db)
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/cases/case-working-memory-api", headers={"X-API-Key": raw_key})

    assert resp.status_code == 200
    body = resp.json()
    assert body["working_memory_summary"]["running_summary"] == "folded summary text"
    assert len(body["working_memory_summary"]["recent_items"]) == 1


def test_case_response_fields_are_null_when_never_set(isolated_db):
    """A case that never went through fraud scoring or diagnosis (e.g.
    freshly created) must return null for both, not error or omit the
    keys entirely - the API contract must be stable regardless of case
    lifecycle stage."""
    from app.core.db import ExceptionCase, CaseState
    db = isolated_db.SessionLocal()
    db.add(ExceptionCase(
        id="case-fresh-no-fields", order_id="ORD-3", customer_id="CUST-3",
        channel="direct", exception_type="return", state=CaseState.DETECTED,
    ))
    db.commit()

    raw_key = _make_admin_key(isolated_db)
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/cases/case-fresh-no-fields", headers={"X-API-Key": raw_key})

    assert resp.status_code == 200
    body = resp.json()
    assert body["fraud_reasons"] is None
    assert body["working_memory_summary"] is None


def test_orchestrator_persists_fraud_reasons_onto_the_case(isolated_db):
    """THE actual wiring test: aggregate_node must set case.fraud_reasons
    from fraud_result.reasons, not just fraud_risk_score/fraud_flag -
    otherwise the API field above would always be null in a real run."""
    from app.core.db import ExceptionCase, CaseState
    db = isolated_db.SessionLocal()
    db.add(ExceptionCase(
        id="case-orchestrator-fraud-reasons", order_id="ORD-4", customer_id="CUST-4",
        channel="direct", exception_type="fraud", state=CaseState.DIAGNOSING,
    ))
    db.commit()

    from app.agents.orchestrator import make_aggregate_node
    aggregate_node = make_aggregate_node()
    aggregate_node({
        "case_id": "case-orchestrator-fraud-reasons", "order_id": "ORD-4", "customer_id": "CUST-4",
        "diagnosis_findings": {}, "diagnosis_root_causes": [], "diagnosis_terminated_reason": "concluded",
        "fraud_result": {"risk_score": 0.9, "flag": True, "reasons": ["prior fraud flag"]},
        "inventory_result": {}, "customer_context_result": {},
    })

    updated_case = db.get(ExceptionCase, "case-orchestrator-fraud-reasons")
    assert updated_case.fraud_reasons == ["prior fraud flag"]


def test_orchestrator_leaves_fraud_reasons_null_when_not_flagged(isolated_db):
    from app.core.db import ExceptionCase, CaseState
    db = isolated_db.SessionLocal()
    db.add(ExceptionCase(
        id="case-orchestrator-no-fraud-reasons", order_id="ORD-5", customer_id="CUST-5",
        channel="direct", exception_type="return", state=CaseState.DIAGNOSING,
    ))
    db.commit()

    from app.agents.orchestrator import make_aggregate_node
    aggregate_node = make_aggregate_node()
    aggregate_node({
        "case_id": "case-orchestrator-no-fraud-reasons", "order_id": "ORD-5", "customer_id": "CUST-5",
        "diagnosis_findings": {}, "diagnosis_root_causes": [], "diagnosis_terminated_reason": "concluded",
        "fraud_result": {"risk_score": 0.1, "flag": False, "reasons": []},
        "inventory_result": {}, "customer_context_result": {},
    })

    updated_case = db.get(ExceptionCase, "case-orchestrator-no-fraud-reasons")
    assert updated_case.fraud_reasons is None
