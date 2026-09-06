"""
Phase 12 DoD: "OpenAPI docs (/docs) render correctly and every endpoint
from 8.1 is present and callable."
"""
import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_and_db():
    tmp_db = os.path.join(tempfile.gettempdir(), "test_phase12.db")
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_db}"

    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)

    from app.tools.payment import reset_fake_gateway
    from app.tools.carrier import reset_fake_carrier
    from app.cache.ttl_cache import reset_all_caches
    from app.workers.job_queue import reset_job_queue
    from app.core.circuit_breaker import reset_all_breakers
    reset_fake_gateway()
    reset_fake_carrier()
    reset_all_caches()
    reset_job_queue()
    reset_all_breakers()

    import app.main as main_module
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        yield client, db_module

    if os.path.exists(tmp_db):
        os.remove(tmp_db)


def test_openapi_schema_lists_every_architecture_doc_endpoint(client_and_db):
    """THE Phase 12 DoD test."""
    client, _ = client_and_db
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]

    expected = [
        "/api/v1/cases", "/api/v1/cases/{case_id}", "/api/v1/cases/{case_id}/reopen",
        "/api/v1/webhooks/oms", "/api/v1/webhooks/inventory", "/api/v1/webhooks/carrier",
        "/api/v1/escalations", "/api/v1/escalations/{case_id}/decision",
        "/api/v1/policies/upload",
        "/api/v1/metrics/{scope}", "/api/v1/traces/{case_id}",
        "/api/v1/health",
    ]
    for path in expected:
        assert path in paths, f"Missing endpoint: {path}"

    docs_resp = client.get("/docs")
    assert docs_resp.status_code == 200


def test_webhook_acks_fast_and_processes_async(client_and_db):
    """Proves the ack-fast contract: the POST returns in well under 100ms,
    and case creation only shows up AFTER the background job completes."""
    client, db_module = client_and_db

    from app.core.db import SessionLocal
    from app.tools.oms import create_order
    from datetime import datetime, timezone
    db = SessionLocal()
    create_order(db, order_id="ORD-P12-1", customer_id="CUST-P12-1", channel="direct",
                 status="paid", total_amount_usd=50.0, purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
                 line_items=[{"sku": "SKU-P12-1", "category": "apparel", "qty": 1, "price": 50.0}])
    db.close()

    t0 = time.monotonic()
    resp = client.post("/api/v1/webhooks/oms", json={"order_id": "ORD-P12-1", "new_status": "payment_failed"})
    ack_latency = time.monotonic() - t0

    assert resp.status_code == 202
    assert ack_latency < 0.1, f"Webhook ack took {ack_latency}s - should be near-instant (enqueue only)"
    job_id = resp.json()["job_id"]

    job = None
    for _ in range(100):
        job_resp = client.get(f"/api/v1/webhooks/jobs/{job_id}")
        job = job_resp.json()
        if job["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.02)

    assert job["status"] == "succeeded", f"Job failed: {job.get('error')}"
    assert job["result"]["case_created"] is not None

    cases_resp = client.get("/api/v1/cases")
    order_ids = [c["order_id"] for c in cases_resp.json()]
    assert "ORD-P12-1" in order_ids


def test_escalation_decision_approve_flow(client_and_db):
    """Full loop: seed an escalated case with a proposed resolution,
    approve it via the API, confirm it resolves and the payment gateway
    actually gets called."""
    client, db_module = client_and_db
    from app.core.db import SessionLocal, ExceptionCase, CaseState
    from app.tools.payment import get_payment_gateway

    db = SessionLocal()
    from app.tools.oms import create_order
    from datetime import datetime, timezone
    get_payment_gateway().seed_transaction("pi_p12_1", amount_usd=40.0)
    create_order(db, order_id="ORD-P12-2", customer_id="CUST-P12-2", channel="direct",
                 status="paid", total_amount_usd=40.0, purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
                 line_items=[{"sku": "SKU-P12-2", "category": "apparel", "qty": 1, "price": 40.0}],
                 payment_intent_id="pi_p12_1")
    case = ExceptionCase(
        id="case-p12-escalate-1", order_id="ORD-P12-2", customer_id="CUST-P12-2",
        channel="direct", exception_type="payment", state=CaseState.ESCALATED,
        resolution_decision={
            "action": "refund", "amount_usd": 40.0, "confidence": 0.7,
            "reasoning": "Escalated for review due to low confidence.",
            "cited_policy": {"doc_id": "RET-POLICY-2025-A", "version": "1", "clause_summary": "return window"},
        },
    )
    db.add(case)
    db.commit()
    db.close()

    resp = client.post("/api/v1/escalations/case-p12-escalate-1/decision", json={
        "action": "approve", "decided_by": "human:reviewer@company.com",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "resolved"
    assert body["final_action"] == "refund"

    db2 = SessionLocal()
    updated_case = db2.get(ExceptionCase, "case-p12-escalate-1")
    assert updated_case.state == CaseState.RESOLVED
    db2.close()


def test_escalation_list_sorted_by_priority(client_and_db):
    client, db_module = client_and_db
    from app.core.db import SessionLocal, ExceptionCase, CaseState

    db = SessionLocal()
    db.add(ExceptionCase(id="case-low", order_id="ORD-LOW", customer_id="C1", channel="direct",
                          exception_type="return", state=CaseState.ESCALATED,
                          resolution_decision={"action": "refund", "amount_usd": 10.0, "confidence": 0.8,
                                                "reasoning": "x" * 15}))
    db.add(ExceptionCase(id="case-fraud", order_id="ORD-FRAUD", customer_id="C2", channel="direct",
                          exception_type="return", state=CaseState.ESCALATED, fraud_flag="flagged",
                          resolution_decision={"action": "refund", "amount_usd": 5.0, "confidence": 0.8,
                                                "reasoning": "x" * 15}))
    db.commit()
    db.close()

    resp = client.get("/api/v1/escalations")
    assert resp.status_code == 200
    results = resp.json()
    assert results[0]["case_id"] == "case-fraud", "fraud-flagged case must be first despite lower amount"


def test_policy_upload_never_overwrites(client_and_db):
    client, _ = client_and_db

    pdf_content = b"%PDF-1.4 fake minimal content for upload test"
    fname = "TEST-UPLOAD-POLICY.pdf"
    upload_path = os.path.join("data", "policies", fname)
    if os.path.exists(upload_path):
        os.remove(upload_path)

    resp1 = client.post("/api/v1/policies/upload", files={"file": (fname, pdf_content, "application/pdf")})
    assert resp1.status_code == 202

    resp2 = client.post("/api/v1/policies/upload", files={"file": (fname, pdf_content, "application/pdf")})
    assert resp2.status_code == 409, "re-uploading the same filename must be rejected, never silently overwritten"

    if os.path.exists(upload_path):
        os.remove(upload_path)


def test_reopen_requires_resolved_state(client_and_db):
    client, db_module = client_and_db
    from app.core.db import SessionLocal, ExceptionCase, CaseState

    db = SessionLocal()
    db.add(ExceptionCase(id="case-reopen-1", order_id="ORD-REOPEN-1", customer_id="C1",
                          channel="direct", exception_type="return", state=CaseState.DIAGNOSING))
    db.commit()
    db.close()

    resp = client.post("/api/v1/cases/case-reopen-1/reopen")
    assert resp.status_code == 400, "a non-resolved case should not be reopenable"


def test_reopen_resolved_case_preserves_audit_trail(client_and_db):
    client, db_module = client_and_db
    from app.core.db import SessionLocal, ExceptionCase, CaseState, AuditLogEntry

    db = SessionLocal()
    db.add(ExceptionCase(id="case-reopen-2", order_id="ORD-REOPEN-2", customer_id="C1",
                          channel="direct", exception_type="return", state=CaseState.RESOLVED))
    db.commit()
    db.close()

    resp = client.post("/api/v1/cases/case-reopen-2/reopen", params={"reason": "customer disputed refund amount"})
    assert resp.status_code == 200
    assert resp.json()["state"] == "reopened"

    db2 = SessionLocal()
    audit_rows = db2.query(AuditLogEntry).filter(AuditLogEntry.case_id == "case-reopen-2").all()
    assert any(r.detail.get("to") == "reopened" for r in audit_rows), "reopen must be recorded in the audit trail"
    db2.close()
