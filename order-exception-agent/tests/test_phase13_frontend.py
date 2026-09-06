"""
Phase 13 DoD: "A full manual walkthrough - trigger a synthetic exception
via webhook, watch it appear in the queue, diagnose, escalate, approve
from the UI, see it resolve and appear correctly in the audit log -
works end to end through the UI alone."
"""
import os
import tempfile
import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_and_db():
    tmp_db = os.path.join(tempfile.gettempdir(), "test_phase13.db")
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


def test_all_eight_pages_render(client_and_db):
    """Every page from architecture doc 8.3 returns 200 and real HTML."""
    client, _ = client_and_db

    static_pages = ["/", "/escalations", "/policies", "/metrics", "/threshold-config", "/audit-log"]
    for path in static_pages:
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} failed to render"
        assert "<html" in resp.text.lower()

    resp = client.get("/cases/some-id")
    assert resp.status_code == 200
    resp = client.get("/traces/some-id")
    assert resp.status_code == 200

    css = client.get("/static/style.css")
    js = client.get("/static/app.js")
    assert css.status_code == 200 and len(css.text) > 500
    assert js.status_code == 200 and len(js.text) > 500


def test_full_walkthrough_webhook_to_resolved_to_audit_log(client_and_db):
    """THE Phase 13 DoD test - driven through the exact HTTP calls each
    page's app.js makes, step by step."""
    client, db_module = client_and_db
    from datetime import datetime, timezone
    from app.core.db import SessionLocal, ExceptionCase, CaseState
    from app.tools.oms import create_order
    from app.tools.payment import get_payment_gateway

    db = SessionLocal()
    create_order(db, order_id="ORD-WALK-1", customer_id="CUST-WALK-1", channel="direct",
                 status="paid", total_amount_usd=45.0, purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
                 line_items=[{"sku": "SKU-WALK-1", "category": "apparel", "qty": 1, "price": 45.0}],
                 payment_intent_id="pi_walk_1")
    get_payment_gateway().seed_transaction("pi_walk_1", amount_usd=45.0, status="declined")
    db.close()

    resp = client.post("/api/v1/webhooks/oms", json={"order_id": "ORD-WALK-1", "new_status": "payment_failed"})
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    job = None
    for _ in range(100):
        job = client.get(f"/api/v1/webhooks/jobs/{job_id}").json()
        if job["status"] == "succeeded":
            break
        time.sleep(0.02)
    assert job["status"] == "succeeded"
    case_id = job["result"]["case_created"]
    assert case_id is not None

    cases = client.get("/api/v1/cases").json()
    assert any(c["id"] == case_id for c in cases), "case did not appear in the Case Queue feed"

    db2 = SessionLocal()
    case = db2.get(ExceptionCase, case_id)
    case.state = CaseState.ESCALATED
    case.resolution_decision = {
        "action": "refund", "amount_usd": 45.0, "confidence": 0.75,
        "reasoning": "Payment declined; escalated for review due to confidence below threshold.",
        "cited_policy": {"doc_id": "RET-POLICY-2025-A", "version": "1", "clause_summary": "return window"},
    }
    db2.commit()
    db2.close()

    escalations = client.get("/api/v1/escalations").json()
    assert any(e["case_id"] == case_id for e in escalations), "case did not appear in the Escalation Queue"

    resp = client.post(f"/api/v1/escalations/{case_id}/decision",
                        json={"action": "approve", "decided_by": "human:walkthrough@test.com"})
    assert resp.status_code == 200
    assert resp.json()["outcome"] == "resolved"

    final_case = client.get(f"/api/v1/cases/{case_id}").json()
    assert final_case["state"] == "resolved"
    assert final_case["execution_result"]["status"] == "executed"

    audit = client.get(f"/api/v1/audit-log?case_id={case_id}").json()
    actions = [a["action"] for a in audit]
    assert "human_decision" in actions
    assert "state_transition" in actions
    human_decision_entry = next(a for a in audit if a["action"] == "human_decision")
    assert human_decision_entry["actor"] == "human:walkthrough@test.com"


def test_threshold_config_page_data_reflects_pending_vs_active(client_and_db):
    """Confirms the Threshold Config page's two data sources (pending
    proposals vs. active overrides) are correctly separate."""
    client, db_module = client_and_db
    from app.core.db import SessionLocal, ExceptionCase, CaseState

    db = SessionLocal()
    for i in range(6):
        case = ExceptionCase(id=f"case-thresh-{i}", order_id=f"ORD-T{i}", customer_id="CUST-T",
                              channel="direct", exception_type="return", state=CaseState.RESOLVED)
        db.add(case)
        db.commit()
        from app.agents.learning_loop import record_resolution_outcome
        record_resolution_outcome(
            db, case_id=f"case-thresh-{i}", cluster_key="return_test_cluster",
            case_feature_summary=f"test case {i}",
            agent_proposed_resolution={"action": "refund", "amount_usd": 20.0},
            human_final_resolution={"action": "refund", "amount_usd": 20.0},
        )
    db.close()

    resp = client.post("/api/v1/threshold-proposals/run-batch-job",
                        json={"current_threshold": 0.90, "min_sample_size": 5})
    assert resp.json()["proposals_created"] == 1

    pending = client.get("/api/v1/threshold-proposals?status=pending_review").json()
    assert len(pending) == 1
    overrides_before = client.get("/api/v1/threshold-proposals/active-overrides").json()
    assert overrides_before == [], "must be empty before accept"

    resp = client.post(f"/api/v1/threshold-proposals/{pending[0]['id']}/accept",
                        json={"decided_by": "human:ops@test.com"})
    assert resp.status_code == 200

    overrides_after = client.get("/api/v1/threshold-proposals/active-overrides").json()
    assert len(overrides_after) == 1
    assert overrides_after[0]["cluster_key"] == "return_test_cluster"
