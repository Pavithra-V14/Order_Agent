"""
Phase 0 DoD tests: API is up, DB is reachable, and a case round-trips
through create -> get -> list with an audit row written automatically.

Fixture-based setup (not module-level import-order-dependent) — matches
every other test file's pattern. The original version of this file set
DATABASE_URL and imported app.main at MODULE level, relying on being the
FIRST test file to ever import app.core.db in the pytest session; this
broke once later-added test files (test_easypost_gateway.py,
test_groq_client.py, etc.) sorted alphabetically BEFORE this file and
imported+reloaded app.core.db first, leaving this file's own
DATABASE_URL setting with no effect on the already-cached module. Fixed
by reloading app.core.db explicitly in a fixture, exactly like every
other phase's test file already does.
"""
import os
import tempfile

import pytest

@pytest.fixture
def client_and_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_phase0_{os.getpid()}_{id(object())}.db")
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}"

    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)

    import app.main as main_module
    importlib.reload(main_module)

    from fastapi.testclient import TestClient
    with TestClient(main_module.app) as client:
        yield client, db_module

    try:

        if os.path.exists(tmp_path):

            os.remove(tmp_path)

    except PermissionError:

        pass  # Windows may still hold a brief lock from engine cleanup; harmless to leave a stray temp file

def test_health_ok(client_and_db):
    client, _ = client_and_db
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"

def test_case_round_trip_writes_audit_row(client_and_db):
    client, db_module = client_and_db
    create_resp = client.post(
        "/api/v1/cases",
        json={
            "order_id": "ORD-TEST-1",
            "customer_id": "CUST-TEST-1",
            "channel": "direct",
            "exception_type": "payment",
        },
    )
    assert create_resp.status_code == 201
    case = create_resp.json()
    assert case["state"] == "detected"
    case_id = case["id"]

    get_resp = client.get(f"/api/v1/cases/{case_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["order_id"] == "ORD-TEST-1"

    list_resp = client.get("/api/v1/cases")
    assert list_resp.status_code == 200
    assert any(c["id"] == case_id for c in list_resp.json())

    # Audit trail is written automatically on state transition (Layer 13 / 8.8)
    db = db_module.SessionLocal()
    audit_rows = db.query(db_module.AuditLogEntry).filter(db_module.AuditLogEntry.case_id == case_id).all()
    assert len(audit_rows) == 1
    assert audit_rows[0].action == "state_transition"
    assert audit_rows[0].detail["to"] == "detected"
    db.close()

def test_get_nonexistent_case_returns_404(client_and_db):
    client, _ = client_and_db
    resp = client.get("/api/v1/cases/does-not-exist")
    assert resp.status_code == 404
