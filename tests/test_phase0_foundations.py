"""
Phase 0 DoD tests: API is up, DB is reachable, and a case round-trips
through create -> get -> list with an audit row written automatically.

Uses an isolated in-memory SQLite DB per test run (not the dev data/ file),
so this is safe to run repeatedly and in CI without state leaking between runs.
"""
import os
import tempfile

# Use a real temp file, not sqlite:///:memory: — an in-memory SQLite DB is
# per-connection, and SQLAlchemy opens a new connection per session by
# default, so :memory: would silently lose the tables between requests
# without extra StaticPool wiring we don't need for a throwaway test DB.
_tmp_db_path = os.path.join(tempfile.gettempdir(), "test_case_state.db")
if os.path.exists(_tmp_db_path):
    os.remove(_tmp_db_path)
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp_db_path}"

from fastapi.testclient import TestClient

from app.main import app
from app.core.db import SessionLocal, AuditLogEntry

# Using TestClient as a context manager triggers FastAPI's startup event
# (init_db(), which creates tables) — without `with`, startup never fires
# and every query 404s with "no such table" even though the DB connects fine.
client = TestClient(app)
client.__enter__()


def test_health_ok():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"


def test_case_round_trip_writes_audit_row():
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
    db = SessionLocal()
    audit_rows = db.query(AuditLogEntry).filter(AuditLogEntry.case_id == case_id).all()
    assert len(audit_rows) == 1
    assert audit_rows[0].action == "state_transition"
    assert audit_rows[0].detail["to"] == "detected"


def test_get_nonexistent_case_returns_404():
    resp = client.get("/api/v1/cases/does-not-exist")
    assert resp.status_code == 404
