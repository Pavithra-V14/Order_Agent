"""
Tests for app/api/v1/admin.py - the delete/reset endpoints for audit
log, RAG index, and full Postgres data wipe.
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_admin_{os.getpid()}_{id(object())}.db")
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


def test_delete_audit_log_requires_confirm():
    from app.main import app
    with TestClient(app) as client:
        resp = client.delete("/api/v1/admin/audit-log")
        assert resp.status_code == 400
        assert "confirm=true" in resp.json()["detail"]


def test_delete_audit_log_actually_deletes(isolated_db):
    from app.core.db import SessionLocal, AuditLogEntry
    db = SessionLocal()
    db.add(AuditLogEntry(case_id="c1", actor="test", action="test_action", detail={}))
    db.add(AuditLogEntry(case_id="c2", actor="test", action="test_action", detail={}))
    db.commit()
    assert db.query(AuditLogEntry).count() == 2
    db.close()

    from app.main import app
    with TestClient(app) as client:
        resp = client.delete("/api/v1/admin/audit-log?confirm=true")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 2

    db2 = SessionLocal()
    assert db2.query(AuditLogEntry).count() == 0
    db2.close()


def test_delete_all_data_requires_confirm():
    from app.main import app
    with TestClient(app) as client:
        resp = client.delete("/api/v1/admin/all-data")
        assert resp.status_code == 400


def test_delete_all_data_wipes_every_table(isolated_db):
    from app.core.db import SessionLocal, ExceptionCase, CaseState, AuditLogEntry, MockOrderRecord
    from datetime import datetime, timezone
    db = SessionLocal()
    db.add(MockOrderRecord(order_id="ORD-1", customer_id="C1", channel="direct", status="paid",
                            total_amount_usd=10.0, purchase_date=datetime(2025, 1, 1, tzinfo=timezone.utc),
                            line_items=[]))
    db.add(ExceptionCase(id="case-1", order_id="ORD-1", customer_id="C1", channel="direct",
                          exception_type="return", state=CaseState.DETECTED))
    db.add(AuditLogEntry(case_id="case-1", actor="test", action="test_action", detail={}))
    db.commit()
    db.close()

    from app.main import app
    with TestClient(app) as client:
        resp = client.delete("/api/v1/admin/all-data?confirm=true")
        assert resp.status_code == 200
        counts = resp.json()["deleted_counts"]
        assert counts["exception_cases"] == 1
        assert counts["audit_log_entries"] == 1
        assert counts["mock_order_records"] == 1
        assert resp.json()["total_rows_deleted"] == 3

    db2 = SessionLocal()
    assert db2.query(ExceptionCase).count() == 0
    assert db2.query(AuditLogEntry).count() == 0
    assert db2.query(MockOrderRecord).count() == 0
    db2.close()


def test_delete_rag_index_requires_confirm():
    from app.main import app
    with TestClient(app) as client:
        resp = client.delete("/api/v1/admin/rag-index")
        assert resp.status_code == 400


def test_delete_rag_index_removes_collection(tmp_path):
    """Uses a REAL embedded Qdrant client against a throwaway temp path
    - not mocked - to genuinely exercise collection creation/deletion,
    the actual mechanism this endpoint relies on."""
    qdrant_path = str(tmp_path / "qdrant_test")
    os.environ["QDRANT_LOCAL_PATH"] = qdrant_path
    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.rag.vectorstore as vs
    if vs._client_singleton is not None:
        vs._client_singleton.close()
    vs._client_singleton = None
    vs._client_singleton_key = None

    client = vs.get_qdrant_client()
    vs.ensure_collection(client, get_settings().qdrant_collection, dim=8)
    existing = [c.name for c in client.get_collections().collections]
    assert get_settings().qdrant_collection in existing

    import app.api.v1.admin as admin_module
    result = admin_module.delete_rag_index(confirm=True)
    assert result["deleted_collection"] is True

    existing_after = [c.name for c in client.get_collections().collections]
    assert get_settings().qdrant_collection not in existing_after

    os.environ.pop("QDRANT_LOCAL_PATH", None)
    get_settings.cache_clear()


def test_backend_status_reports_fake_when_nothing_configured():
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        resp = client.get("/api/v1/admin/backend-status")
        assert resp.status_code == 200
        data = resp.json()
        assert "FakePaymentGateway" in data["payment"]["active"]
        assert "FakeCarrierGateway" in data["carrier"]["active"]
        assert "FakeLLMClient" in data["llm"]["active"]
        assert "TF-IDF" in data["embedder"]["active"]


def test_backend_status_reports_real_when_credentials_configured(monkeypatch):
    """THE regression test proving this diagnostic actually reflects
    real configuration state, not a hardcoded answer."""
    monkeypatch.setenv("STRIPE_API_KEY", "sk_test_fake_for_status_check")
    monkeypatch.setenv("SHIPPO_API_KEY", "shippo_test_fake_for_status_check")
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        data = client.get("/api/v1/admin/backend-status").json()
        assert "Stripe" in data["payment"]["active"]
        assert "Shippo" in data["carrier"]["active"]

    get_settings.cache_clear()


def test_backend_status_flags_easypost_priority_when_both_carrier_keys_set(monkeypatch):
    monkeypatch.setenv("EASYPOST_API_KEY", "EZTK_fake")
    monkeypatch.setenv("SHIPPO_API_KEY", "shippo_test_fake")
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        data = client.get("/api/v1/admin/backend-status").json()
        assert "EasyPost" in data["carrier"]["active"]
        assert data["carrier"]["note"] is not None

    get_settings.cache_clear()
