"""
Tests proving authentication/authorization is REAL, not just present in
code - explicitly disabling the conftest.py auth-bypass fixture (which
every other test in this suite relies on) so these specifically
exercise the genuine, enforced path.
"""
import os
import tempfile

import pytest


@pytest.fixture
def real_auth_client():
    """Explicitly re-enables real auth for this test only, undoing
    conftest.py's session-wide bypass, and gives a fresh isolated DB so
    created API keys don't leak into other tests."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_auth_{os.getpid()}_{id(object())}.db")
    import app.core.db as db_module
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    fresh_engine = create_engine(f"sqlite:///{tmp_path}", connect_args={"check_same_thread": False})
    db_module.Base.metadata.create_all(bind=fresh_engine)
    db_module.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=fresh_engine)
    db_module.engine = fresh_engine

    os.environ["AUTH_ENABLED"] = "true"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        yield client, db_module

    os.environ["AUTH_ENABLED"] = "false"  # restore conftest.py's session-wide bypass
    get_settings.cache_clear()
    fresh_engine.dispose()
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except PermissionError:
        pass


def _create_key(db_module, name: str, role: str) -> str:
    from app.core.auth import hash_api_key, generate_api_key
    raw_key = generate_api_key()
    db = db_module.SessionLocal()
    db.add(db_module.ApiKeyRecord(key_hash=hash_api_key(raw_key), name=name, role=role))
    db.commit()
    db.close()
    return raw_key


def test_request_with_no_api_key_is_rejected(real_auth_client):
    client, db_module = real_auth_client
    resp = client.get("/api/v1/cases")
    assert resp.status_code == 401


def test_request_with_wrong_api_key_is_rejected(real_auth_client):
    client, db_module = real_auth_client
    resp = client.get("/api/v1/cases", headers={"X-API-Key": "not_a_real_key"})
    assert resp.status_code == 401


def test_request_with_valid_readonly_key_can_read(real_auth_client):
    client, db_module = real_auth_client
    key = _create_key(db_module, "Reporting Dashboard", "readonly")
    resp = client.get("/api/v1/cases", headers={"X-API-Key": key})
    assert resp.status_code == 200


def test_readonly_key_cannot_reopen_a_case(real_auth_client):
    """THE regression test for real authorization, not just
    authentication: a valid key with the WRONG role must be rejected
    with 403, not merely 401 for having no key at all."""
    client, db_module = real_auth_client
    key = _create_key(db_module, "Reporting Dashboard", "readonly")
    resp = client.post("/api/v1/cases/some-fake-id/reopen", headers={"X-API-Key": key})
    assert resp.status_code == 403


def test_service_key_forbidden_from_human_decision(real_auth_client):
    """THE regression test for the exact design mistake found and fixed
    before this was ever used: an earlier version of require_roles used
    one linear hierarchy where "service" and "cs_agent" were treated as
    the same level, meaning a webhook's service credential could have
    approved a human escalation decision."""
    client, db_module = real_auth_client
    key = _create_key(db_module, "OMS Webhook Service", "service")
    resp = client.post("/api/v1/escalations/some-fake-id/decision",
                        json={"action": "approve"}, headers={"X-API-Key": key})
    assert resp.status_code == 403, (
        "a 'service' role key must NOT be able to approve human escalation decisions — "
        "these are different kinds of access, not one a linear step below the other"
    )


def test_cs_agent_key_cannot_hit_admin_endpoints(real_auth_client):
    client, db_module = real_auth_client
    key = _create_key(db_module, "Jane Doe (CS)", "cs_agent")
    resp = client.get("/api/v1/admin/backend-status", headers={"X-API-Key": key})
    assert resp.status_code == 403


def test_admin_key_can_do_everything(real_auth_client):
    """Admin is a strict superset — verified across three different
    role-gated endpoints, not just one."""
    client, db_module = real_auth_client
    key = _create_key(db_module, "Ops Admin", "admin")

    assert client.get("/api/v1/cases", headers={"X-API-Key": key}).status_code == 200
    assert client.get("/api/v1/admin/backend-status", headers={"X-API-Key": key}).status_code == 200
    resp = client.post("/api/v1/webhooks/oms", json={"order_id": "ORD-X", "new_status": "payment_failed"},
                        headers={"X-API-Key": key})
    assert resp.status_code == 202


def test_inactive_key_is_rejected(real_auth_client):
    """A revoked/deactivated key must stop working immediately, not
    just be soft-hidden from a listing somewhere."""
    client, db_module = real_auth_client
    from app.core.auth import hash_api_key, generate_api_key
    raw_key = generate_api_key()
    db = db_module.SessionLocal()
    db.add(db_module.ApiKeyRecord(key_hash=hash_api_key(raw_key), name="Revoked Key", role="admin", active=False))
    db.commit()
    db.close()

    resp = client.get("/api/v1/cases", headers={"X-API-Key": raw_key})
    assert resp.status_code == 401


def test_health_endpoint_requires_no_auth(real_auth_client):
    """Health checks must remain unauthenticated — load balancers and
    uptime monitors can't be expected to carry an API key."""
    client, db_module = real_auth_client
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
