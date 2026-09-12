"""
Tests for real signup/login/JWT auth, backed by a DEDICATED, isolated
test Postgres database (order_exception_agent_test) - never the main
order_exception_agent database, so running this suite never pollutes
or depends on real account data. Requires a local Postgres with this
database already created (see scripts/setup_local_postgres.sh).

Skipped automatically if no Postgres is reachable, so this doesn't
break `pytest tests/` in an environment without one — same pattern as
test_redis_cache.py's Redis check. Found necessary directly from a real
user report: this previously ERRORED (not skipped) when Postgres
wasn't running, which is misleading in a full-suite run — a missing
optional dependency should skip cleanly, not look like a failure.
"""
import os

import pytest


def _postgres_available() -> bool:
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost", port=5432, user="postgres",
            password="postgres_dev_password", dbname="postgres", connect_timeout=2,
        )
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(),
    reason="No local Postgres reachable on localhost:5432 - see scripts/setup_auth_postgres.py",
)


@pytest.fixture(autouse=True)
def isolated_auth_postgres():
    """Points auth_database_url at a dedicated test database, resets
    the cached engine, creates a fresh users table, and truncates it
    before AND after each test - real Postgres, genuinely isolated."""
    os.environ["AUTH_DATABASE_URL"] = "postgresql://postgres:postgres_dev_password@localhost:5432/order_exception_agent_test"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.core.auth_db import reset_auth_engine_cache, init_auth_db, get_auth_engine
    reset_auth_engine_cache()
    init_auth_db()

    from sqlalchemy import text
    with get_auth_engine().connect() as conn:
        conn.execute(text("TRUNCATE TABLE users"))
        conn.commit()

    yield

    with get_auth_engine().connect() as conn:
        conn.execute(text("TRUNCATE TABLE users"))
        conn.commit()

    reset_auth_engine_cache()
    os.environ.pop("AUTH_DATABASE_URL", None)
    get_settings.cache_clear()


@pytest.fixture
def real_auth_client():
    """Same real-auth-enabled pattern as tests/test_auth.py — this
    specifically needs auth actually enforced to test the JWT it
    produces."""
    os.environ["AUTH_ENABLED"] = "true"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        yield client

    os.environ["AUTH_ENABLED"] = "false"
    get_settings.cache_clear()


def test_signup_creates_a_real_user_and_returns_a_real_jwt(real_auth_client):
    resp = real_auth_client.post("/api/v1/auth/signup", json={
        "email": "signup-test@example.com", "password": "realpassword123",
        "name": "Signup Test", "role": "cs_agent",
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["access_token"]
    assert data["role"] == "cs_agent"
    assert data["name"] == "Signup Test"

    # The password is genuinely hashed in the database, never stored raw
    from app.core.auth_db import get_auth_session_factory, User
    db = get_auth_session_factory()()
    user = db.query(User).filter(User.email == "signup-test@example.com").first()
    assert user is not None
    assert user.hashed_password != "realpassword123"
    assert user.hashed_password.startswith("$2b$")  # a real bcrypt hash
    db.close()


def test_signup_rejects_duplicate_email(real_auth_client):
    real_auth_client.post("/api/v1/auth/signup", json={
        "email": "dup@example.com", "password": "realpassword123", "name": "First", "role": "cs_agent",
    })
    resp = real_auth_client.post("/api/v1/auth/signup", json={
        "email": "dup@example.com", "password": "differentpassword", "name": "Second", "role": "readonly",
    })
    assert resp.status_code == 409


def test_signup_rejects_admin_role_self_service(real_auth_client):
    """THE regression test for a real, deliberate authorization
    decision: admin (and service) roles must NOT be grantable through
    public self-service signup."""
    resp = real_auth_client.post("/api/v1/auth/signup", json={
        "email": "wannabe-admin@example.com", "password": "realpassword123",
        "name": "Wannabe Admin", "role": "admin",
    })
    assert resp.status_code == 422


def test_signup_rejects_short_password(real_auth_client):
    resp = real_auth_client.post("/api/v1/auth/signup", json={
        "email": "short@example.com", "password": "short", "name": "Short Password", "role": "cs_agent",
    })
    assert resp.status_code == 422


def test_login_with_correct_password_succeeds(real_auth_client):
    real_auth_client.post("/api/v1/auth/signup", json={
        "email": "login-test@example.com", "password": "realpassword123", "name": "Login Test", "role": "cs_agent",
    })
    resp = real_auth_client.post("/api/v1/auth/login", json={
        "email": "login-test@example.com", "password": "realpassword123",
    })
    assert resp.status_code == 200
    assert resp.json()["access_token"]


def test_login_with_wrong_password_fails_with_generic_message(real_auth_client):
    """The error message must be the SAME whether the email doesn't
    exist or the password is wrong - distinguishing them lets an
    attacker enumerate valid accounts."""
    real_auth_client.post("/api/v1/auth/signup", json={
        "email": "wrongpw-test@example.com", "password": "realpassword123", "name": "Test", "role": "cs_agent",
    })
    resp_wrong_password = real_auth_client.post("/api/v1/auth/login", json={
        "email": "wrongpw-test@example.com", "password": "wrongpassword",
    })
    resp_nonexistent_email = real_auth_client.post("/api/v1/auth/login", json={
        "email": "never-signed-up@example.com", "password": "whatever123",
    })
    assert resp_wrong_password.status_code == 401
    assert resp_nonexistent_email.status_code == 401
    assert resp_wrong_password.json()["detail"] == resp_nonexistent_email.json()["detail"]


def test_jwt_from_login_authenticates_subsequent_requests(real_auth_client):
    real_auth_client.post("/api/v1/auth/signup", json={
        "email": "jwt-flow-test@example.com", "password": "realpassword123", "name": "JWT Flow Test", "role": "readonly",
    })
    login_resp = real_auth_client.post("/api/v1/auth/login", json={
        "email": "jwt-flow-test@example.com", "password": "realpassword123",
    })
    token = login_resp.json()["access_token"]

    resp = real_auth_client.get("/api/v1/cases", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200


def test_jwt_role_is_enforced_same_as_api_key_role(real_auth_client):
    """THE regression test proving JWT-based human sign-in goes through
    the SAME real role enforcement as service API keys — a readonly
    user must be rejected from an admin-only endpoint exactly like a
    readonly API key would be."""
    real_auth_client.post("/api/v1/auth/signup", json={
        "email": "readonly-jwt-test@example.com", "password": "realpassword123",
        "name": "Readonly JWT Test", "role": "readonly",
    })
    login_resp = real_auth_client.post("/api/v1/auth/login", json={
        "email": "readonly-jwt-test@example.com", "password": "realpassword123",
    })
    token = login_resp.json()["access_token"]

    resp = real_auth_client.get("/api/v1/admin/backend-status", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 403


def test_invalid_jwt_is_rejected(real_auth_client):
    resp = real_auth_client.get("/api/v1/cases", headers={"Authorization": "Bearer not.a.real.jwt.token"})
    assert resp.status_code == 401


def test_expired_jwt_is_rejected(real_auth_client):
    from app.core.auth import create_access_token
    from app.core.config import get_settings
    import time

    # A token that's already expired (negative expiry) - proves
    # expiration is actually checked, not just accepted on signature
    # validity alone.
    from jose import jwt
    settings = get_settings()
    expired_payload = {"sub": "fake-id", "email": "fake@example.com", "role": "admin",
                        "exp": int(time.time()) - 3600}
    expired_token = jwt.encode(expired_payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

    resp = real_auth_client.get("/api/v1/cases", headers={"Authorization": f"Bearer {expired_token}"})
    assert resp.status_code == 401
