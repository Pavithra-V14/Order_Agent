"""
Dedicated database connection for auth data - a SEPARATE engine/session
from app.core.db's main application database, deliberately backed by
real Postgres (see Settings.auth_database_url) rather than sharing
whatever the main DATABASE_URL happens to be.

Why separate: real enterprise systems commonly run identity/auth as its
own subsystem with its own datastore - both for security isolation (a
compromise of case/order data doesn't automatically expose credentials,
and vice versa) and so auth can be scaled, backed up, and audited on
its own schedule independent of transactional order/case data. Postgres
specifically (not SQLite, which the main app DB defaults to for local
dev) because real credential storage benefits from Postgres's stronger
concurrent-write guarantees and constraint enforcement (a unique email
constraint under real concurrent signups is exactly the kind of thing
SQLite's file-level locking handles far less gracefully).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, String, Boolean, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import get_settings

AuthBase = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class User(AuthBase):
    """A human sign-in identity - separate from ApiKeyRecord
    (app.core.db), which is for service/system credentials (webhooks,
    long-lived integration keys). Users sign up/log in through a real
    UI and get a JWT; services use a long-lived API key. Both end up
    producing the same kind of "current principal with a role" object
    at the dependency layer (see app/core/auth.py), so the rest of the
    app doesn't need to care which kind of credential authenticated a
    given request.
    """
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    email = Column(String, nullable=False, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)  # "admin" | "cs_agent" | "readonly" - same roles as ApiKeyRecord
    active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    last_login_at = Column(DateTime(timezone=True), nullable=True)


_engine = None
_SessionLocal = None


def get_auth_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.auth_database_url, pool_pre_ping=True)
    return _engine


def get_auth_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_auth_engine())
    return _SessionLocal


def init_auth_db():
    """Creates the users table if it doesn't exist yet. Called at app
    startup (see app/main.py's lifespan) and directly by tests/scripts
    that need a real, ready auth database."""
    AuthBase.metadata.create_all(bind=get_auth_engine())


def get_auth_db():
    """FastAPI dependency - yields a session against the DEDICATED auth
    database, never the main app database."""
    db = get_auth_session_factory()()
    try:
        yield db
    finally:
        db.close()


def reset_auth_engine_cache():
    """Test-only: forces get_auth_engine()/get_auth_session_factory() to
    rebuild against whatever settings.auth_database_url currently is,
    rather than keep using a stale cached engine from before a test
    changed it."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
