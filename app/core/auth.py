"""
Real authentication and authorization - found completely missing during
a direct audit: every endpoint was open, "per-customer" processing
worked purely because callers self-reported identity strings
(customer_id, decided_by) with zero verification. Anyone hitting these
endpoints could claim to be any customer, or approve escalations as any
named human, with nothing checking that claim.

Two credential types, unified into one dependency:
- Service API keys (X-API-Key header, ApiKeyRecord in app.core.db) -
  for webhooks and system-to-system integrations, long-lived.
- Human JWT sign-in (Authorization: Bearer <token>, User in
  app.core.auth_db, backed by Postgres) - for CS agents/admins signing
  in through a real UI, short-lived tokens issued at login.

Keys/passwords are hashed before storage - never stored raw, same
principle as never storing a plaintext password. The raw API key is
shown to the person creating it exactly once; a JWT is issued at login
and expires on its own.
"""
from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import bcrypt
from sqlalchemy.orm import Session

from app.core.db import get_db, ApiKeyRecord

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


class CurrentPrincipal:
    """A unified "who is making this request and what role do they
    have" object, regardless of whether they authenticated via a
    service API key or a human JWT sign-in. The rest of the app
    (require_roles, decided_by derivation in escalations.py/threshold.py)
    only ever needs .name and .role - it doesn't need to know or care
    which credential type produced them.
    """
    def __init__(self, name: str, role: str, principal_type: str, principal_id: str):
        self.name = name
        self.role = role
        self.principal_type = principal_type  # "api_key" | "user"
        self.id = principal_id


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    """A cryptographically random key, prefixed for easy identification
    in logs/screenshots without revealing anything about the key itself
    (matching the style of Stripe/GitHub's own key prefixes)."""
    return f"oea_{secrets.token_urlsafe(32)}"


def hash_password(raw_password: str) -> str:
    """bcrypt used directly, not through passlib's CryptContext wrapper
    - found and fixed a real, known compatibility issue: passlib's own
    internal backend self-test ("detect_wrap_bug") crashes against
    bcrypt>=4.1's changed internals with a misleading "password cannot
    be longer than 72 bytes" error, even for genuinely short passwords.
    This is a documented passlib/bcrypt version mismatch, not a bug in
    the password itself - bcrypt used directly has no such issue.

    bcrypt has a real 72-BYTE input limit (not characters - a real
    constraint of the algorithm itself, unrelated to the passlib bug
    above), so passwords are truncated to 72 bytes before hashing, the
    standard, documented way to use bcrypt within its actual limit.
    """
    return bcrypt.hashpw(raw_password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(raw_password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(raw_password.encode("utf-8")[:72], hashed_password.encode("utf-8"))


def create_access_token(user_id: str, email: str, role: str) -> str:
    from app.core.config import get_settings
    settings = get_settings()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_access_token_expire_minutes)
    payload = {"sub": user_id, "email": email, "role": role, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> dict:
    from app.core.config import get_settings
    settings = get_settings()
    return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])


def get_current_principal(
    raw_key: str = Security(_api_key_header),
    bearer: HTTPAuthorizationCredentials = Security(_bearer_scheme),
    db: Session = Depends(get_db),
) -> CurrentPrincipal:
    """The core dependency every protected endpoint uses. Accepts
    EITHER a service API key (X-API-Key) or a human JWT
    (Authorization: Bearer <token>) - checks the JWT first since a
    signed-in human in the browser is the more common real interactive
    case, then falls back to the API key header. Raises 401 if neither
    is present or valid - never silently treats a missing credential as
    "anonymous but allowed", which was the previous, real behavior of
    every endpoint in this project.

    settings.auth_enabled=False bypasses this entirely, returning a
    fixed fake admin principal — checked fresh on every call via
    get_settings(), which is deliberately what makes this survive the
    app-instance reloading several test fixtures do (unlike patching
    app.dependency_overrides directly, which gets wiped whenever
    importlib.reload(app.main) creates a fresh app object elsewhere in
    this project's test suite).
    """
    from app.core.config import get_settings
    if not get_settings().auth_enabled:
        return CurrentPrincipal(name="Auth Disabled (dev/test)", role="admin",
                                 principal_type="disabled", principal_id="auth-disabled")

    if bearer is not None and bearer.credentials:
        try:
            payload = decode_access_token(bearer.credentials)
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid or expired session token")
        return CurrentPrincipal(name=payload["email"], role=payload["role"],
                                 principal_type="user", principal_id=payload["sub"])

    if raw_key:
        key_hash = hash_api_key(raw_key)
        record = db.query(ApiKeyRecord).filter(ApiKeyRecord.key_hash == key_hash).first()
        if record is None or not record.active:
            raise HTTPException(status_code=401, detail="Invalid or inactive API key")
        record.last_used_at = datetime.now(timezone.utc)
        db.commit()
        return CurrentPrincipal(name=record.name, role=record.role,
                                 principal_type="api_key", principal_id=record.id)

    raise HTTPException(status_code=401, detail="Missing credentials - provide X-API-Key or a Bearer token")


# Backward-compatible alias — every route file already depends on
# get_current_api_key by name; keeping the name means those call sites
# don't all need editing, while the function itself now accepts BOTH
# credential types, not just API keys.
get_current_api_key = get_current_principal


def require_roles(*allowed_roles: str):
    """Dependency factory taking explicit allowed roles, not a single
    linear hierarchy number. Found and fixed a real design mistake
    before this was ever used: an earlier version used one ordered
    hierarchy (readonly < cs_agent/service < admin) — but "service"
    (webhook/system calls) and "cs_agent" (a human approving
    escalations) aren't actually a "more/less" relationship at all,
    they're different KINDS of access. Under that flawed hierarchy, a
    webhook's service key would have passed a cs_agent-only check and
    could have approved an escalation as if it were a human — exactly
    the kind of authorization mistake this whole audit exists to catch.

    "admin" is always implicitly allowed, on top of whatever explicit
    roles are listed, since admin is meant to mean "can do everything."
    """
    allowed = set(allowed_roles) | {"admin"}

    def _dependency(current: CurrentPrincipal = Depends(get_current_principal)) -> CurrentPrincipal:
        if current.role not in allowed:
            raise HTTPException(
                status_code=403,
                detail=f"This action requires one of roles {sorted(allowed)}; "
                       f"your credential has role '{current.role}'.",
            )
        return current

    return _dependency


# Convenience dependencies for the common cases, so route files read
# clearly (Depends(require_admin) rather than repeating the role list
# everywhere with the strings easy to typo).
require_readonly = require_roles("readonly", "cs_agent", "service")  # any valid, active credential can read
require_cs_agent = require_roles("cs_agent")   # cs_agent or admin — explicitly NOT service
require_service = require_roles("service")     # service or admin — explicitly NOT cs_agent
require_admin = require_roles()                # admin only (the implicit admin add-in above)
