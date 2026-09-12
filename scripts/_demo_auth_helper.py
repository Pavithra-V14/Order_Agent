"""
Shared helper for demo scripts: creates a real, working admin API key
so demos exercise the actual, secured request flow - not bypass auth
entirely, which would undermine the whole point of having added it.

A real API key can't be recovered once created (only its hash is ever
stored - the same principle as a password). So across separate demo
script process invocations, the practical choice is either persisting
a raw secret to a local file (never appropriate, even for a demo) or
creating a fresh key each run. This does the latter: deletes any
previous demo key and creates a new one, every time a demo script
calls this - simple, and never writes a raw secret to disk.
"""
from app.core.db import SessionLocal, ApiKeyRecord, init_db
from app.core.auth import hash_api_key, generate_api_key

_DEMO_KEY_NAME = "Demo Scripts (auto-created)"


def get_or_create_demo_api_key() -> str:
    """Returns a fresh, real, working admin-role API key for demo
    scripts to use against this local instance."""
    init_db()
    db = SessionLocal()
    try:
        db.query(ApiKeyRecord).filter(ApiKeyRecord.name == _DEMO_KEY_NAME).delete()
        db.commit()

        raw_key = generate_api_key()
        record = ApiKeyRecord(key_hash=hash_api_key(raw_key), name=_DEMO_KEY_NAME, role="admin")
        db.add(record)
        db.commit()
        return raw_key
    finally:
        db.close()
