"""
Creates a real API key for this system. The raw key is printed exactly
ONCE, here — only its hash is ever stored, so if you lose it, you
create a new one rather than "recover" the old one (same principle as
a password reset flow, not a password lookup).

Usage:
    python3 scripts/create_api_key.py "Jane Doe (CS)" cs_agent
    python3 scripts/create_api_key.py "OMS webhook service" service
    python3 scripts/create_api_key.py "Ops Admin" admin
    python3 scripts/create_api_key.py "Reporting dashboard" readonly
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import init_db, SessionLocal, ApiKeyRecord
from app.core.auth import hash_api_key, generate_api_key

VALID_ROLES = {"admin", "cs_agent", "service", "readonly"}


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/create_api_key.py \"<name>\" <role>")
        print(f"Valid roles: {', '.join(sorted(VALID_ROLES))}")
        sys.exit(1)

    name, role = sys.argv[1], sys.argv[2]
    if role not in VALID_ROLES:
        print(f"Invalid role '{role}'. Valid roles: {', '.join(sorted(VALID_ROLES))}")
        sys.exit(1)

    init_db()
    db = SessionLocal()

    raw_key = generate_api_key()
    record = ApiKeyRecord(key_hash=hash_api_key(raw_key), name=name, role=role)
    db.add(record)
    db.commit()
    db.close()

    print(f"Created API key for '{name}' (role={role})")
    print()
    print(f"  {raw_key}")
    print()
    print("This is shown ONCE - it is not recoverable. Store it securely.")
    print("Use it as: -H 'X-API-Key: <the key above>'")


if __name__ == "__main__":
    main()
