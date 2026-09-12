"""
Creates an admin (or any role) USER account directly in the auth
database - bypassing /signup, which deliberately REJECTS admin/service
roles as a real security choice (see app/api/v1/auth.py's
VALID_SIGNUP_ROLES). Someone has to be able to create the first admin
account somehow; this is that path - a deliberate, explicit script
action (matching scripts/create_api_key.py's pattern for service
credentials), not a public HTTP endpoint anyone could hit.

Usage:
    python3 scripts/create_admin_user.py "Jane Doe" jane@example.com admin
    python3 scripts/create_admin_user.py "Ops Reviewer" ops@example.com cs_agent
"""
import sys
import os
import getpass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.auth_db import init_auth_db, get_auth_session_factory, User
from app.core.auth import hash_password

VALID_ROLES = {"admin", "cs_agent", "readonly"}


def main():
    if len(sys.argv) != 4:
        print('Usage: python3 scripts/create_admin_user.py "<name>" <email> <role>')
        print(f"Valid roles: {', '.join(sorted(VALID_ROLES))}")
        sys.exit(1)

    name, email, role = sys.argv[1], sys.argv[2], sys.argv[3]
    if role not in VALID_ROLES:
        print(f"Invalid role '{role}'. Valid roles: {', '.join(sorted(VALID_ROLES))}")
        sys.exit(1)

    password = getpass.getpass("Set a password for this account: ")
    if len(password) < 8:
        print("Password must be at least 8 characters.")
        sys.exit(1)
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Passwords did not match.")
        sys.exit(1)

    init_auth_db()
    db = get_auth_session_factory()()

    existing = db.query(User).filter(User.email == email).first()
    if existing is not None:
        print(f"An account with email '{email}' already exists (role={existing.role}).")
        db.close()
        sys.exit(1)

    user = User(email=email, hashed_password=hash_password(password), name=name, role=role)
    db.add(user)
    db.commit()
    db.close()

    print(f"\nCreated {role} account: {name} <{email}>")
    print("Sign in at http://127.0.0.1:8000/login with this email and password.")


if __name__ == "__main__":
    main()
