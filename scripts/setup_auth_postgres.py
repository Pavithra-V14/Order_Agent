"""
Verifies AUTH_DATABASE_URL is reachable and creates the users table if
needed. Run this once after setting up Postgres (see .env.example's
Authentication section for setup instructions per platform) and before
first using signup/login.

Usage:
    python3 scripts/setup_auth_postgres.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    from app.core.config import get_settings
    settings = get_settings()

    print(f"Checking connection to: {settings.auth_database_url.split('@')[-1]}")
    print("(host/database shown only — credentials are not printed)")
    print()

    try:
        from app.core.auth_db import get_auth_engine, init_auth_db
        engine = get_auth_engine()
        with engine.connect() as conn:
            from sqlalchemy import text
            result = conn.execute(text("SELECT version();"))
            version = result.fetchone()[0]
        print(f"Connected successfully.")
        print(f"  {version}")
        print()
    except Exception as e:
        print(f"{'!' * 70}")
        print("COULD NOT CONNECT to the auth database.")
        print(f"Error: {e}")
        print()
        print("Common causes:")
        print("  - Postgres isn't installed or isn't running")
        print("  - AUTH_DATABASE_URL in .env doesn't match your real Postgres")
        print("    username/password/host/port/database name")
        print("  - The database itself doesn't exist yet (Postgres running,")
        print("    but you haven't created the specific database in the URL)")
        print()
        print("Windows quick-start (no Docker):")
        print("  1. Install PostgreSQL from https://www.postgresql.org/download/windows/")
        print("     (the installer lets you set the 'postgres' user's password directly)")
        print("  2. Open 'SQL Shell (psql)' from the Start menu, connect with the")
        print("     password you set, then run:")
        print("       CREATE DATABASE order_exception_agent;")
        print("  3. Update AUTH_DATABASE_URL in .env to match the password you set")
        print()
        print("Or, easier — use a free-tier cloud Postgres (no local install at all):")
        print("  Neon (neon.tech) or Supabase (supabase.com) both give a connection")
        print("  string in about a minute. Paste it in as AUTH_DATABASE_URL.")
        print(f"{'!' * 70}")
        sys.exit(1)

    init_auth_db()
    print("users table ready (created if it didn't already exist).")
    print()
    print("Next: create your first account at http://127.0.0.1:8000/signup")
    print("(after starting the server: uvicorn app.main:app --reload)")


if __name__ == "__main__":
    main()
