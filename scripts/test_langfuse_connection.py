"""
Standalone Langfuse connectivity test - creates a real trace/span with
your actual configured credentials and flushes it immediately, with
debug=True so any underlying error is printed directly rather than
silently swallowed into a warning log (which is what the real app does
in app/core/tracing.py's _push_to_langfuse — appropriate for production,
since a tracing failure should never crash a real request, but useless
for diagnosing WHY nothing shows up).

Usage:
    python3 scripts/test_langfuse_connection.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import get_settings


def main():
    settings = get_settings()

    print("=== Configuration check ===")
    print(f"TRACING_ENABLED: {settings.tracing_enabled}")
    print(f"LANGFUSE_PUBLIC_KEY set: {bool(settings.langfuse_public_key)}"
          + (f" (starts with {settings.langfuse_public_key[:7]}...)" if settings.langfuse_public_key else ""))
    print(f"LANGFUSE_SECRET_KEY set: {bool(settings.langfuse_secret_key)}"
          + (f" (starts with {settings.langfuse_secret_key[:7]}...)" if settings.langfuse_secret_key else ""))
    print(f"LANGFUSE_HOST: {settings.langfuse_host}")
    print()

    if not settings.tracing_enabled:
        print("FAILED: TRACING_ENABLED is not true. Set TRACING_ENABLED=true in .env and restart.")
        sys.exit(1)
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        print("FAILED: LANGFUSE_PUBLIC_KEY and/or LANGFUSE_SECRET_KEY not set.")
        sys.exit(1)

    # Public keys start with "pk-lf-", secret keys with "sk-lf-" - a
    # common, easy-to-make mistake is swapping them, which auths
    # incorrectly. Checked directly rather than assumed.
    if not settings.langfuse_public_key.startswith("pk-"):
        print(f"WARNING: LANGFUSE_PUBLIC_KEY doesn't start with 'pk-' - "
              f"got {settings.langfuse_public_key[:10]}... - check you haven't swapped public/secret keys.")
    if not settings.langfuse_secret_key.startswith("sk-"):
        print(f"WARNING: LANGFUSE_SECRET_KEY doesn't start with 'sk-' - "
              f"got {settings.langfuse_secret_key[:10]}... - check you haven't swapped public/secret keys.")

    print("=== Attempting a real trace + span + flush, with debug=True ===")
    from langfuse import Langfuse
    try:
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
            debug=True,
        )
        trace_id = client.create_trace_id(seed="diagnostic-test-trace")
        span = client.start_observation(
            trace_context={"trace_id": trace_id},
            name="diagnostic_test_span",
            input={"test": "input"},
            output={"test": "output"},
            metadata={"source": "test_langfuse_connection.py"},
        )
        span.end()
        client.flush()
        print()
        print("SUCCESS - no exception was raised. If you still don't see this trace in your")
        print("Langfuse dashboard within a minute, check:")
        print("  1. You're looking at the correct PROJECT in Langfuse (top-left project switcher)")
        print("  2. The trace search/filter isn't excluding it (try clearing all filters)")
        print(f"  3. Direct link pattern: {settings.langfuse_host}/project/<project-id>/traces")
    except Exception as e:
        print()
        print(f"FAILED: {type(e).__name__}: {e}")
        print()
        print("Common causes:")
        print("  - Wrong host: LANGFUSE_HOST must match where your project actually lives")
        print("    (cloud.langfuse.com for US region, or your self-hosted URL)")
        print("  - Invalid/revoked keys: regenerate them at your Langfuse project's")
        print("    Settings > API Keys page")
        print("  - Network/firewall blocking the connection to Langfuse's servers")
        sys.exit(1)


if __name__ == "__main__":
    main()
