"""
Tests for the Langfuse Cloud dual-write in app/core/tracing.py.

Unlike Groq/Mistral/EasyPost (plain per-call REST, cleanly mocked with
respx), Langfuse's SDK is OpenTelemetry-based with a batching exporter -
not a simple one-request-per-span transport. What IS directly testable
and tested here: (1) the client is constructed with the correct
credentials from settings, (2) record_span() NEVER raises even when the
Langfuse push fails - this sandbox has no route to cloud.langfuse.com, so
a real attempt genuinely fails, and that failure must stay non-fatal,
exactly as the module's docstring promises.
"""
import os
import tempfile

import pytest

@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_langfuse_{os.getpid()}_{id(object())}.db")
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}"
    from app.core.config import get_settings
    get_settings.cache_clear()
    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)
    db_module.init_db()
    yield db_module
    try:

        if os.path.exists(tmp_path):

            os.remove(tmp_path)

    except PermissionError:

        pass  # Windows may still hold a brief lock from engine cleanup; harmless to leave a stray temp file

@pytest.fixture(autouse=True)
def langfuse_env_cleanup():
    import app.core.tracing as tracing_module
    tracing_module.reset_langfuse_client()
    yield
    for var in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "TRACING_ENABLED"):
        os.environ.pop(var, None)
    from app.core.config import get_settings
    get_settings.cache_clear()
    tracing_module.reset_langfuse_client()

def test_langfuse_push_is_a_noop_when_tracing_disabled(isolated_db, monkeypatch):
    """The default state (TRACING_ENABLED=false) must never even attempt
    to construct a Langfuse client."""
    import app.core.tracing as tracing_module
    from app.core.config import get_settings
    get_settings.cache_clear()

    def _should_not_be_called():
        raise AssertionError("Langfuse client must not be constructed when tracing is disabled")
    monkeypatch.setattr(tracing_module, "_get_langfuse_client", _should_not_be_called)

    db = isolated_db.SessionLocal()
    span = tracing_module.record_span(db, "case-1", "test_agent", {"in": 1}, {"out": 2})
    assert span is not None
    db.close()

def test_langfuse_client_constructed_with_correct_credentials(isolated_db, monkeypatch):
    """Confirms settings are threaded through to the Langfuse client
    constructor correctly."""
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-fake-test"
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-fake-test"
    os.environ["TRACING_ENABLED"] = "true"
    from app.core.config import get_settings
    get_settings.cache_clear()

    captured = {}

    class FakeLangfuse:
        def __init__(self, public_key, secret_key, host):
            captured["public_key"] = public_key
            captured["secret_key"] = secret_key
            captured["host"] = host

    import app.core.tracing as tracing_module
    monkeypatch.setattr("langfuse.Langfuse", FakeLangfuse)
    tracing_module.reset_langfuse_client()

    tracing_module._get_langfuse_client()
    assert captured["public_key"] == "pk-fake-test"
    assert captured["secret_key"] == "sk-fake-test"
    assert captured["host"] == "https://cloud.langfuse.com"

def test_record_span_never_raises_even_when_langfuse_push_fails(isolated_db):
    """THE key property: with tracing enabled and fake credentials, a
    real attempt to reach cloud.langfuse.com from this sandbox genuinely
    fails (no network route) - record_span() must swallow that failure
    and still return the successfully-written SQL span."""
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-fake-test"
    os.environ["LANGFUSE_SECRET_KEY"] = "sk-fake-test"
    os.environ["TRACING_ENABLED"] = "true"
    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.core.tracing as tracing_module
    tracing_module.reset_langfuse_client()

    db = isolated_db.SessionLocal()
    span = tracing_module.record_span(db, "case-2", "test_agent", {"in": 1}, {"out": 2})
    assert span is not None

    from app.core.tracing import get_trace
    trace = get_trace(db, "case-2")
    assert len(trace) == 1
    assert trace[0]["agent_or_tool_name"] == "test_agent"
    db.close()
