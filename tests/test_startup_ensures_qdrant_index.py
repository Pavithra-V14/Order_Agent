"""
Test for the app startup lifecycle's ensure_collection() call - the fix
for a real, recurring production bug: a Qdrant collection created
BEFORE the payload-index fix existed (or simply never re-ingested in a
given session) would keep hitting "Index required but not found" on
every search forever, since ensure_collection() was previously only
ever called from the ingestion path, never from search or startup.
"""
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_env():
    tmp_dir = tempfile.mkdtemp(prefix="test_startup_qdrant_")
    os.environ["QDRANT_LOCAL_PATH"] = os.path.join(tmp_dir, "qdrant")
    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.rag.vectorstore as vs
    if vs._client_singleton is not None:
        vs._client_singleton.close()
    vs._client_singleton = None
    vs._client_singleton_key = None

    yield tmp_dir

    os.environ.pop("QDRANT_LOCAL_PATH", None)
    get_settings.cache_clear()


def test_startup_calls_ensure_collection(isolated_env, monkeypatch):
    """THE regression test for the actual reported bug: ensure_collection()
    (which creates the payload indexes Qdrant Cloud requires for
    filtering) was previously only ever called from the ingestion path -
    a collection that already existed before that fix, or one that
    simply hasn't been re-ingested in a given session, would keep
    failing search forever with no way to recover short of re-running
    ingestion. This test proves the app's own startup lifecycle now
    calls ensure_collection() unconditionally, every time the app
    starts - not just conditionally on ingestion having run.

    NOTE: verifying actual Qdrant Cloud payload-index STATE isn't
    possible against this project's local embedded Qdrant default,
    since embedded mode treats create_payload_index() as a documented
    no-op with no inspectable index state at all (confirmed directly:
    Qdrant itself warns "Payload indexes have no effect in the local
    Qdrant" on every call). Verifying the CALL happens is the correct,
    backend-agnostic claim to test instead.
    """
    calls = []

    import app.main as main_module
    original_ensure_collection = main_module.ensure_collection if hasattr(main_module, "ensure_collection") else None

    def fake_ensure_collection(client, collection, dim):
        calls.append((collection, dim))

    monkeypatch.setattr("app.rag.vectorstore.ensure_collection", fake_ensure_collection)

    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app):
        pass  # entering/exiting the context runs the lifespan startup/shutdown

    assert len(calls) == 1, "ensure_collection() must be called exactly once during app startup"
    from app.core.config import get_settings
    assert calls[0][0] == get_settings().qdrant_collection


def test_startup_ensure_collection_failure_does_not_crash_the_app(isolated_env, monkeypatch):
    """A Qdrant connectivity problem at startup (e.g. Aura temporarily
    unreachable) must not prevent the app from starting at all - it's
    logged and the app continues, since most of the application doesn't
    depend on RAG being available at every single moment."""
    def failing_ensure_collection(client, collection, dim):
        raise RuntimeError("simulated Qdrant connectivity failure")

    monkeypatch.setattr("app.rag.vectorstore.ensure_collection", failing_ensure_collection)

    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200, "app must still start and serve requests even if ensure_collection fails at startup"
