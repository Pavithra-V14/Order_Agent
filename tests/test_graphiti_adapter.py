"""
Tests for the Graphiti long-term memory integration.

Scope, stated honestly: Graphiti's add_episode() internally makes SEVERAL
distinct LLM calls (entity extraction, edge extraction, node/edge
deduplication), each against a schema Graphiti's own library code defines
internally - not a request/response contract this project controls the
way it controls Groq/Mistral/EasyPost's calls. Faithfully mocking all of
those steps would mean reverse-engineering Graphiti's internal,
version-fragile prompt contracts, which is a worse use of effort than the
verification it would buy. What IS tested here, for real:

  1. Kuzu (the embedded graph driver) actually works - real graph writes/
     reads, zero external services, zero mocking.
  2. The Graphiti client is constructed with the correct LLM/embedder/
     driver wiring from settings.
  3. The embedder adapter correctly bridges this project's own
     get_embedder() into Graphiti's abstract EmbedderClient interface,
     against the REAL TF-IDF embedder (no network needed for this part).
  4. episodic.py's routing: GROQ_API_KEY present -> delegates to Graphiti;
     absent -> uses the SQL substitute.
"""
import asyncio
import os
import shutil
import tempfile

import pytest

@pytest.fixture(autouse=True)
def cleanup_env():
    import app.memory.graphiti_adapter as ga
    ga.reset_graphiti_client()
    yield
    for var in ("GROQ_API_KEY", "NEO4J_URI", "NEO4J_PASSWORD", "MISTRAL_API_KEY"):
        os.environ.pop(var, None)
    from app.core.config import get_settings
    get_settings.cache_clear()
    ga.reset_graphiti_client()

def test_graphiti_unavailable_without_groq_key():
    os.environ.pop("GROQ_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.memory.graphiti_adapter import _is_graphiti_available
    assert _is_graphiti_available() is False

def test_graphiti_available_with_groq_key():
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.memory.graphiti_adapter import _is_graphiti_available
    assert _is_graphiti_available() is True

def test_get_graphiti_client_raises_clearly_without_groq_key():
    os.environ.pop("GROQ_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.memory.graphiti_adapter import _get_graphiti_client, reset_graphiti_client
    reset_graphiti_client()
    with pytest.raises(RuntimeError, match="GROQ_API_KEY not configured"):
        _get_graphiti_client()

def test_kuzu_embedded_driver_works_standalone_for_real():
    """The real proof that the local, no-server, no-Docker graph store
    genuinely works - a real Kuzu database is created and queried, zero
    mocking, zero external services."""
    import kuzu
    tmp_dir = tempfile.mkdtemp(prefix="test_kuzu_")
    db_path = f"{tmp_dir}/test.kz"  # a path Kuzu creates itself — NOT the pre-existing tmp_dir directly
    try:
        db = kuzu.Database(db_path)
        conn = kuzu.Connection(db)
        conn.execute("CREATE NODE TABLE TestEntity(id STRING, name STRING, PRIMARY KEY(id))")
        conn.execute("CREATE (n:TestEntity {id: '1', name: 'Test Customer'})")
        result = conn.execute("MATCH (n:TestEntity) WHERE n.id = '1' RETURN n.name")
        row = result.get_next()
        assert row[0] == "Test Customer"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def test_graphiti_client_constructed_with_kuzu_by_default(monkeypatch):
    """Confirms: with GROQ_API_KEY set and no NEO4J_URI, the client is
    built with the embedded Kuzu driver, not Neo4j."""
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    os.environ.pop("NEO4J_URI", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.memory.graphiti_adapter as ga
    ga.reset_graphiti_client()

    captured = {}

    class FakeGraphiti:
        def __init__(self, llm_client, embedder, graph_driver, cross_encoder=None):
            captured["driver_type"] = type(graph_driver).__name__

    monkeypatch.setattr("graphiti_core.Graphiti", FakeGraphiti)

    ga._get_graphiti_client()
    assert captured["driver_type"] == "KuzuDriver"

def test_graphiti_client_constructed_with_neo4j_when_configured(monkeypatch):
    """Confirms: with NEO4J_URI set, the client is built with the Neo4j
    driver instead of Kuzu - the cloud (Aura) path."""
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    os.environ["NEO4J_URI"] = "neo4j+s://fake-instance.databases.neo4j.io"
    os.environ["NEO4J_PASSWORD"] = "fake_password"
    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.memory.graphiti_adapter as ga
    ga.reset_graphiti_client()

    captured = {}

    class FakeGraphiti:
        def __init__(self, llm_client, embedder, graph_driver, cross_encoder=None):
            captured["driver_type"] = type(graph_driver).__name__

    monkeypatch.setattr("graphiti_core.Graphiti", FakeGraphiti)

    ga._get_graphiti_client()
    assert captured["driver_type"] == "Neo4jDriver"

def test_graphiti_client_uses_groq_llm_config(monkeypatch):
    """Confirms the LLM client is configured against Groq's real
    OpenAI-compatible endpoint with the configured API key."""
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key_xyz"
    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.memory.graphiti_adapter as ga
    ga.reset_graphiti_client()

    captured = {}

    class FakeGraphiti:
        def __init__(self, llm_client, embedder, graph_driver, cross_encoder=None):
            captured["llm_client"] = llm_client

    monkeypatch.setattr("graphiti_core.Graphiti", FakeGraphiti)

    ga._get_graphiti_client()
    llm_client = captured["llm_client"]
    assert llm_client.config.api_key == "gsk_fake_test_key_xyz"
    assert llm_client.config.base_url == "https://api.groq.com/openai/v1"

def test_embedder_adapter_bridges_to_real_tfidf_embedder():
    """Confirms _ProjectEmbedderAdapter correctly calls into this
    project's real get_embedder() (TF-IDF fallback, no network needed)
    and returns a plain list of floats, matching Graphiti's
    EmbedderClient contract."""
    os.environ.pop("MISTRAL_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.memory.graphiti_adapter import _ProjectEmbedderAdapter
    adapter = _ProjectEmbedderAdapter()
    adapter._embedder.fit(["some sample text", "another sample"])

    result = asyncio.run(adapter.create("some sample text"))

    assert isinstance(result, list)
    assert all(isinstance(x, float) for x in result)

def test_episodic_routes_to_sql_without_groq_key():
    """Regression: the SQL substitute must remain the default path when
    Graphiti isn't available."""
    os.environ.pop("GROQ_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.memory.episodic import _use_graphiti
    assert _use_graphiti() is False

def test_episodic_routes_to_graphiti_with_groq_key():
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.memory.episodic import _use_graphiti
    assert _use_graphiti() is True

def test_episodic_log_episode_delegates_to_graphiti_when_available(monkeypatch):
    """Confirms the actual delegation happens - not just that the
    routing flag is correct, but that log_episode() genuinely calls the
    Graphiti path instead of touching SQL at all when available."""
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    calls = {}

    def fake_log_episode_graphiti(customer_id, episode_type, content, occurred_at, case_id=None):
        calls["called"] = True
        calls["customer_id"] = customer_id
        return {"customer_id": customer_id, "episode_type": episode_type, "occurred_at": occurred_at.isoformat()}

    monkeypatch.setattr("app.memory.graphiti_adapter.log_episode_graphiti", fake_log_episode_graphiti)

    from app.memory.episodic import log_episode
    from datetime import datetime, timezone
    result = log_episode(db=None, customer_id="CUST-GRAPHITI-1", episode_type="case_resolved",
                          content={"exception_type": "return"}, occurred_at=datetime(2025, 6, 1, tzinfo=timezone.utc))

    assert calls["called"] is True
    assert calls["customer_id"] == "CUST-GRAPHITI-1"
    assert result["customer_id"] == "CUST-GRAPHITI-1"
