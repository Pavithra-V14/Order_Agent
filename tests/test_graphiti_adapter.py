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


def test_graphiti_client_is_not_reused_across_separate_asyncio_run_calls(monkeypatch):
    """THE regression test for the actual root cause found in production:
    a persistent, globally-cached Graphiti client whose underlying async
    Neo4j driver gets bound to whichever asyncio event loop is active
    when first used. Since _run_async() calls asyncio.run() fresh each
    time (creating AND TEARING DOWN a new event loop per call), reusing
    one cached client across multiple SEPARATE _run_async() invocations —
    exactly what happens in a real orchestrator run, where diagnosis,
    fraud, and customer_context nodes each independently call into
    Graphiti — broke with "Unable to retrieve routing information" on
    the second and later calls, regardless of event loop TYPE (an
    earlier, incorrect fix attempt tried swapping Windows event loop
    implementations and made no difference, confirming the real bug was
    never about loop type).

    This test proves the fix: _build_graphiti_client() is called fresh
    inside EACH separate asyncio.run() invocation (never cached), so
    multiple sequential calls — simulating diagnosis -> fraud ->
    customer_context each calling into Graphiti in turn — must all
    succeed, using a genuinely NEW client instance each time.
    """
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    construction_count = {"n": 0}

    class FakeGraphitiTracksLoop:
        def __init__(self, llm_client, embedder, graph_driver, cross_encoder=None):
            construction_count["n"] += 1

        async def build_indices_and_constraints(self):
            pass

        async def add_episode(self, **kwargs):
            pass

        async def close(self):
            pass

    monkeypatch.setattr("graphiti_core.Graphiti", FakeGraphitiTracksLoop)

    import app.memory.graphiti_adapter as ga
    from datetime import datetime, timezone

    # Simulate 3 separate agent nodes each independently calling into
    # Graphiti - exactly the real orchestrator's call pattern.
    for i in range(3):
        ga.log_episode_graphiti("CUST-LOOP-TEST", "case_resolved", {"n": i},
                                 occurred_at=datetime(2025, 6, 1, tzinfo=timezone.utc))

    assert construction_count["n"] == 3, (
        "each call must construct a genuinely fresh client, never reuse one across separate "
        "asyncio.run() invocations — reusing one is the exact bug this test guards against, since "
        "a cached client's driver gets bound to whichever event loop was active at construction, "
        "and asyncio.run() tears down and recreates a new loop on every single call"
    )


def test_graphiti_client_is_closed_after_each_call(monkeypatch):
    """The other half of correct lifecycle management: since the client
    is no longer a persistent cached singleton, each fresh instance must
    be explicitly closed after use, or connections leak across the many
    short-lived clients this design now creates."""
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    close_count = {"n": 0}

    class FakeGraphitiTracksClose:
        def __init__(self, llm_client, embedder, graph_driver, cross_encoder=None):
            pass

        async def build_indices_and_constraints(self):
            pass

        async def add_episode(self, **kwargs):
            pass

        async def close(self):
            close_count["n"] += 1

    monkeypatch.setattr("graphiti_core.Graphiti", FakeGraphitiTracksClose)

    import app.memory.graphiti_adapter as ga
    from datetime import datetime, timezone

    ga.log_episode_graphiti("CUST-CLOSE-TEST", "case_resolved", {},
                             occurred_at=datetime(2025, 6, 1, tzinfo=timezone.utc))

    assert close_count["n"] == 1, "each fresh client must be explicitly closed after use"


def test_waits_for_neo4j_driver_background_init_task_before_proceeding():
    """THE regression test for the actual, confirmed root cause (found
    by reading graphiti-core's own source directly): Neo4jDriver.__init__
    schedules build_indices_and_constraints() as a fire-and-forget
    background asyncio.Task on construction (`loop.create_task(...)`,
    stored as `_init_task`), but nothing in Graphiti's own code makes a
    caller wait for it before using the driver. A real operation
    (retrieve_episodes, add_episode) starting to use the driver's
    connection pool WHILE that background task is still concurrently
    trying to establish routing/run its own queries on the SAME pool is
    exactly what produced "Unable to retrieve routing information" in
    production — confirmed as genuinely inside Graphiti's code, not this
    project's, by isolating the neo4j package's own async driver
    completely standalone (zero Graphiti) and confirming it connects
    successfully every single time.

    This test proves _wait_for_neo4j_driver_init() genuinely waits for a
    slow background task to complete before returning, rather than
    immediately returning while it's still pending.
    """
    import asyncio
    from app.memory.graphiti_adapter import _wait_for_neo4j_driver_init

    class FakeDriver:
        def __init__(self):
            self.completed = False

            async def slow_init():
                await asyncio.sleep(0.1)
                self.completed = True

            self._init_task = asyncio.get_event_loop().create_task(slow_init())

    class FakeClient:
        def __init__(self):
            self.driver = FakeDriver()

    async def run_test():
        client = FakeClient()
        assert client.driver.completed is False, "background task should not have finished yet"
        await _wait_for_neo4j_driver_init(client)
        assert client.driver.completed is True, (
            "must genuinely wait for the background init task to complete before returning — "
            "this is the exact race this fix closes"
        )

    asyncio.run(run_test())


def test_waits_is_a_noop_when_no_init_task_present():
    """The Kuzu backend (and any driver without this background-task
    pattern) has no _init_task attribute at all — must not raise."""
    import asyncio
    from app.memory.graphiti_adapter import _wait_for_neo4j_driver_init

    class FakeDriverNoInitTask:
        pass

    class FakeClient:
        def __init__(self):
            self.driver = FakeDriverNoInitTask()

    asyncio.run(_wait_for_neo4j_driver_init(FakeClient()))  # must not raise


def test_kuzu_driver_is_cached_across_multiple_sequential_calls(monkeypatch):
    """THE regression test for the actual production bug: with NEO4J_URI
    unset (Kuzu active), constructing a FRESH KuzuDriver on every single
    call — the exact strategy that correctly fixes Neo4j's bug — breaks
    Kuzu instead, because kuzu.Database() takes an EXCLUSIVE lock on its
    storage path and does not support re-opening it while a prior
    instance's lock hasn't been released. Confirmed directly: this
    failed in production with "IO exception: Could not set lock on file"
    the moment NEO4J_URI was unset and multiple LangGraph nodes each
    called into Graphiti in turn.

    This test uses the REAL KuzuDriver against a real temp path (only
    the Graphiti wrapper class itself is mocked, to avoid needing a real
    LLM call) — proving the actual file-locking behavior is exercised,
    not just a mocked stand-in that wouldn't reproduce the real bug.
    """
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    os.environ.pop("NEO4J_URI", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    tmp_dir = tempfile.mkdtemp(prefix="test_kuzu_cache_")
    kuzu_path = f"{tmp_dir}/test.kz"
    os.environ["KUZU_LOCAL_PATH"] = kuzu_path
    get_settings.cache_clear()

    import app.memory.graphiti_adapter as ga
    ga.reset_graphiti_client()

    captured_drivers = []

    class FakeGraphitiCapturesDriver:
        def __init__(self, llm_client, embedder, graph_driver, cross_encoder=None):
            captured_drivers.append(graph_driver)

    monkeypatch.setattr("graphiti_core.Graphiti", FakeGraphitiCapturesDriver)

    try:
        # Simulate 3 separate sequential calls — exactly what diagnosis,
        # fraud, and customer_context each independently trigger in a
        # real orchestrator run. With a REAL KuzuDriver, this would raise
        # "Could not set lock on file" on the second call if a fresh
        # instance were constructed each time instead of reusing one.
        for _ in range(3):
            ga._build_graphiti_client()

        assert len(captured_drivers) == 3
        assert captured_drivers[0] is captured_drivers[1] is captured_drivers[2], (
            "the SAME KuzuDriver instance must be reused across all 3 calls — constructing a "
            "fresh one each time is the exact bug that broke the real Kuzu file lock in production"
        )
    finally:
        os.environ.pop("KUZU_LOCAL_PATH", None)
        ga.reset_graphiti_client()
        shutil.rmtree(tmp_dir, ignore_errors=True)
