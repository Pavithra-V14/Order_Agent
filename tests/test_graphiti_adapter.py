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


def test_graphiti_uses_a_dedicated_model_with_json_schema_mode(monkeypatch):
    """THE regression test for a real production bug found from an
    actual Neo4j write: a Pydantic validation error ("Field required:
    summaries") on Graphiti's own internal SummarizedEntities schema,
    traced to json_object mode only guaranteeing valid JSON, not that
    it matches Graphiti's exact expected shape.

    graphiti_router_model started as moonshotai/kimi-k2-instruct-0905
    (a model Groq documented as reliably supporting json_schema), but a
    real run against real Groq returned a 404 "model does not exist" —
    confirmed directly against Groq's own deprecation announcements:
    retired March 23, 2026 in favor of openai/gpt-oss-120b, alongside
    most other alternatives (Kimi K2, Llama 4 Maverick, Llama Guard 4,
    Qwen3-32B, Llama 4 Scout), all deprecated in favor of the same
    model. gpt-oss-120b is now the realistic, actively-maintained
    choice — this proves the dedicated setting and the stronger
    json_schema mode are genuinely wired in; whether this specific
    model enforces the schema reliably in practice is what the real
    log_episode_failure alerting (app/core/alerting.py) exists to
    surface, not something assumed here."""
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
    assert llm_client.config.model == "openai/gpt-oss-120b"
    assert llm_client.structured_output_mode == "json_schema"
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


def test_kuzu_driver_construction_is_thread_safe_under_real_concurrency():
    """THE regression test for why the previous fix (caching a Kuzu
    singleton) still broke in production: the cache check was a plain
    `if _kuzu_driver_singleton is None: construct()`, with no lock — a
    classic check-then-act race. LangGraph runs diagnosis, fraud,
    inventory, and customer_context IN PARALLEL via a thread pool
    (confirmed directly from a real production traceback's
    concurrent.futures.thread frame), so multiple threads genuinely call
    _build_graphiti_client() at the same time. Two threads can both see
    `is None` before either finishes constructing, and both proceed to
    open their own kuzu.Database() on the identical path simultaneously —
    reproducing "Could not set lock on file" from real concurrency, not
    just sequential calls (which is exactly why the earlier
    sequential-only test above did not catch this).

    This test uses REAL threads and the REAL KuzuDriver/kuzu.Database
    class (only the Graphiti wrapper itself is mocked, to avoid a real
    LLM call) — genuinely exercising the file-lock race, not simulating
    it. Before the threading.Lock() fix, this test reliably fails with
    the exact "Could not set lock on file" RuntimeError when run
    multiple times; with the fix, it passes reliably.
    """
    import threading

    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    os.environ.pop("NEO4J_URI", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    tmp_dir = tempfile.mkdtemp(prefix="test_kuzu_concurrent_")
    kuzu_path = f"{tmp_dir}/test.kz"
    os.environ["KUZU_LOCAL_PATH"] = kuzu_path
    get_settings.cache_clear()

    import app.memory.graphiti_adapter as ga
    ga.reset_graphiti_client()

    errors = []
    constructed_drivers = []
    lock = threading.Lock()

    def worker():
        try:
            driver = ga._get_or_create_kuzu_driver(kuzu_path)
            with lock:
                constructed_drivers.append(driver)
        except Exception as e:
            with lock:
                errors.append(e)

    try:
        # 8 threads hitting the SAME construction path at the same time —
        # deliberately more than LangGraph's actual 4 parallel nodes, to
        # stress the race harder than the minimum needed to reproduce it.
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, (
            f"Expected zero errors from concurrent construction, got: {errors} — "
            f"a 'Could not set lock on file' here means the race is NOT actually closed"
        )
        assert len(constructed_drivers) == 8
        assert len(set(id(d) for d in constructed_drivers)) == 1, (
            "all 8 threads must receive the exact SAME driver instance — any thread getting "
            "a DIFFERENT instance means it won the race and opened a second, conflicting lock"
        )
    finally:
        os.environ.pop("KUZU_LOCAL_PATH", None)
        ga.reset_graphiti_client()
        shutil.rmtree(tmp_dir, ignore_errors=True)


# --- Stage 1 memory upgrade: find_related_fraud_signals --------------------
#
# Cross-customer fraud query tests. Scope, stated honestly (same
# discipline as this file's own module docstring): what's verified here
# is (1) every "signal genuinely unavailable" path returns [] rather
# than raising, and (2) the Cypher/params sent to a MOCKED Neo4j driver
# are exactly what find_related_fraud_signals claims to send, and its
# result-parsing correctly reconstructs the expected dicts from a
# realistic mocked EagerResult shape. A real query against a live Neo4j
# Aura instance, containing real Graphiti-written Episodic nodes, is NOT
# run here — that requires a real Aura instance this sandbox cannot
# reach (see scripts/test_neo4j_connection.py for that verification,
# meant to be run against a real instance).

def test_find_related_fraud_signals_returns_empty_with_no_fingerprint():
    from app.memory.graphiti_adapter import find_related_fraud_signals
    assert find_related_fraud_signals(None, exclude_customer_id="CUST-1") == []
    assert find_related_fraud_signals("", exclude_customer_id="CUST-1") == []


def test_find_related_fraud_signals_returns_empty_without_groq_key():
    os.environ.pop("GROQ_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.memory.graphiti_adapter import find_related_fraud_signals
    assert find_related_fraud_signals("fp_abc123", exclude_customer_id="CUST-1") == []


def test_find_related_fraud_signals_returns_empty_on_kuzu_backend(caplog):
    """Groq key present (Graphiti active) but NEO4J_URI unset -> Kuzu is
    the active backend. This query is Neo4j-only by design (see the
    function's own docstring on why Kuzu's dialect isn't wired here) -
    must return [] with a clear log message, not silently do nothing
    and not attempt a Kuzu query with Neo4j Cypher syntax."""
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    os.environ.pop("NEO4J_URI", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.memory.graphiti_adapter import find_related_fraud_signals
    import logging
    with caplog.at_level(logging.INFO, logger="graphiti_adapter"):
        result = find_related_fraud_signals("fp_abc123", exclude_customer_id="CUST-1")
    assert result == []
    assert any("Neo4j Aura not configured" in r.message for r in caplog.records)


def test_find_related_fraud_signals_queries_neo4j_with_correct_params(monkeypatch):
    """The core mechanism test: with Neo4j configured, verifies (1) the
    exact Cypher query text and parameter dict sent to the driver match
    what the function's own docstring claims — group_id exclusion, the
    JSON substring markers for payment_fingerprint and
    episode_type=fraud_flag_raised — and (2) a realistic mocked
    EagerResult (list of dict-like records) is correctly turned into the
    expected [{"customer_id", "fraud_episode_id", "flagged_at"}, ...]
    shape."""
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    os.environ["NEO4J_URI"] = "neo4j+s://fake-instance.databases.neo4j.io"
    os.environ["NEO4J_PASSWORD"] = "fake_password"
    from app.core.config import get_settings
    get_settings.cache_clear()

    captured = {}

    class FakeRecord(dict):
        def __getitem__(self, key):
            return dict.__getitem__(self, key)

    class FakeEagerResult:
        def __init__(self, records):
            self.records = records

    class FakeNeo4jDriverForQuery:
        def __init__(self, uri, user, password, database):
            captured["construct_args"] = {"uri": uri, "user": user, "password": password, "database": database}

        async def execute_query(self, cypher_query_, params=None):
            captured["cypher"] = cypher_query_
            captured["params"] = params
            return FakeEagerResult([
                FakeRecord(other_customer_id="CUST-OTHER-1", fraud_episode_uuid="ep-uuid-1", flagged_at="2025-06-01T00:00:00Z"),
            ])

        async def close(self):
            captured["closed"] = True

    monkeypatch.setattr("graphiti_core.driver.neo4j_driver.Neo4jDriver", FakeNeo4jDriverForQuery)

    from app.memory.graphiti_adapter import find_related_fraud_signals
    result = find_related_fraud_signals("fp_shared_card_123", exclude_customer_id="CUST-UNDER-REVIEW")

    assert result == [
        {"customer_id": "CUST-OTHER-1", "fraud_episode_id": "ep-uuid-1", "flagged_at": "2025-06-01T00:00:00Z"},
    ]
    assert captured["closed"] is True
    assert captured["params"]["exclude_customer_id"] == "CUST-UNDER-REVIEW"
    assert captured["params"]["fingerprint_marker"] == '"payment_fingerprint": "fp_shared_card_123"'
    assert captured["params"]["fraud_marker"] == '"episode_type": "fraud_flag_raised"'
    # The query must exclude the reviewed customer's own group_id and
    # join through group_id, not through episode identity - a customer
    # can be flagged in a DIFFERENT case than the one carrying the
    # matching fingerprint.
    assert "matching_episode.group_id <> $exclude_customer_id" in captured["cypher"]
    assert "fraud_episode:Episodic {group_id: other_customer_id}" in captured["cypher"]


def test_find_related_fraud_signals_returns_empty_when_no_match(monkeypatch):
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    os.environ["NEO4J_URI"] = "neo4j+s://fake-instance.databases.neo4j.io"
    os.environ["NEO4J_PASSWORD"] = "fake_password"
    from app.core.config import get_settings
    get_settings.cache_clear()

    class FakeEagerResultEmpty:
        records = []

    class FakeNeo4jDriverNoMatch:
        def __init__(self, uri, user, password, database):
            pass

        async def execute_query(self, cypher_query_, params=None):
            return FakeEagerResultEmpty()

        async def close(self):
            pass

    monkeypatch.setattr("graphiti_core.driver.neo4j_driver.Neo4jDriver", FakeNeo4jDriverNoMatch)

    from app.memory.graphiti_adapter import find_related_fraud_signals
    assert find_related_fraud_signals("fp_unique_no_matches", exclude_customer_id="CUST-1") == []


def test_find_related_fraud_signals_swallows_a_real_query_error(monkeypatch, caplog):
    """A genuinely down/misconfigured Neo4j instance must degrade to 'no
    signal available', not propagate and break fraud scoring - same
    non-fatal discipline as every other memory-write path in this
    project (resolution_completion.py, orchestrator.py's
    fraud_flag_raised logging)."""
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    os.environ["NEO4J_URI"] = "neo4j+s://fake-instance.databases.neo4j.io"
    os.environ["NEO4J_PASSWORD"] = "fake_password"
    from app.core.config import get_settings
    get_settings.cache_clear()

    class FakeNeo4jDriverBroken:
        def __init__(self, uri, user, password, database):
            pass

        async def execute_query(self, cypher_query_, params=None):
            raise RuntimeError("Unable to retrieve routing information")

        async def close(self):
            pass

    monkeypatch.setattr("graphiti_core.driver.neo4j_driver.Neo4jDriver", FakeNeo4jDriverBroken)

    import logging
    from app.memory.graphiti_adapter import find_related_fraud_signals
    with caplog.at_level(logging.WARNING, logger="graphiti_adapter"):
        result = find_related_fraud_signals("fp_abc123", exclude_customer_id="CUST-1")
    assert result == []
    assert any("failed (non-fatal" in r.message for r in caplog.records)


# --- Genuine Graphiti-native cross-customer relations -----------------
#
# Answers the actual question this was built for: "does Graphiti's own
# extraction pipeline have relations across customers now?" - via a
# SEPARATE, opt-in shared group_id, not by changing per-customer
# episode isolation (unchanged, and confirmed unchanged by these tests).

def test_add_episode_default_group_id_is_still_customer_id(monkeypatch):
    """Regression guard: the group_id_override parameter must be
    opt-in only - every existing caller (log_episode_graphiti) that
    doesn't pass it must see IDENTICAL group_id behavior to before this
    change (per-customer isolation, unchanged)."""
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    captured = {}

    class FakeGraphitiCapturesGroupId:
        def __init__(self, llm_client, embedder, graph_driver, cross_encoder=None):
            pass

        async def build_indices_and_constraints(self):
            pass

        async def add_episode(self, **kwargs):
            captured["group_id"] = kwargs["group_id"]

        async def close(self):
            pass

    monkeypatch.setattr("graphiti_core.Graphiti", FakeGraphitiCapturesGroupId)

    import app.memory.graphiti_adapter as ga
    from datetime import datetime, timezone
    ga.log_episode_graphiti("CUST-DEFAULT-GROUP", "case_resolved", {"n": 1},
                             occurred_at=datetime(2025, 6, 1, tzinfo=timezone.utc))

    assert captured["group_id"] == "CUST-DEFAULT-GROUP"


def test_log_cross_customer_signal_uses_the_shared_group_id_not_customer_id(monkeypatch):
    """THE mechanism test: unlike every other episode this project
    writes, this one must land in the SHARED group_id - this is
    specifically what gives Graphiti's own dedup a chance to relate two
    DIFFERENT customers' mentions of the same signal, per
    log_cross_customer_signal's own docstring."""
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    captured = {}

    class FakeGraphitiCapturesGroupId:
        def __init__(self, llm_client, embedder, graph_driver, cross_encoder=None):
            pass

        async def build_indices_and_constraints(self):
            pass

        async def add_episode(self, **kwargs):
            captured["group_id"] = kwargs["group_id"]
            captured["episode_body"] = kwargs["episode_body"]

        async def close(self):
            pass

    monkeypatch.setattr("graphiti_core.Graphiti", FakeGraphitiCapturesGroupId)

    from app.memory.graphiti_adapter import log_cross_customer_signal, _CROSS_CUSTOMER_SIGNALS_GROUP_ID
    result = log_cross_customer_signal(
        customer_id="CUST-A", signal_type="payment_fingerprint", signal_value="fp_shared_123",
        case_id="case-1",
    )

    assert captured["group_id"] == _CROSS_CUSTOMER_SIGNALS_GROUP_ID
    assert captured["group_id"] != "CUST-A"
    assert "fp_shared_123" in captured["episode_body"]
    assert "CUST-A" in captured["episode_body"]
    assert result == {"customer_id": "CUST-A", "signal_type": "payment_fingerprint", "signal_value": "fp_shared_123"}


def test_log_cross_customer_signal_returns_empty_dict_without_groq_key():
    os.environ.pop("GROQ_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.memory.graphiti_adapter import log_cross_customer_signal
    assert log_cross_customer_signal("CUST-A", "payment_fingerprint", "fp_1") == {}


def test_search_cross_customer_relations_queries_the_shared_group_id(monkeypatch):
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    captured = {}

    class FakeEdge:
        def __init__(self, fact, name, uuid):
            self.fact, self.name, self.uuid = fact, name, uuid

    class FakeGraphitiSearch:
        def __init__(self, llm_client, embedder, graph_driver, cross_encoder=None):
            pass

        async def build_indices_and_constraints(self):
            pass

        async def search(self, query, group_ids=None, num_results=10):
            captured["query"] = query
            captured["group_ids"] = group_ids
            captured["num_results"] = num_results
            return [FakeEdge("Customer CUST-A shares a payment method with CUST-B", "RELATES_TO", "edge-uuid-1")]

        async def close(self):
            pass

    monkeypatch.setattr("graphiti_core.Graphiti", FakeGraphitiSearch)

    from app.memory.graphiti_adapter import search_cross_customer_relations, _CROSS_CUSTOMER_SIGNALS_GROUP_ID
    results = search_cross_customer_relations("fp_shared_123", num_results=5)

    assert captured["group_ids"] == [_CROSS_CUSTOMER_SIGNALS_GROUP_ID]
    assert captured["num_results"] == 5
    assert results == [{"fact": "Customer CUST-A shares a payment method with CUST-B",
                         "name": "RELATES_TO", "uuid": "edge-uuid-1"}]


def test_search_cross_customer_relations_returns_empty_on_search_error(monkeypatch):
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    class FakeGraphitiBrokenSearch:
        def __init__(self, llm_client, embedder, graph_driver, cross_encoder=None):
            pass

        async def build_indices_and_constraints(self):
            pass

        async def search(self, query, group_ids=None, num_results=10):
            raise RuntimeError("Neo4j unreachable")

        async def close(self):
            pass

    monkeypatch.setattr("graphiti_core.Graphiti", FakeGraphitiBrokenSearch)

    from app.memory.graphiti_adapter import search_cross_customer_relations
    assert search_cross_customer_relations("anything") == []


def test_search_cross_customer_relations_returns_empty_without_groq_key():
    os.environ.pop("GROQ_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.memory.graphiti_adapter import search_cross_customer_relations
    assert search_cross_customer_relations("anything") == []


def test_find_related_fraud_signals_converts_neo4j_datetime_to_json_safe_string(monkeypatch):
    """Regression test for a REAL bug found only against a live Neo4j
    Aura instance, not this sandbox's own mocked tests (which
    previously used a plain string for flagged_at and so never
    exercised this path): the neo4j Python driver returns its OWN
    temporal type (neo4j.time.DateTime) for a Cypher-returned
    timestamp, not a stdlib datetime - and it is not JSON-serializable.
    This crashed the whole aggregate_node commit (AuditLogEntry.detail
    being a JSON column) AFTER the fraud check had already correctly
    found a real cross-customer match. This test uses the REAL
    neo4j.time.DateTime class, not a fake stand-in, so it fails again
    immediately if the conversion is ever removed."""
    import json
    from neo4j.time import DateTime as Neo4jDateTime

    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    os.environ["NEO4J_URI"] = "neo4j+s://fake-instance.databases.neo4j.io"
    os.environ["NEO4J_PASSWORD"] = "fake_password"
    from app.core.config import get_settings
    get_settings.cache_clear()

    real_neo4j_datetime = Neo4jDateTime(2026, 9, 19, 5, 26, 34, 270228000)

    class FakeRecord(dict):
        def __getitem__(self, key):
            return dict.__getitem__(self, key)

    class FakeEagerResult:
        def __init__(self, records):
            self.records = records

    class FakeNeo4jDriverRealDatetime:
        def __init__(self, uri, user, password, database):
            pass

        async def execute_query(self, cypher_query_, params=None):
            return FakeEagerResult([
                FakeRecord(other_customer_id="CUST-OTHER", fraud_episode_uuid="ep-1",
                           flagged_at=real_neo4j_datetime),
            ])

        async def close(self):
            pass

    monkeypatch.setattr("graphiti_core.driver.neo4j_driver.Neo4jDriver", FakeNeo4jDriverRealDatetime)

    from app.memory.graphiti_adapter import find_related_fraud_signals
    result = find_related_fraud_signals("fp_abc123", exclude_customer_id="CUST-1")

    assert len(result) == 1
    assert isinstance(result[0]["flagged_at"], str), (
        "flagged_at must be converted to a plain string - a raw neo4j.time.DateTime "
        "object here is exactly what crashed AuditLogEntry.detail's JSON serialization"
    )
    json.dumps(result)  # must not raise - this is the actual failure mode found in production


# --- Redundant build_indices_and_constraints() call, found and removed ----
#
# Real bug found against a live deployment: a genuine 30s Graphiti
# timeout on log_episode(), traced to firing 31 separate Neo4j index/
# constraint queries TWICE per call (62 total) - once automatically via
# Neo4jDriver.__init__'s own background _init_task (confirmed by
# reading its actual source: it schedules build_indices_and_constraints()
# on construction), and once again via this project's own now-removed
# explicit call, immediately after _wait_for_neo4j_driver_init already
# awaited the first one. Graphiti's own docstring for
# build_indices_and_constraints says it "should typically be called
# once during initial setup" - not on every episode write.

def test_add_episode_does_not_redundantly_call_build_indices_and_constraints(monkeypatch):
    """The actual regression test: proves add_episode() no longer calls
    build_indices_and_constraints() a second time - the driver's own
    background _init_task (awaited via _wait_for_neo4j_driver_init) is
    the only place this runs now."""
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    call_count = {"n": 0}

    class FakeGraphitiTracksIndexCalls:
        def __init__(self, llm_client, embedder, graph_driver, cross_encoder=None):
            pass

        async def build_indices_and_constraints(self):
            call_count["n"] += 1

        async def add_episode(self, **kwargs):
            pass

        async def close(self):
            pass

    monkeypatch.setattr("graphiti_core.Graphiti", FakeGraphitiTracksIndexCalls)

    import app.memory.graphiti_adapter as ga
    from datetime import datetime, timezone
    ga.log_episode_graphiti("CUST-INDEX-TEST", "case_resolved", {"n": 1},
                             occurred_at=datetime(2025, 6, 1, tzinfo=timezone.utc))

    assert call_count["n"] == 0, (
        "add_episode() must not call build_indices_and_constraints() itself - "
        "the driver's own background _init_task already does this on construction"
    )
