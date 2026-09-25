"""
Long-term/episodic memory graph - real Graphiti implementation,
architecture doc 8.4. Two graph-store backends, chosen automatically:

  1. NEO4J_URI + NEO4J_PASSWORD set -> Neo4j Aura (cloud, free tier, no
     Docker). Real bolt-protocol connection via Graphiti's built-in
     Neo4jDriver. THIS IS THE RECOMMENDED PATH — see the deprecation
     note below.
  2. Neither set -> Kuzu, an EMBEDDED graph database (like SQLite is to
     relational DBs) - no server, no Docker, no credentials at all,
     writes to kuzu_local_path. Local-first default for zero-setup
     testing ONLY.

  IMPORTANT, discovered while building this: Graphiti's own KuzuDriver
  emits "The Kuzu backend is deprecated and will be removed in a future
  release — the upstream Kuzu project is no longer maintained. Migrate
  to Neo4j or FalkorDB." at construction time. This wasn't known before
  actually running the integration — Kuzu looked like the ideal
  zero-setup local option (genuinely embedded, no server, unlike Neo4j/
  FalkorDB), but Graphiti's own maintainers are moving away from it. It
  is kept here, working, because it's still the only true zero-service
  local option for development/testing without any credentials — but
  Neo4j Aura (free tier, no Docker, ~2 minutes to set up) is the
  actually-recommended path for anything beyond quick local testing, not
  just an optional cloud upgrade. Documented here rather than silently
  presenting a deprecated dependency as an equally-good permanent choice.

Either way, Graphiti activates ONLY when GROQ_API_KEY is also set:
Graphiti's whole value (entity/relationship extraction from episode
text) requires a real LLM call, so "Graphiti with no LLM" isn't a
meaningful mode - without a Groq key, app/memory/episodic.py's
SQL-backed functions are used instead.

LLM client: Graphiti's own OpenAIGenericClient (bundled - targets any
OpenAI-compatible /chat/completions endpoint) pointed at Groq's real
OpenAI-compatible API, in json_object structured-output mode (Groq
doesn't support strict json_schema constrained decoding).

Embedder: a thin adapter around this project's own get_embedder()
(app/rag/embeddings.py) - Mistral when configured, TF-IDF fallback
otherwise - so Graphiti's embeddings come from the exact same place
every other embedding in this project does.

Not network-tested end-to-end from this sandbox for the Neo4j Aura path
(no route to a real Aura instance) - but the Kuzu (embedded) path runs
for real with zero external services (tests/test_graphiti_adapter.py's
test_kuzu_embedded_driver_works_standalone_for_real), and Graphiti's
internal multi-step LLM extraction pipeline (entity/edge extraction,
deduplication — several distinct calls with schemas Graphiti's own
library code defines, not a contract this project controls) is NOT
mocked end-to-end — see that test file's module docstring for why that
specific boundary was drawn where it was.
"""
from __future__ import annotations

import asyncio
import json
import threading
from datetime import datetime, timezone

from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.cross_encoder.client import CrossEncoderClient


def _is_graphiti_available() -> bool:
    from app.core.config import get_settings
    settings = get_settings()
    return bool(settings.groq_api_key)


class _ProjectEmbedderAdapter(EmbedderClient):
    """Adapts this project's BaseEmbedder into Graphiti's abstract
    EmbedderClient interface - Graphiti's embeddings come from the same
    MistralEmbedder/TfidfEmbedder every other part of this project uses.

    MUST actually subclass EmbedderClient, not just duck-type its
    methods — Graphiti's internal GraphitiClients (a pydantic model)
    validates with `isinstance`, not structural typing. A duck-typed
    class passes every method-signature check but still fails
    construction with a ValidationError - caught only by actually
    constructing a real (non-mocked) Graphiti instance, which an earlier
    version of this integration's tests never did (they mocked Graphiti
    itself, which hid this exact class of bug).
    """

    def __init__(self):
        from app.rag.embeddings import get_embedder
        self._embedder = get_embedder()

    async def create(self, input_data) -> list:
        if isinstance(input_data, str):
            input_data = [input_data]
        vectors = self._embedder.embed(list(input_data))
        return vectors[0].tolist()

    async def create_batch(self, input_data_list: list) -> list:
        vectors = self._embedder.embed(input_data_list)
        return [v.tolist() for v in vectors]


class _BM25CrossEncoder(CrossEncoderClient):
    """Real local reranker (rank_bm25, already a dependency for the RAG
    pipeline's own hybrid search) - NOT Graphiti's default
    OpenAIRerankerClient.

    Found and fixed during production testing: Graphiti's Graphiti()
    constructor defaults `cross_encoder` to OpenAIRerankerClient() when
    not explicitly passed, which requires a real OPENAI_API_KEY
    regardless of whatever LLM client (Groq, here) is configured for the
    main entity-extraction path — a genuine "Missing credentials"
    crash in production for anyone who (correctly, per this project's
    design) never configured a real OpenAI key.

    Pointing OpenAIRerankerClient at Groq's endpoint instead (same
    LLMConfig pattern as the main llm_client) was considered and
    rejected: that reranker's rank() method hardcodes `logit_bias` values
    tied to OpenAI's SPECIFIC tokenizer's token IDs for "True"/"False" —
    against Groq/Llama's different tokenizer, those IDs bias toward
    unrelated tokens, so it would silently produce wrong rankings, not
    just an error. A real, provider-agnostic local reranker is the
    correct fix, not a same-shaped call to a different endpoint.

    MUST actually subclass CrossEncoderClient, same isinstance-validation
    reason as _ProjectEmbedderAdapter above.
    """

    def rank_sync(self, query: str, passages: list) -> list:
        from rank_bm25 import BM25Okapi
        if not passages:
            return []
        tokenized = [p.lower().split() for p in passages]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
        return ranked

    async def rank(self, query: str, passages: list) -> list:
        return self.rank_sync(query, passages)


_kuzu_driver_singleton = None
_kuzu_driver_lock = threading.Lock()


def _build_graphiti_client():
    """Constructs a Graphiti client. Two DIFFERENT lifecycle strategies
    for the graph driver, deliberately not the same for both backends —
    found necessary by hitting two DIFFERENT, unrelated bugs from
    treating them identically:

    NEO4J: constructed FRESH on every call, never cached. Its async
    driver's internal connection pool/routing-table state gets bound to
    whichever asyncio event loop is active when first used — since
    _run_async() calls asyncio.run() fresh each time (creating AND
    TEARING DOWN a new event loop per call), a cached Neo4j driver first
    used under event-loop #1 silently breaks the moment a later,
    separate call runs under event-loop #2. Confirmed as the actual
    mechanism by isolating the neo4j package's own async driver with
    zero Graphiti involvement and it connecting successfully every time.

    KUZU: cached as a PROCESS-WIDE SINGLETON, the OPPOSITE strategy —
    constructing a fresh KuzuDriver on every call (matching Neo4j's
    approach) breaks Kuzu specifically, because it's an embedded,
    file-locked database: opening a `kuzu.Database()` takes an EXCLUSIVE
    lock on its storage path, and Kuzu does not support multiple
    Database instances (even sequential, non-overlapping ones, if the
    previous one's lock hasn't been released yet by the time the next
    tries to open) against the same path. Confirmed directly: "always
    construct fresh" — the exact fix that resolved Neo4j's bug —
    immediately broke Kuzu with "IO exception: Could not set lock on
    file" once NEO4J_URI was unset and Kuzu became the active backend.
    Kuzu's underlying engine is a synchronous, purely local C++ library
    (no network sockets, no event-loop-bound async state), so caching it
    across separate asyncio.run() calls is safe in exactly the way
    caching Neo4j's driver is not.
    """
    from app.core.config import get_settings
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured - Graphiti requires an LLM for entity extraction.")

    # Graphiti sends anonymous usage telemetry to PostHog by default on
    # every call — a genuine surprise network dependency nobody opted
    # into, found because this sandbox's network allowlist blocked it
    # loudly (non-fatal, but noisy). Disabled here programmatically so
    # nobody needs to know this obscure env var exists.
    import os
    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")

    from graphiti_core import Graphiti
    from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
    from graphiti_core.llm_client.config import LLMConfig

    llm_client = OpenAIGenericClient(
        config=LLMConfig(
            api_key=settings.groq_api_key,
            model=settings.graphiti_router_model,
            base_url="https://api.groq.com/openai/v1",
        ),
        # json_schema (graphiti_core's own actual default) — NOT
        # json_object. Found and fixed a real production bug directly:
        # this was previously forced to json_object specifically because
        # the main pipeline's model (gpt-oss-120b) doesn't reliably
        # support json_schema for this kind of call — but json_object
        # only guarantees VALID JSON, not that it matches the exact
        # shape Graphiti's own internal schemas (e.g. SummarizedEntities)
        # require, causing a real, reproducible Pydantic validation
        # failure ("Field required: summaries") during a real Neo4j
        # write. graphiti_router_model is a model Groq documents as
        # reliably supporting json_schema, so this can now use the
        # stronger, schema-enforced mode instead of guessing.
        structured_output_mode="json_schema",
    )
    embedder = _ProjectEmbedderAdapter()

    if settings.neo4j_uri:
        from graphiti_core.driver.neo4j_driver import Neo4jDriver
        driver = Neo4jDriver(
            uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
    else:
        driver = _get_or_create_kuzu_driver(settings.kuzu_local_path)

    return Graphiti(
        llm_client=llm_client, embedder=embedder, graph_driver=driver, cross_encoder=_BM25CrossEncoder(),
    )


def _get_or_create_kuzu_driver(kuzu_local_path: str):
    """Thread-safe singleton construction — the actual bug that broke
    the previous fix in production. LangGraph runs diagnosis, fraud,
    inventory, and customer_context IN PARALLEL via a thread pool
    (confirmed from the real traceback's concurrent.futures.thread
    frame), so multiple threads can call _build_graphiti_client()
    genuinely concurrently. The earlier fix's plain
    `if _kuzu_driver_singleton is None: construct()` check has a classic
    check-then-act race: two threads can both see None before either has
    finished constructing, and both proceed to open their own
    kuzu.Database() on the identical path at the same time — the exact
    "Could not set lock on file" failure, just from real thread
    concurrency rather than sequential calls (which is why the earlier
    sequential-only regression test didn't catch this). Fixed with the
    standard double-checked-locking pattern: re-check inside the lock
    before constructing, so only ever exactly one thread wins the race.
    """
    global _kuzu_driver_singleton
    if _kuzu_driver_singleton is not None:
        return _kuzu_driver_singleton

    with _kuzu_driver_lock:
        # Re-check after acquiring the lock — another thread may have
        # already constructed it while this one was waiting.
        if _kuzu_driver_singleton is None:
            from graphiti_core.driver.kuzu_driver import KuzuDriver
            # NOTE: do NOT pre-create kuzu_local_path as a directory —
            # Kuzu's Database() creates its own storage AT that exact
            # path (a file, not a directory) and raises "Database path
            # cannot be a directory" if something already exists there.
            _kuzu_driver_singleton = KuzuDriver(db=kuzu_local_path)
        return _kuzu_driver_singleton


def _get_graphiti_client():
    """Backward-compatible alias — see _build_graphiti_client's docstring
    for why the Neo4j driver no longer caches a persistent client (the
    Kuzu driver still does, via a separate mechanism)."""
    return _build_graphiti_client()


def reset_graphiti_client() -> None:
    """Test helper — resets the Kuzu driver singleton (the only
    persistent state left after the Neo4j caching fix)."""
    global _kuzu_driver_singleton
    _kuzu_driver_singleton = None


GRAPHITI_CALL_TIMEOUT_SECONDS = 30.0


def _run_async(coro):
    """Bridges Graphiti's async API into this project's entirely-sync
    codebase. Safe here since nothing in this project runs its own event
    loop that this would conflict with.

    Wrapped in a bounded timeout — nothing in this bridge should be
    allowed to hang unbounded regardless of which external dependency
    (Neo4j or Groq) stalls; an earlier version of this function let a
    Neo4j connectivity problem retry for MINUTES with escalating delays
    before finally raising.

    Raised from 15.0s to 30.0s after a real, reproducible timeout: a
    genuine production run hit this exact timeout even though Neo4j
    Aura and Groq were BOTH independently confirmed reachable and
    correctly configured moments later (via scripts/test_neo4j_connection.py
    and scripts/test_groq_connection.py) — meaning 15s was too tight for
    what this call actually does, not a sign of misconfiguration. Every
    Graphiti call builds a brand-new Neo4j driver connection from
    scratch (by design — see _build_graphiti_client()'s docstring on
    event-loop safety), waits for its index/constraint setup to
    complete, then runs the actual query — three real network
    round-trips to a free-tier Aura instance, whose latency can
    genuinely vary, all inside one timeout window. 30s gives that
    realistic headroom while still bounding the wait to something
    reasonable, not unbounded.
    """
    try:
        return asyncio.run(asyncio.wait_for(coro, timeout=GRAPHITI_CALL_TIMEOUT_SECONDS))
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"Graphiti call did not complete within {GRAPHITI_CALL_TIMEOUT_SECONDS}s — this usually "
            f"means either Neo4j (NEO4J_URI) or Groq (GROQ_API_KEY) is unreachable or misconfigured. "
            f"But if scripts/test_neo4j_connection.py and scripts/test_groq_connection.py BOTH "
            f"succeed independently, this is more likely a genuine transient slowdown (a free-tier "
            f"Aura instance waking from an idle/paused state, or momentary network latency) than a "
            f"real misconfiguration — simply retrying often succeeds."
        ) from None


async def _wait_for_neo4j_driver_init(client) -> None:
    """Works around a genuine race condition inside graphiti-core's own
    Neo4jDriver.__init__ (confirmed by reading its source directly):
    construction schedules build_indices_and_constraints() as a
    fire-and-forget background asyncio.Task via
    `loop.create_task(...)`, stored on the driver as `_init_task` — but
    nothing in Graphiti's own code makes any CALLER wait for that task
    before using the driver for a real operation. This is exactly the
    "Unable to retrieve routing information" failures seen in practice:
    a real operation (retrieve_episodes, add_episode) starts using the
    driver's connection pool WHILE that background init task is still
    concurrently trying to establish routing/run its own index-creation
    queries on the SAME pool, racing against each other. Confirmed this
    is genuinely inside Graphiti's code, not this project's or the
    underlying neo4j package's, by isolating the neo4j async driver
    completely standalone (zero Graphiti involvement) and confirming it
    connects successfully every time — the bug only appears once
    Graphiti's Neo4jDriver wrapper is involved.

    Awaiting client.driver._init_task here — reaching into a private
    attribute, acknowledged — closes the race by ensuring the background
    init genuinely finishes before this project's own code does anything
    else with the client. A no-op for the Kuzu backend (which has no
    such background task) since the attribute simply won't be present.
    """
    init_task = getattr(getattr(client, "driver", None), "_init_task", None)
    if init_task is not None:
        await init_task


async def _add_episode_async(customer_id: str, episode_type: str, content: dict,
                              occurred_at: datetime, case_id: str = None,
                              group_id_override: str = None) -> None:
    """group_id_override (default None -> uses customer_id, the existing,
    unchanged per-customer isolation): lets a caller deliberately opt an
    episode INTO a shared group_id instead - see
    log_cross_customer_signal() below for why this exists and what it's
    for. Every other caller in this codebase passes no override and
    sees identical behavior to before."""
    client = _build_graphiti_client()
    try:
        # NOT calling client.build_indices_and_constraints() here -
        # found, by reading Neo4jDriver.__init__'s actual source, to be
        # pure redundancy causing a real, confirmed problem: the driver
        # ALREADY schedules this exact operation (31 separate index/
        # constraint queries, fired concurrently) as a background task
        # on construction (self._init_task, see
        # _wait_for_neo4j_driver_init's own docstring for the race this
        # project already found and fixed around that task). Since a
        # FRESH driver is built on every single call here, an explicit
        # second call meant this project was firing 62 index queries
        # against Neo4j on every single log_episode() - not 31 - which
        # is exactly the kind of load that can push a real 30s timeout
        # on a free-tier Aura instance from "usually fine" to "fails
        # under real use," and a strong candidate for the
        # 'Neo4jDriver._execute_index_query was never awaited' warning
        # observed in practice (two concurrent full index-build runs
        # racing on the same connection pool). _wait_for_neo4j_driver_
        # init already awaits the ONE the driver does on its own -
        # that's sufficient, and matches Graphiti's own documented
        # intent (build_indices_and_constraints' docstring: "should
        # typically be called once during initial setup," not per-call).
        await _wait_for_neo4j_driver_init(client)
        episode_body = json.dumps({"episode_type": episode_type, "content": content, "case_id": case_id})
        await client.add_episode(
            name=episode_type,
            episode_body=episode_body,
            source_description=f"case:{case_id}" if case_id else "system",
            reference_time=occurred_at,
            group_id=group_id_override or customer_id,
        )
    finally:
        await client.close()


async def _get_customer_history_async(customer_id: str, episode_type: str = None, limit: int = 20) -> list:
    client = _build_graphiti_client()
    try:
        await _wait_for_neo4j_driver_init(client)
        nodes = await client.retrieve_episodes(
            reference_time=datetime.now(timezone.utc), last_n=limit, group_ids=[customer_id],
        )
    finally:
        await client.close()
    results = []
    for node in nodes:
        try:
            parsed = json.loads(node.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if episode_type and parsed.get("episode_type") != episode_type:
            continue
        results.append({
            "id": str(node.uuid),
            "case_id": parsed.get("case_id"),
            "episode_type": parsed.get("episode_type"),
            "content": parsed.get("content"),
            "occurred_at": node.valid_at.isoformat() if node.valid_at else None,
        })
    results.sort(key=lambda h: h["occurred_at"] or "", reverse=True)
    return results


# --- Genuine Graphiti-native cross-customer relations -----------------------
#
# The question this answers: "does Graphiti's OWN extraction pipeline
# have relations across customers now?" Answer, stated precisely: not
# through the normal per-customer episode path (group_id=customer_id
# still isolates every customer's graph from every other's, by design -
# unchanged, and unfixable without abandoning that isolation entirely).
# What's added here is a SEPARATE, PARALLEL, opt-in path: episodes about
# a cross-customer SIGNAL (e.g. a payment fingerprint) are logged under
# one SHARED group_id instead of the customer's own - and Graphiti's
# dedup (see find_related_fraud_signals's docstring for how it was
# verified to scope by group_id) then operates WITHIN that one shared
# space, so it genuinely CAN merge/relate entities extracted from
# different customers' signal mentions there. This is real, not a
# rename of the Stage 1 deterministic query - the two are complementary,
# not the same mechanism, and are kept clearly separate below.
#
# WHY THIS DOESN'T REPLACE find_related_fraud_signals() AS THE
# AUTHORITATIVE FRAUD CHECK: relying on an LLM to correctly and
# consistently extract "this is the same payment method" as one
# identically-named entity across separately-written episodes is
# inherently non-deterministic - exactly the reason Stage 1 avoided it
# for the fraud decision itself. So this graph is populated as a real,
# genuine best-effort signal for human/audit exploration (surfaced
# alongside a case, not silently folded into an automated risk score),
# while the exact-match Cypher query remains the one thing the fraud
# agent's automated decision actually depends on. Tier 1's own
# "guardrails can't be gamed by fuzzy LLM output" principle applies
# here too: a non-deterministic relationship shouldn't silently drive
# an automated action ceiling.

_CROSS_CUSTOMER_SIGNALS_GROUP_ID = "cross_customer_signals"


def log_cross_customer_signal(customer_id: str, signal_type: str, signal_value: str,
                               case_id: str = None, occurred_at: datetime = None) -> dict:
    """Logs one occurrence of a shareable signal (e.g. a payment
    fingerprint) under the SHARED cross-customer group_id, not the
    customer's own - so Graphiti's OWN entity extraction and dedup,
    operating within that one shared space, has a genuine chance to
    recognize the SAME signal_value mentioned by a DIFFERENT customer
    later and relate the two. The episode content explicitly names both
    the customer and the signal ("customer CUST-X used payment method
    fp_abc") specifically to give the extraction LLM a concrete, minimal
    text to work from, rather than raw JSON alone.

    Returns {} (never raises) when Graphiti isn't configured (no
    GROQ_API_KEY) - there is no meaningful shared-graph signal to log in
    that case, same "unavailable is a normal state" discipline as
    find_related_fraud_signals()."""
    if not _is_graphiti_available():
        return {}
    occurred_at = occurred_at or datetime.now(timezone.utc)
    content = {
        "customer_id": customer_id, "signal_type": signal_type, "signal_value": signal_value,
        "narrative": f"Customer {customer_id} is associated with {signal_type} {signal_value}.",
    }
    _run_async(_add_episode_async(
        customer_id=customer_id, episode_type="cross_customer_signal", content=content,
        occurred_at=occurred_at, case_id=case_id, group_id_override=_CROSS_CUSTOMER_SIGNALS_GROUP_ID,
    ))
    return {"customer_id": customer_id, "signal_type": signal_type, "signal_value": signal_value}


def log_cross_customer_signal_async(customer_id: str, signal_type: str, signal_value: str,
                                     case_id: str = None) -> str | None:
    """Async wrapper - same job-queue pattern as
    app/memory/episodic.py's log_episode_async(), and for the identical
    reason (a Graphiti/Neo4j/Groq round trip must not sit on the case-
    resolution/fraud-decision hot path). Returns None (not a job_id)
    when Graphiti isn't configured - nothing meaningful to enqueue, same
    as log_cross_customer_signal()'s own {} return in that case."""
    if not _is_graphiti_available():
        return None
    from app.workers.job_queue import get_job_queue
    from app.workers.handlers import handle_log_cross_customer_signal
    q = get_job_queue()
    q.register_handler("log_cross_customer_signal", handle_log_cross_customer_signal)
    q.start_worker()
    return q.enqueue("log_cross_customer_signal", {
        "customer_id": customer_id, "signal_type": signal_type, "signal_value": signal_value,
        "case_id": case_id,
    })


def search_cross_customer_relations(query_text: str, num_results: int = 10) -> list[dict]:
    """Genuine Graphiti-native relationship search, scoped to the shared
    cross-customer group (see log_cross_customer_signal above) - this is
    what actually answers "does Graphiti have relations now": a real
    call to Graphiti's own hybrid search (semantic + BM25 + graph
    traversal), not a raw Cypher query. Returns the matching facts
    Graphiti's extraction found, as plain dicts.

    Best-effort by design - returns [] (never raises) when Graphiti
    isn't configured, or on any real search error (a down Neo4j
    instance, a malformed query), logged but non-fatal. This is
    exploratory/audit context, never something an automated decision
    should depend on existing - see this section's module comment for
    why."""
    import logging
    logger = logging.getLogger("graphiti_adapter")
    if not _is_graphiti_available():
        return []
    try:
        return _run_async(_search_cross_customer_relations_async(query_text, num_results))
    except Exception as e:
        logger.warning("search_cross_customer_relations failed (non-fatal, returning no signal): %s", e)
        return []


async def _search_cross_customer_relations_async(query_text: str, num_results: int) -> list[dict]:
    client = _build_graphiti_client()
    try:
        await _wait_for_neo4j_driver_init(client)
        edges = await client.search(
            query_text, group_ids=[_CROSS_CUSTOMER_SIGNALS_GROUP_ID], num_results=num_results,
        )
        return [{"fact": edge.fact, "name": edge.name, "uuid": edge.uuid} for edge in edges]
    finally:
        await client.close()


def log_episode_graphiti(customer_id: str, episode_type: str, content: dict,
                          occurred_at: datetime, case_id: str = None) -> dict:
    _run_async(_add_episode_async(customer_id, episode_type, content, occurred_at, case_id))
    return {"customer_id": customer_id, "episode_type": episode_type, "occurred_at": occurred_at.isoformat()}


def get_customer_history_graphiti(customer_id: str, episode_type: str = None, limit: int = 20) -> list:
    return _run_async(_get_customer_history_async(customer_id, episode_type, limit))


# --- Cross-customer fraud query (Stage 1 memory upgrade) -------------------
#
# Real multi-hop, CROSS-CUSTOMER fraud query — deliberately NOT built on
# Graphiti's own automatic entity extraction/dedup. Two independent
# reasons, found by reading graphiti-core 0.30.2's actual source rather
# than assuming:
#
#   1. Graphiti's entity/edge deduplication scopes its candidate search
#      to the episode's own group_id (confirmed directly in
#      graphiti_core/utils/maintenance/edge_operations.py:
#      `group_ids=[extracted_edge.group_id]`) — and this project sets
#      group_id=customer_id per episode (see _add_episode_async below).
#      Two different customers' episodes therefore NEVER get their
#      extracted entities merged/linked automatically; each customer's
#      graph is its own isolated island BY DESIGN (group_id is Graphiti's
#      multi-tenant isolation mechanism). A cross-customer fraud link
#      cannot be discovered through Graphiti's own extraction pipeline as
#      currently configured, no matter what text is fed into it.
#   2. Even setting that aside, relying on an LLM to consistently extract
#      "the same card" as an identically-named entity across separately
#      -written episodes is inherently non-deterministic — a real fraud
#      check needs an exact match, not a "the model probably phrased it
#      the same way" match.
#
# So this queries the underlying Episodic nodes DIRECTLY — the raw
# content this project's own code writes deterministically (see
# _add_episode_async's json.dumps call), not the fuzzy Entity graph
# Graphiti extracts on top of it — via a direct Cypher call through the
# real Neo4j driver. Verified against graphiti-core 0.30.2's actual (not
# assumed) Neo4j schema by reading node_db_queries.py/edge_db_queries.py
# directly: Episodic nodes carry {uuid, name, group_id, content, ...},
# where `content` is exactly the JSON string this project writes.
# MENTIONS/RELATES_TO edges (the LLM-extracted layer) are not involved
# in this query at all.
#
# NEO4J-ONLY: Kuzu (the embedded, dev-only fallback) uses a different
# query dialect and is not wired here — this returns [] with a clear log
# message rather than silently doing nothing, so the gap is visible
# rather than mistaken for "no related fraud found." Also returns []
# (silently — this is the normal/expected state, not an error) when
# Graphiti/Groq isn't configured at all. The fraud agent
# (app/agents/workflow_agents.py) must never break or change its
# behavior just because this signal happens to be unavailable.

_RELATED_FRAUD_SIGNALS_QUERY = """
MATCH (matching_episode:Episodic)
WHERE matching_episode.group_id <> $exclude_customer_id
  AND matching_episode.content CONTAINS $fingerprint_marker
WITH DISTINCT matching_episode.group_id AS other_customer_id
MATCH (fraud_episode:Episodic {group_id: other_customer_id})
WHERE fraud_episode.content CONTAINS $fraud_marker
RETURN DISTINCT other_customer_id,
       fraud_episode.uuid AS fraud_episode_uuid,
       fraud_episode.created_at AS flagged_at
"""


def _to_iso_string(value) -> str | None:
    """Converts a timestamp value from whatever the active graph
    backend's driver returns into a plain, JSON-safe ISO string.
    Handles every real case seen so far, checked - not guessed -
    against real drivers: neo4j.time.DateTime (the real Neo4j Aura
    driver's own temporal type, which has its own .iso_format() and is
    NOT a stdlib datetime, found the hard way against a live Aura
    instance - see the docstring at this function's call site), a
    plain stdlib datetime.datetime, an already-a-string value, or None
    (no timestamp on the row). Never raises - a formatting quirk in a
    timestamp must not break the whole fraud check; falls back to
    str(value) for anything unrecognized rather than losing the field
    entirely."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if hasattr(value, "iso_format"):  # neo4j.time.DateTime and similar
        return value.iso_format()
    if hasattr(value, "isoformat"):  # stdlib datetime.datetime
        return value.isoformat()
    return str(value)


def find_related_fraud_signals(payment_fingerprint: str, exclude_customer_id: str) -> list[dict]:
    """Given a payment fingerprint (see app/tools/oms.py's
    compute_payment_fingerprint), finds OTHER customers (excluding the
    one currently under review) who have ever logged an episode
    mentioning the same fingerprint AND have a fraud_flag_raised episode
    of their own. Returns [] — never raises — when the signal genuinely
    isn't available (no fingerprint, Graphiti/Neo4j not configured, or a
    real query error), so callers can always treat this as "additional
    context, possibly empty" rather than something they need to guard
    against failing.

    Returns a list of {"customer_id", "fraud_episode_id", "flagged_at"}
    dicts — one entry per OTHER customer found, not per matching episode
    (a customer with multiple fraud episodes still appears once, via the
    query's DISTINCT).
    """
    import logging
    logger = logging.getLogger("graphiti_adapter")

    if not payment_fingerprint:
        return []

    if not _is_graphiti_available():
        return []

    from app.core.config import get_settings
    settings = get_settings()
    if not settings.neo4j_uri:
        logger.info(
            "find_related_fraud_signals skipped: Neo4j Aura not configured (NEO4J_URI unset). "
            "This cross-customer fraud query requires Neo4j - the embedded Kuzu fallback "
            "(this deployment's current active backend) is not wired for it. Set NEO4J_URI "
            "to enable — see .env.example's Neo4j Aura section."
        )
        return []

    try:
        return _run_async(_find_related_fraud_signals_async(payment_fingerprint, exclude_customer_id))
    except Exception as e:
        # Same non-fatal discipline as log_episode()'s callers throughout
        # this project (resolution_completion.py, orchestrator.py): a
        # genuinely down/misconfigured Neo4j instance must never break
        # fraud scoring — it should just mean this ONE extra signal is
        # unavailable for this call, same as if no fingerprint existed.
        logger.warning("find_related_fraud_signals failed (non-fatal, returning no signal): %s", e)
        return []


async def _find_related_fraud_signals_async(payment_fingerprint: str, exclude_customer_id: str) -> list[dict]:
    from app.core.config import get_settings
    settings = get_settings()
    from graphiti_core.driver.neo4j_driver import Neo4jDriver

    # Fresh driver per call, same reasoning as _build_graphiti_client()'s
    # docstring: a cached async driver's connection pool binds to
    # whichever event loop was active at construction, and _run_async()
    # tears down its event loop after every call.
    driver = Neo4jDriver(
        uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    try:
        # Exact-match substring markers against the deterministic JSON
        # this project's own code writes (json.dumps with default
        # separators: '"key": "value"') - sufficient for an exact-match
        # fraud signal and avoids an APOC/JSON-parsing dependency that
        # may not be enabled on every Neo4j Aura free-tier instance.
        result = await driver.execute_query(
            _RELATED_FRAUD_SIGNALS_QUERY,
            params={
                "exclude_customer_id": exclude_customer_id,
                "fingerprint_marker": f'"payment_fingerprint": "{payment_fingerprint}"',
                "fraud_marker": '"episode_type": "fraud_flag_raised"',
            },
        )
        return [
            {
                "customer_id": record["other_customer_id"],
                "fraud_episode_id": record["fraud_episode_uuid"],
                # REAL BUG, found only against a live Neo4j Aura instance
                # (this sandbox's mocked tests never caught it, since
                # they hand back plain strings): the neo4j Python
                # driver returns its OWN temporal type
                # (neo4j.time.DateTime) for a Cypher-returned
                # timestamp property, NOT a stdlib datetime.datetime -
                # and it is not JSON-serializable. This value flows
                # into fraud_result, then into AuditLogEntry.detail (a
                # JSON column), so leaving it as-is crashed the whole
                # aggregate_node commit with "Object of type DateTime
                # is not JSON serializable" - AFTER the fraud check had
                # already correctly found the match (confirmed against
                # a real deployment: risk_score=0.95, flag=True). Fixed
                # by explicitly converting to an ISO string here, at
                # the query boundary, so every caller always gets a
                # plain, JSON-safe string regardless of what the
                # backend driver happens to return.
                "flagged_at": _to_iso_string(record["flagged_at"]),
            }
            for record in result.records
        ]
    finally:
        await driver.close()
