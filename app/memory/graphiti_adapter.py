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


def _build_graphiti_client():
    """Constructs a FRESH Graphiti client — deliberately NOT cached as a
    persistent global anymore. Found to be the actual root cause of the
    persistent "Unable to retrieve routing information" failures: the
    Neo4j async driver's internal connection pool/routing-table state
    gets bound to whichever asyncio event loop is active when it's first
    used. _run_async() calls asyncio.run() fresh each time, which creates
    AND TEARS DOWN a new event loop per call — so a cached client's
    driver, first used under event-loop-instance #1, becomes silently
    broken the moment a SECOND _run_async() call (a different LangGraph
    node — diagnosis, then fraud, then customer_context — each calling
    into Graphiti separately) runs under event-loop-instance #2. This
    is a well-documented asyncio anti-pattern (reusing a loop-bound async
    resource across different event loops) — and explains why switching
    event loop TYPE (Selector vs Proactor, an earlier attempted fix)
    made no difference at all: the problem was never about which kind of
    loop, it was about reusing one driver across many separate loops.

    The fix: construct the whole client fresh inside the SAME event loop
    that will use it (called from within _add_episode_async /
    _get_customer_history_async, each of which owns one complete
    asyncio.run() lifecycle end to end), and close it before returning.
    This costs a fresh Neo4j connection per call instead of reusing one —
    a real, accepted tradeoff for correctness, and Aura connections are
    fast enough that this isn't a meaningful performance concern at this
    project's actual call volume.
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
            model=settings.router_model,
            base_url="https://api.groq.com/openai/v1",
        ),
        structured_output_mode="json_object",
    )
    embedder = _ProjectEmbedderAdapter()

    if settings.neo4j_uri:
        from graphiti_core.driver.neo4j_driver import Neo4jDriver
        driver = Neo4jDriver(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password)
    else:
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        # NOTE: do NOT pre-create kuzu_local_path as a directory — Kuzu's
        # Database() creates its own storage AT that exact path (a file,
        # not a directory) and raises "Database path cannot be a
        # directory" if something already exists there. Confirmed
        # directly: an earlier version of this code called os.makedirs()
        # first and broke on exactly this.
        driver = KuzuDriver(db=settings.kuzu_local_path)

    return Graphiti(
        llm_client=llm_client, embedder=embedder, graph_driver=driver, cross_encoder=_BM25CrossEncoder(),
    )


def _get_graphiti_client():
    """Backward-compatible alias — see _build_graphiti_client's docstring
    for why this no longer caches a persistent client."""
    return _build_graphiti_client()


def reset_graphiti_client() -> None:
    """No-op now that there's no persistent global to reset — kept so
    existing test fixtures calling this don't break."""
    pass


GRAPHITI_CALL_TIMEOUT_SECONDS = 15.0


def _run_async(coro):
    """Bridges Graphiti's async API into this project's entirely-sync
    codebase. Safe here since nothing in this project runs its own event
    loop that this would conflict with.

    Wrapped in a bounded timeout — nothing in this bridge should be
    allowed to hang unbounded regardless of which external dependency
    (Neo4j or Groq) stalls; an earlier version of this function let a
    Neo4j connectivity problem retry for MINUTES with escalating delays
    before finally raising.
    """
    try:
        return asyncio.run(asyncio.wait_for(coro, timeout=GRAPHITI_CALL_TIMEOUT_SECONDS))
    except asyncio.TimeoutError:
        raise RuntimeError(
            f"Graphiti call did not complete within {GRAPHITI_CALL_TIMEOUT_SECONDS}s — this almost "
            f"always means either Neo4j (NEO4J_URI) or Groq (GROQ_API_KEY) is unreachable or "
            f"misconfigured, not a normal slow response. Check both independently before retrying — "
            f"see the troubleshooting scripts mentioned in this project's README."
        ) from None


async def _add_episode_async(customer_id: str, episode_type: str, content: dict,
                              occurred_at: datetime, case_id: str = None) -> None:
    client = _build_graphiti_client()
    try:
        await client.build_indices_and_constraints()
        episode_body = json.dumps({"episode_type": episode_type, "content": content, "case_id": case_id})
        await client.add_episode(
            name=episode_type,
            episode_body=episode_body,
            source_description=f"case:{case_id}" if case_id else "system",
            reference_time=occurred_at,
            group_id=customer_id,
        )
    finally:
        await client.close()


async def _get_customer_history_async(customer_id: str, episode_type: str = None, limit: int = 20) -> list:
    client = _build_graphiti_client()
    try:
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


def log_episode_graphiti(customer_id: str, episode_type: str, content: dict,
                          occurred_at: datetime, case_id: str = None) -> dict:
    _run_async(_add_episode_async(customer_id, episode_type, content, occurred_at, case_id))
    return {"customer_id": customer_id, "episode_type": episode_type, "occurred_at": occurred_at.isoformat()}


def get_customer_history_graphiti(customer_id: str, episode_type: str = None, limit: int = 20) -> list:
    return _run_async(_get_customer_history_async(customer_id, episode_type, limit))
