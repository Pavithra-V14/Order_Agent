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


def _is_graphiti_available() -> bool:
    from app.core.config import get_settings
    settings = get_settings()
    return bool(settings.groq_api_key)


class _ProjectEmbedderAdapter:
    """Adapts this project's BaseEmbedder into Graphiti's abstract
    EmbedderClient interface - Graphiti's embeddings come from the same
    MistralEmbedder/TfidfEmbedder every other part of this project uses."""

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


_graphiti_client = None


def _get_graphiti_client():
    """Constructs (once) and returns the process-wide Graphiti client."""
    global _graphiti_client
    if _graphiti_client is not None:
        return _graphiti_client

    from app.core.config import get_settings
    settings = get_settings()
    if not settings.groq_api_key:
        raise RuntimeError("GROQ_API_KEY not configured - Graphiti requires an LLM for entity extraction.")

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

    _graphiti_client = Graphiti(llm_client=llm_client, embedder=embedder, graph_driver=driver)
    return _graphiti_client


def reset_graphiti_client() -> None:
    """Test helper - forces reconstruction on next use."""
    global _graphiti_client
    _graphiti_client = None


def _run_async(coro):
    """Bridges Graphiti's async API into this project's entirely-sync
    codebase. Safe here since nothing in this project runs its own event
    loop that this would conflict with."""
    return asyncio.run(coro)


async def _add_episode_async(customer_id: str, episode_type: str, content: dict,
                              occurred_at: datetime, case_id: str = None) -> None:
    client = _get_graphiti_client()
    await client.build_indices_and_constraints()
    episode_body = json.dumps({"episode_type": episode_type, "content": content, "case_id": case_id})
    await client.add_episode(
        name=episode_type,
        episode_body=episode_body,
        source_description=f"case:{case_id}" if case_id else "system",
        reference_time=occurred_at,
        group_id=customer_id,
    )


async def _get_customer_history_async(customer_id: str, episode_type: str = None, limit: int = 20) -> list:
    client = _get_graphiti_client()
    nodes = await client.retrieve_episodes(
        reference_time=datetime.now(timezone.utc), last_n=limit, group_ids=[customer_id],
    )
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
