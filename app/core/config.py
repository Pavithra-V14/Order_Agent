"""
Central configuration, per architecture doc Part 8.1.

Cloud-first, no-Docker deployment: every external dependency has a real
cloud-service implementation (not a stub) that activates automatically
once its credentials are set — Neon/Supabase Postgres, Upstash Redis,
Qdrant Cloud, Groq, Mistral, Langfuse Cloud, EasyPost, Stripe. Leave any
of them unset and the corresponding local-first fallback (SQLite,
in-process cache/queue, embedded Qdrant, fake LLM/carrier) is used
instead — nothing crashes on a partial cloud configuration, and nothing
requires Docker at any point in this stack.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "order-exception-agent"
    environment: str = "local"  # local | staging | production
    log_level: str = "INFO"

    # --- Database (Layer 5/13: case state + audit log) ---
    # Falls back to a local SQLite file if DATABASE_URL isn't set. For a
    # no-Docker cloud Postgres: Neon (neon.tech) or Supabase both give a
    # free-tier Postgres instance in ~1 minute — copy their connection
    # string directly into DATABASE_URL, e.g.:
    #   postgresql://user:pass@ep-xxx.neon.tech/dbname?sslmode=require
    # (uses the `psycopg` v3 driver, installed — SQLAlchemy picks it up
    # automatically for any postgresql:// URL, no separate driver prefix needed)
    database_url: str = "sqlite:///./data/case_state.db"

    # --- Cache / Queue (Layer 5, 8.9) ---
    # Upstash (upstash.com) — no Docker needed, a free-tier Redis database
    # in ~1 minute. Copy its "Redis Connect" URL (rediss://...) directly.
    # Powers BOTH the cache layer (app/cache/ttl_cache.py) AND the async
    # job queue (app/workers/job_queue.py, RQ-backed when this is set).
    redis_url: str | None = None

    # --- Vector store (8.2.6) ---
    # Qdrant Cloud (cloud.qdrant.io) — free-tier cluster, no Docker. Set
    # BOTH qdrant_url (your cluster's URL) and qdrant_api_key (from the
    # cluster's dashboard) — Qdrant Cloud requires the API key on every
    # request, unlike a self-hosted/local instance.
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_local_path: str = "./data/qdrant_local"  # embedded-mode fallback if qdrant_url is unset
    qdrant_collection: str = "policy_docs"

    # --- LLM providers (8.10 free-tier matrix) — all cloud APIs, no Docker ---
    groq_api_key: str | None = None
    mistral_api_key: str | None = None
    google_api_key: str | None = None
    cohere_api_key: str | None = None

    router_model: str = "llama-3.3-70b-versatile"       # Groq-hosted
    # Separate from router_model in principle (so a future Graphiti-
    # specific need doesn't have to follow the main pipeline's model),
    # but currently the SAME value: moonshotai/kimi-k2-instruct-0905 was
    # tried here first and found, via a real 404 from Groq's own API,
    # to have been deprecated outright (confirmed directly against
    # Groq's own deprecation announcements: retired March 23, 2026, in
    # favor of openai/gpt-oss-120b). Groq has similarly deprecated most
    # other alternatives (Kimi K2, Llama 4 Maverick, Llama Guard 4,
    # Qwen3-32B, Llama 4 Scout) — ALL in favor of gpt-oss-120b, leaving
    # it as the realistic, actively-maintained choice rather than one
    # option among several. If gpt-oss-120b's structured-output
    # reliability (a community-reported concern from October 2025) is
    # still an issue, log_episode_failure alerts (see app/core/alerting.py)
    # will surface it directly — falling back to structured_output_mode=
    # "json_object" in graphiti_adapter.py is the documented recovery
    # path if so, not a silent guess.
    graphiti_router_model: str = "openai/gpt-oss-120b"
    reasoning_model: str = "mistral-large-latest"         # Mistral La Plateforme
    generation_model: str = "gemini-2.0-flash"             # Google AI Studio

    # --- Embeddings / Reranker (8.2, 8.10) ---
    # embedding_backend: "mistral" (cloud, default once mistral_api_key is
    # set) | "cohere" (cloud) | "tfidf" (local fallback, no key needed)
    embedding_backend: str = "tfidf"
    reranker_backend: str = "lexical"  # "lexical" (local fallback) | "cohere" (cloud rerank API)

    # --- Observability (8.8) ---
    # Langfuse Cloud (cloud.langfuse.com) — free tier, no Docker/self-host
    # needed. Get public+secret keys from a Langfuse Cloud project.
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    tracing_enabled: bool = False

    # --- Carrier (EasyPost — cloud API, no Docker) ---
    easypost_api_key: str | None = None
    # --- Carrier alternative (Shippo — cloud API, no Docker) ---
    # If BOTH easypost_api_key and shippo_api_key are set, EasyPost takes
    # priority (see get_carrier_gateway()) — pick one, not both, in practice.
    shippo_api_key: str | None = None

    # --- Payment (Stripe — cloud API, no Docker) ---
    stripe_api_key: str | None = None

    # --- Long-term memory graph (Graphiti — 8.4) ---
    # Neo4j Aura (neo4j.com/cloud/aura) — free tier, no Docker. Set all
    # three to activate Graphiti's real graph-backed episodic memory
    # (time-aware relationship reasoning) instead of the SQL-backed
    # substitute (app/memory/episodic.py). Leave unset and Graphiti still
    # activates with an EMBEDDED graph (Kuzu — no server, no Docker, no
    # credentials at all) as long as groq_api_key is set, since Graphiti
    # needs a real LLM for entity/relationship extraction regardless of
    # which graph store backs it — without an LLM key, the SQL substitute
    # is used instead (there's no meaningful "Graphiti without an LLM").
    neo4j_uri: str | None = None
    neo4j_user: str = "neo4j"
    neo4j_password: str | None = None
    # Graphiti's Neo4jDriver defaults this to "neo4j" internally if not
    # passed explicitly — but not every Aura instance actually uses that
    # literal database name (confirmed directly: a real Aura Free
    # instance returned "Neo.ClientError.Database.DatabaseNotFound...
    # database 'neo4j' does not exist"). Check your Aura console's
    # instance connection details for the actual database name if the
    # default doesn't work, and override here.
    neo4j_database: str = "neo4j"
    kuzu_local_path: str = "./data/kuzu_local"

    # --- Guardrail thresholds (Part 5, Phase 1's Architecture Decision Sheet) ---
    auto_execute_confidence_threshold: float = 0.90
    auto_execute_value_ceiling_usd: float = 50.00

    # --- Rollback switch (Phase 16) ---
    auto_execution_enabled: bool = True

    @property
    def is_local(self) -> bool:
        return self.environment == "local"


@lru_cache
def get_settings() -> Settings:
    return Settings()
