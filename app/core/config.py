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

    # Defaults to True (real auth enforced) for every real deployment.
    # Set to False ONLY for local dev convenience or automated tests —
    # tests/conftest.py sets this via os.environ for the whole suite,
    # deliberately at the settings level rather than patching a specific
    # FastAPI app instance's dependency_overrides, since several test
    # fixtures throughout this project call importlib.reload(app.main)
    # for isolation, which creates a BRAND NEW app object with an empty
    # dependency_overrides dict each time — silently discarding any
    # override applied to the previous instance. A settings flag checked
    # freshly inside get_current_api_key() itself survives reloading,
    # since it's not tied to any specific app object at all.
    auth_enabled: bool = True

    # Dedicated auth storage - deliberately a SEPARATE database/connection
    # from the main app's DATABASE_URL (which can stay SQLite for local
    # dev of everything else). Real enterprise systems commonly run
    # identity/auth as its own subsystem with its own datastore, both
    # for security isolation (a compromise of case/order data doesn't
    # automatically expose credentials, and vice versa) and so auth can
    # be scaled/backed-up/audited independently of transactional data.
    auth_database_url: str = "postgresql://postgres:postgres_dev_password@localhost:5432/order_exception_agent"
    jwt_secret_key: str = "dev-only-insecure-secret-change-in-production-via-JWT_SECRET_KEY-env-var"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60

    # Real, centralized business defaults — previously scattered as
    # hardcoded 0.90/50.0 magic numbers in every demo script
    # individually, with no single source of truth. A real deployment
    # tunes these per Part 5's guardrail table (CS Ops owner's call,
    # not an engineering default) — the ThresholdOverrideRecord table
    # already supports PER-CLUSTER overrides on top of these; these
    # two settings are only the fallback for a cluster with no
    # override yet.
    default_auto_execute_confidence_threshold: float = 0.90
    default_auto_execute_value_ceiling_usd: float = 50.0

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
    # Multi-provider LLM fallback: found declared (google_api_key,
    # cohere_api_key above) but NEVER ACTUALLY WIRED into any LLM
    # client anywhere in this codebase - LiteLLMClient's Router had
    # exactly one deployment (Groq), so "Available Model Group
    # Fallbacks=None" on a real Groq rate-limit error was litellm
    # telling the truth, not a misconfiguration. Fixed by building the
    # Router's model_list and fallback chain DYNAMICALLY from whichever
    # of groq_api_key/google_api_key/cohere_api_key are actually
    # configured (same settings-driven pattern as every other backend
    # in this project), rather than assuming any one of them is always
    # present. cohere_generation_model is a plain "command-r-plus" -
    # verified only that this model name exists in Cohere's product
    # line as of this project's training data, NOT confirmed against a
    # live Cohere API call from this sandbox (no network route here) -
    # check Cohere's current model list if this specific name is ever
    # deprecated, same honest caveat as graphiti_router_model above.
    cohere_generation_model: str = "command-r-plus"

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

    # Absolute Tier 1 ceiling for any single action, automated OR human.
    max_single_action_ceiling_usd: float = 1000.00
    # Denials send the customer a "we can't approve this" message; by
    # default a person confirms every one rather than the rules alone.
    auto_execute_denials: bool = False
    # Largest amount a reviewer of each role may approve or edit to, on
    # top of the Tier 1 ceiling above. Roles not listed may not approve.
    human_approval_limit_cs_agent_usd: float = 250.00
    human_approval_limit_admin_usd: float = 1000.00
    # A webhook for an order+exception type that already had a case
    # opened within this many hours is treated as a redelivery.
    webhook_dedupe_window_hours: float = 24.0
    # Fraud score at or above this is a flag, whatever the model says.
    fraud_flag_threshold: float = 0.60

    # --- Rollback switch (Phase 16) ---
    auto_execution_enabled: bool = True

    @property
    def is_local(self) -> bool:
        return self.environment == "local"


@lru_cache
def get_settings() -> Settings:
    return Settings()
