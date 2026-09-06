"""
Central configuration, per architecture doc Part 8.1.

Local-first defaults: every setting has a working fallback so the system
runs on a laptop / this sandbox with zero external services. Swap in real
Postgres/Redis/Qdrant/LLM-provider URLs via .env for staging/production —
see docs/deployment.md (added in a later phase) for the docker-compose path.
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
    # Falls back to a local SQLite file if DATABASE_URL isn't set — swap to
    # Postgres in staging/prod (docker-compose service name shown as example).
    database_url: str = "sqlite:///./data/case_state.db"

    # --- Cache / Queue (Layer 5, 8.9) ---
    # No Redis in this sandbox -> falls back to an in-process cache/queue.
    # In staging/prod, set redis_url, e.g. redis://redis:6379/0
    redis_url: str | None = None

    # --- Vector store (8.2.6) ---
    # Qdrant's embedded "local mode" needs no server — writes to a local
    # on-disk path. Set qdrant_url (e.g. http://qdrant:6333) to switch to a
    # real Qdrant instance in staging/prod.
    qdrant_url: str | None = None
    qdrant_local_path: str = "./data/qdrant_local"
    qdrant_collection: str = "policy_docs"

    # --- LLM providers (8.10 free-tier matrix) ---
    # All optional at this phase — Phase 6+ will fail fast with a clear
    # error if an agent tries to call a provider with no key configured.
    groq_api_key: str | None = None
    mistral_api_key: str | None = None
    google_api_key: str | None = None
    cohere_api_key: str | None = None

    router_model: str = "groq:llama-3.3-70b-versatile"
    reasoning_model: str = "mistral:mistral-large-latest"
    generation_model: str = "google:gemini-2.0-flash"

    # --- Embeddings / Reranker (8.2, 8.10) ---
    # Self-hosted BGE-M3 / BGE-Reranker per the free-tier recommendation.
    # embedding_backend: "local_bge" (default, runs in-process) | "mistral" | "google"
    embedding_backend: str = "local_bge"
    reranker_backend: str = "local_bge"

    # --- Observability (8.8) ---
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    tracing_enabled: bool = False  # off until Phase 10, avoids noisy failures before then

    # --- Guardrail thresholds (Part 5, Phase 1's Architecture Decision Sheet) ---
    auto_execute_confidence_threshold: float = 0.90
    auto_execute_value_ceiling_usd: float = 50.00

    # --- Rollback switch (Phase 16) ---
    # Global kill-switch for auto-execution — per Part 2's scorecard,
    # disabling this must route ALL new resolution decisions to human
    # escalation regardless of confidence/value/fraud-flag, in well under
    # 5 minutes (in practice: as fast as the config can be reloaded,
    # since this is a single boolean check in the routing path, not a
    # redeploy). See tests/test_phase16_rollout.py for the timed proof.
    auto_execution_enabled: bool = True

    @property
    def is_local(self) -> bool:
        return self.environment == "local"


@lru_cache
def get_settings() -> Settings:
    return Settings()
