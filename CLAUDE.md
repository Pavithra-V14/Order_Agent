# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An autonomous order-exception resolution agent (FastAPI + LangGraph). A case (payment / inventory / carrier / return / fraud exception on an order) is detected, diagnosed by parallel agents, decided by a bounded rule-based policy workflow behind guardrails, then auto-executed, escalated to a human, or blocked. The spec is `docs/order-exception-agent-architecture.md`; most design decisions trace to a section of it. `README.md` contains a long history of bugs found and why things are the way they are — check it before "simplifying" something that looks odd.

## Commands

Environment is managed with `uv` (Windows dev box; venv at `.venv`). Python >= 3.11.

```bash
uv run uvicorn app.main:app --reload            # API + Jinja2 frontend; in-process job worker starts automatically
uv run python scripts/run_rq_worker.py          # only needed as a separate process when REDIS_URL is set
uv run python scripts/run_ingestion.py          # ingest data/policies/*.pdf into Qdrant
uv run python -m app.tools.mcp_server           # standalone stdio MCP tool server
uv run python scripts/run_all_demos.py          # end-to-end demo scripts
uv run python scripts/create_api_key.py "Name" admin   # roles: admin, cs_agent, service, readonly

uv run python -m pytest tests/ -q                                   # full suite (run before calling anything done)
uv run python -m pytest tests/test_phase7_resolution.py -v          # one file
uv run python -m pytest tests/test_phase7_resolution.py::test_name  # one test
uv run python -m pytest tests/ --cov=app --cov-report=term-missing  # README cites an 85% coverage floor
uv run python -m pytest tests/test_phase15_golden_set.py -v         # pre-deployment golden-set gate
uv run python scripts/run_live_llm_eval.py --runs 3                 # evaluate the REAL configured LLM (uses Groq tokens)
uv run python -m app.workers.reconciler                             # one reconciler pass (or POST /admin/reconcile)
```

The golden set runs `FakeLLMClient` only. `scripts/run_live_llm_eval.py` (`app/eval/live_llm_eval.py`) is the gate for the real model: run it after any prompt/model change (bump `PROMPT_VERSIONS` first); it writes `data/eval/*.json` and exits 1 below `--min-pass-rate`.

There is no configured linter/formatter. `asyncio_mode = "auto"` is set in `pyproject.toml`. `requirements.txt` is the dependency source of truth (pyproject holds only metadata/tool config).

## Architecture

**Case pipeline** (`app/agents/orchestrator.py::run_full_case_pipeline`):
1. LangGraph graph: `start` fans out in parallel to `diagnosis` (open-ended LLM planning loop with loop-termination ceilings, `diagnosis_agent.py`), `fraud`, `inventory`, `customer_context` (`workflow_agents.py`), then joins at `aggregate` (persists state, writes episodic memory).
2. Traced RAG retrieval (`app/rag/traced_retrieval.py`) of the applicable policy **as of the order's purchase date** — temporal correctness (old vs. new policy version) is a core requirement.
3. `resolution_policy_workflow.py`: deterministic, rule-based decision (not an open LLM call) → Tier 1 hard ceilings (`guardrails/tier1_ceilings.py`) + Tier 2 structural/PII (`guardrails/tier2_structural.py`) → `RoutingOutcome` AUTO_EXECUTE / ESCALATE / BLOCKED. Tier 3 is an async sampled LLM judge run via the job queue.
4. `resolution_completion.py` → `execution_agent.py` (idempotent refund/reship via tools, circuit breakers) → `verification_agent.py` (provider re-check before RESOLVED) → `comms_workflow.py`.

Case lifecycle is the `CaseState` enum in `app/core/db.py` (DETECTED → DIAGNOSING → ESCALATED / BLOCKED / EXECUTING → PENDING_RETRY / VERIFYING → RESOLVED, plus REOPENED); every transition is audit-logged. Human escalation decisions come in via `app/api/v1/escalations.py`, which runs Tier 1/2 plus per-role approval limits on the human's final decision and claims the case atomically (ESCALATED → EXECUTING) so two reviewers can't both execute. `app/workers/reconciler.py` (`POST /admin/reconcile` from cron, or the in-process `ReconcilerScheduler` when `RECONCILER_INTERVAL_SECONDS` > 0 — enable on one instance only) owns every case that stops moving: retries PENDING_RETRY, re-verifies VERIFYING, re-runs or escalates cases stuck mid-pipeline.

**Root-cause contract**: the LLM's free-text root causes are parsed into fixed categories by `app/agents/root_causes.py` (never substring-matched), and a category only counts when tool findings back it (e.g. `carrier_issue` needs a real carrier fault status). Anything inconclusive — empty, `diagnosis_incomplete`/`diagnosis_timeout`, unrecognized, contradicted by the gateway — yields a `requires_human_review` no-money proposal. **No rule may fall through to a refund.** Denials are never auto-executed unless `AUTO_EXECUTE_DENIALS=true`.

**Real-vs-fake backends via settings-driven factories.** Every external dependency has a local substitute behind the same interface, auto-selected by whether a credential is configured in `app/core/config.py` settings:
- `get_llm_client()` (`agents/llm_client.py`): `LiteLLMClient` (Groq → Gemini → Cohere fallback chain) if any key set, else `FakeLLMClient` (rule-based, deterministic).
- `get_embedder()` (Mistral vs TF-IDF), `get_qdrant_client()` (Qdrant Cloud vs embedded local), `get_cache()` (Redis vs in-process TTL), `get_payment_gateway()` (Stripe vs fake), `get_carrier_gateway()` (EasyPost/Shippo vs fake), episodic memory (Graphiti on Neo4j Aura / Kuzu vs SQL).
- `app/workers/job_queue.py`: RQ on Redis vs in-process worker threads. Webhooks ack fast and enqueue; handlers are registered in `app/main.py::_register_job_handlers`. Jobs run in two lanes (`lane_for`): **slow** (case pipelines: OMS/carrier webhooks, Tier 3 judge; `JOB_QUEUE_SLOW_WORKERS`, forced to 1 on SQLite) and **fast** (inventory updates, episodic writes) so housekeeping never waits behind a 30s LLM pipeline. RQ uses `order_exception_agent_fast` + `order_exception_agent`. Episodic memory writes are async (eventually consistent); reads are sync.

**Money-movement invariants:** refund/reship idempotency keys are scoped to order + payment + amount, not the case, so duplicate cases can't double-pay; webhooks dedupe on `event_id` (`webhook_events` table) and on any case for the same order+exception type within `WEBHOOK_DEDUPE_WINDOW_HOURS`. Execution errors that describe the request (Stripe `InvalidRequestError`, "already refunded", missing payment) are `FAILED` → human, are not retried, and do not count against the shared circuit breaker (`CircuitBreaker.call(fn, is_failure=...)`); only transient errors go to PENDING_RETRY.

**Diagnosis evidence is prefetched** (`DIAGNOSIS_PREFETCH_EVIDENCE`, `_prefetch` in `diagnosis_agent.py`): order, payment, inventory (with `requested_qty`/`sufficient`) and carrier data are fetched before the model's first step, so the model interprets complete evidence rather than choosing what to look at (the live eval showed it skipping inventory on paid returns). The planner receives `exception_type` and a redacted view (`_llm_view`) with no customer id, payment id, card fingerprint or payment breakdown; the rules and audit trail still get full findings.

**LLM output is untrusted input**: diagnosis actions are validated against an allow-list and planner errors end the loop as `diagnosis_incomplete`; fraud scores are validated and the flag threshold (`FRAUD_FLAG_THRESHOLD`) is applied in code — on LLM failure the fraud agent falls back to the deterministic rules and marks the result `degraded`, which forces human review. Every real LLM call records an `llm.<purpose>` trace span (model, `PROMPT_VERSIONS` entry, tokens, latency) when inside a case (`set_llm_trace` in the orchestrator nodes); `GET /metrics/llm` aggregates them. Bump `PROMPT_VERSIONS` whenever prompt text changes.

**Safety invariants to preserve:**
- Learning loop (`agents/learning_loop.py`): the batch job only writes `ThresholdProposalRecord`; only `accept_threshold_proposal()` writes overrides and it requires a named human (rejects "system"/"auto"/empty). Drift watch never auto-rebaselines.
- `auto_execution_enabled` in config is the rollback switch — it only changes auto-execute→escalate routing; Tier 1 ceilings stay active regardless.
- Tracing (`core/tracing.py`) redacts PII **before** persisting.

**Database:** SQLite by default (`data/case_state.db`), Postgres via `DATABASE_URL`. No Alembic — `init_db()` does `create_all()` plus `_ensure_new_columns()` to add missing columns to existing tables (register new columns there) and `_ensure_enum_values()` to add new `CaseState` members to Postgres's native enum type. Auth uses a **separate** Postgres DB (`app/core/auth_db.py`, `AUTH_DATABASE_URL`); set `AUTH_ENABLED=false` for local dev without it. Auth supports `X-API-Key` and JWT bearer (`app/core/auth.py`).

**Frontend:** server-rendered Jinja2 (`app/templates/`, routes in `app/pages.py`) + vanilla JS/CSS in `app/static/`, no build step.

## Testing conventions

- `tests/conftest.py` strips cloud credentials from both `.env` and the process environment for the whole session, so tests always get fake/local backends. It also disables `python-dotenv`'s `load_dotenv`, because `graphiti_core` calls it at import time and would otherwise copy real `.env` keys back mid-test (this once sent a golden-set test to the real Stripe API). To simulate a configured credential, monkeypatch settings within the test rather than relying on env.
- RAG tests share the persistent `data/reindex_state.json` + `data/qdrant_local`. `tests/test_phase12_api.py`'s fixture purges `TEST-*` doc ids before and after each test (they used to accumulate and make upload tests see 0 chunks on every run after the first); follow that pattern for any new test that uploads policies.
- `tests/test_review_evidence.py` holds the regression tests for the live-path review (fall-through refunds, duplicate webhooks, human-edit limits, breaker cascade, reconciler).
- Tests that touch the DB use a fixture that sets `DATABASE_URL` to a temp SQLite file, calls `get_settings.cache_clear()`, `importlib.reload(app.core.db)` (and `app.main` for API tests), and resets global singletons (`reset_fake_gateway`, `reset_fake_carrier`, `reset_all_caches`, `reset_job_queue`, `reset_all_breakers`). Follow that pattern (see `tests/test_phase12_api.py`).
- Because of those reloads, **never import `SessionLocal` (or other reloadable module globals) by name at module level** in app code — import inside the function so it re-resolves at call time.
- Cross-test global-state contamination has bitten this repo repeatedly: a test passing alone but failing in the full suite is the usual signature. Always run the full suite.
- Async job effects (e.g. episodic writes) must be polled for in tests, not asserted immediately.
- Redis/toxiproxy/Postgres-dependent tests skip when those servers aren't available.
- `respx` only intercepts `httpx`; cloud clients are written against raw `httpx` (not vendor SDKs using `requests`) so they stay mockable.

## Contribution norms (from CONTRIBUTING.md)

Reproduce before fixing; fix root causes; add a test that fails on the old code; commit messages explain *why*. Flag changes touching auth, guardrail tiers, or real refund/reship execution explicitly.
