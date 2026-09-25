from contextlib import asynccontextmanager

from fastapi import FastAPI

from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.db import init_db
from app.api.v1 import health, cases, metrics, webhooks, escalations, policies, audit, threshold, admin, testing, auth
from app.workers.job_queue import get_job_queue
from app.workers import handlers
from app.pages import router as pages_router

settings = get_settings()


def _register_job_handlers() -> None:
    """Wires job_type -> handler function, per Phase 12's async queue
    design (app/workers/job_queue.py)."""
    q = get_job_queue()
    q.register_handler("process_oms_webhook", handlers.handle_oms_webhook)
    q.register_handler("process_inventory_webhook", handlers.handle_inventory_webhook)
    q.register_handler("process_carrier_webhook", handlers.handle_carrier_webhook)
    q.register_handler("tier3_judge_sample", handlers.handle_tier3_judge_sample)
    q.register_handler("log_episode", handlers.handle_log_episode)
    q.register_handler("log_cross_customer_signal", handlers.handle_log_cross_customer_signal)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # SQLite/dev: create tables directly. Postgres/staging: swap for Alembic
    # migrations before this ships past Phase 0 — noted in docs/deployment.md.
    init_db()
    # Found from a real user report: this ran unconditionally, meaning
    # every single test (or dev run) that merely constructs the app —
    # even with AUTH_ENABLED=false, even for code with nothing to do
    # with auth — required a live Postgres connection just to start up,
    # breaking dozens of otherwise-unrelated tests with a confusing
    # "Connection refused" the moment Postgres wasn't running locally.
    from app.core.config import get_settings
    if get_settings().auth_enabled:
        from app.core.auth_db import init_auth_db
        try:
            init_auth_db()
        except Exception as e:
            # A genuinely down auth database shouldn't crash the WHOLE
            # app on startup either - login/signup will correctly fail
            # with a clear error when actually used, but read-only
            # pages and non-auth endpoints should still come up.
            import logging
            logging.getLogger("order_exception_agent").warning(
                "init_auth_db() failed - human sign-in (login/signup) will not work until this is "
                "resolved, but the rest of the app is starting anyway: %s", e,
            )
    _register_job_handlers()
    get_job_queue().start_worker()

    # Ensures the Qdrant collection AND its required payload indexes
    # exist before the app serves its first request — found necessary
    # directly: ensure_collection() (which creates the payload indexes
    # Qdrant Cloud requires for filtering) was previously only ever
    # called from the ingestion path. A collection created BEFORE that
    # fix existed, or simply never re-ingested in a given session, would
    # still hit "Index required but not found" on every search, since
    # nothing ever retroactively added the missing indexes. Calling this
    # once at startup — not on every search, which would add an
    # unnecessary Qdrant round-trip per query — makes the fix apply
    # unconditionally rather than only the next time someone happens to
    # run ingestion again.
    try:
        from app.rag.vectorstore import get_qdrant_client, ensure_collection
        from app.rag.embeddings import get_embedder
        ensure_collection(get_qdrant_client(), settings.qdrant_collection, get_embedder().dim)
    except Exception as e:
        import logging
        logging.getLogger("startup").warning("ensure_collection at startup failed (non-fatal): %s", e)

    yield
    get_job_queue().stop_worker()


app = FastAPI(
    title=settings.app_name,
    description="Autonomous Omnichannel Order Exception & Fulfillment Resolution Agent",
    version="0.1.0-phase12",
    lifespan=lifespan,
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(cases.router, prefix="/api/v1")
app.include_router(metrics.router, prefix="/api/v1")
app.include_router(webhooks.router, prefix="/api/v1")
app.include_router(escalations.router, prefix="/api/v1")
app.include_router(policies.router, prefix="/api/v1")
app.include_router(audit.router, prefix="/api/v1")
app.include_router(threshold.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(testing.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(pages_router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
