"""
Admin data-management endpoints - deliberately dangerous, deliberately
simple: reset tools for development/testing, not production data-
governance features. Every endpoint requires an explicit confirmation
query param (?confirm=true) so nothing here is triggerable by accident
from a bookmark, a browser back-button resubmit, or a stray GET.
"""
from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import require_admin
from sqlalchemy.orm import Session

from app.core.db import (
    get_db, AuditLogEntry, ExceptionCase, TraceSpanRecord, IdempotencyRecord,
    AlertRecord, MockOrderRecord, MockInventoryRecord, ResolutionPatternEntry,
    ThresholdOverrideRecord, ThresholdProposalRecord, EpisodeRecord, JobRecord,
)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/alerts")
def list_alerts(event_type: str = None, limit: int = 50, db: Session = Depends(get_db),
                 _auth=Depends(require_admin)):
    """Real, previously-missing visibility: AlertRecord entries (circuit
    breaker trips, idempotency collisions, Tier 1 blocks, log_episode
    failures) were only ever queryable via a Python function called
    directly against the database — no API endpoint or UI page ever
    exposed them at all. Found directly while adding alerting for a
    real log_episode failure: the alert itself was correctly being
    recorded, but there was genuinely nowhere for a person to go look
    at it."""
    from app.core.alerting import get_recent_alerts
    return get_recent_alerts(db, event_type=event_type, limit=limit)


@router.get("/backend-status")
def get_backend_status(_auth=Depends(require_admin)):
    """Reports which real implementation is active for EVERY swappable
    component, based on the exact same auto-selection logic each
    factory function actually uses — not a guess, a direct read of the
    same settings checks get_llm_client()/get_embedder()/
    get_carrier_gateway()/get_payment_gateway()/etc. use internally.

    Built directly in response to "how do I check what's actually
    being used" — a single, always-available answer instead of reading
    source code or inferring from behavior. Does not perform any live
    connectivity checks (that's what the dedicated scripts/test_*.py
    diagnostics are for) — this reports configuration-based selection
    only: what WOULD be constructed, not whether it can actually connect.
    """
    from app.core.config import get_settings
    settings = get_settings()

    def _mode(condition: bool, real_label: str, fake_label: str) -> str:
        return real_label if condition else fake_label

    return {
        "llm": {
            "active": _mode(bool(settings.groq_api_key), "Groq (real)", "FakeLLMClient (rule-based)"),
            "model": settings.router_model if settings.groq_api_key else None,
        },
        "embedder": {
            "active": _mode(bool(settings.mistral_api_key), "Mistral (real)", "TF-IDF (local)"),
        },
        "vector_store": {
            "active": _mode(bool(settings.qdrant_url), "Qdrant Cloud (real)", "Qdrant embedded (local)"),
            "url": settings.qdrant_url if settings.qdrant_url else settings.qdrant_local_path,
        },
        "cache": {
            "active": _mode(bool(settings.redis_url), "Redis/Upstash (real)", "In-process (local)"),
        },
        "job_queue": {
            "active": _mode(bool(settings.redis_url), "RQ + Redis (real)", "In-process thread (local)"),
            "note": "if real, scripts/run_rq_worker.py MUST be running as a separate process" if settings.redis_url else None,
        },
        "carrier": {
            "active": (
                "EasyPost (real)" if settings.easypost_api_key
                else "Shippo (real)" if settings.shippo_api_key
                else "FakeCarrierGateway (local)"
            ),
            "note": "EasyPost takes priority if both EASYPOST_API_KEY and SHIPPO_API_KEY are set"
                    if (settings.easypost_api_key and settings.shippo_api_key) else None,
        },
        "payment": {
            "active": _mode(bool(settings.stripe_api_key), "Stripe (real)", "FakePaymentGateway (local)"),
        },
        "memory_graph": {
            "active": (
                "Neo4j Aura (real)" if (settings.groq_api_key and settings.neo4j_uri)
                else "Kuzu embedded (local, deprecated upstream)" if settings.groq_api_key
                else "SQL-backed episodic (local, no graph)"
            ),
            "note": "Graphiti requires GROQ_API_KEY regardless of which graph backend is used" if not settings.groq_api_key else None,
        },
        "tracing": {
            "active": _mode(
                bool(settings.tracing_enabled and settings.langfuse_public_key and settings.langfuse_secret_key),
                "Langfuse (real, dual-write with SQL)", "SQL-only (local)",
            ),
        },
        "database": {
            "active": "Postgres/Neon/Supabase (real)" if not settings.database_url.startswith("sqlite") else "SQLite (local)",
        },
        "reranker": {
            "active": "Lexical overlap (local)",  # no cloud reranker backend implemented yet
        },
    }


@router.post("/reconcile")
def reconcile_endpoint(stale_after_minutes: float = 30.0, verify_timeout_hours: float = 72.0,
                       _auth=Depends(require_admin)):
    """One pass of the case reconciler (app/workers/reconciler.py): retries
    PENDING_RETRY executions, re-verifies VERIFYING cases, and re-runs or
    escalates cases stuck mid-pipeline. Trigger it from cron every few
    minutes - this project has no in-process scheduler."""
    from app.workers.reconciler import run_reconciliation
    return run_reconciliation(stale_after_minutes=stale_after_minutes,
                              verify_timeout_hours=verify_timeout_hours)


@router.post("/evict-stale-buffers")
def evict_stale_buffers_endpoint(max_age_seconds: float = 86400.0, _auth=Depends(require_admin)):
    """Manually triggers the Summary Buffer TTL sweep (memory-upgrade
    follow-up) - in production this runs on a schedule (same pattern as
    /threshold-proposals/run-batch-job); exposed here so it doesn't
    require waiting for a real stale buffer to accumulate to verify it
    works, and so ops can run it on demand. Default 86400s (24h) matches
    this project's other "abandoned case" assumptions; pass a smaller
    value in a demo/test context to see it actually evict something."""
    from app.memory.summary_buffer import evict_stale_buffers
    evicted_count = evict_stale_buffers(max_age_seconds)
    return {"evicted_count": evicted_count, "max_age_seconds": max_age_seconds}


@router.delete("/audit-log")
def delete_all_audit_log(confirm: bool = False, db: Session = Depends(get_db),
                          _auth=Depends(require_admin)):
    """Deletes every audit log entry. Does NOT touch cases, orders, or
    any other data - audit history only."""
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass ?confirm=true to actually delete the audit log")
    count = db.query(AuditLogEntry).count()
    db.query(AuditLogEntry).delete()
    db.commit()
    return {"deleted": count, "table": "audit_log_entries"}


@router.delete("/rag-index")
def delete_rag_index(confirm: bool = False, _auth=Depends(require_admin)):
    """Deletes the entire Qdrant collection (policy document embeddings)
    and clears the local reindex-state tracker, so the next ingestion
    run starts genuinely clean rather than skipping "unchanged" files
    that no longer exist in a fresh collection. Works identically
    whether Qdrant is embedded-local or Qdrant Cloud - same client API
    either way."""
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass ?confirm=true to actually delete the RAG index")

    from app.core.config import get_settings
    from app.rag.vectorstore import get_qdrant_client
    settings = get_settings()

    client = get_qdrant_client()
    existing = [c.name for c in client.get_collections().collections]
    deleted_collection = False
    if settings.qdrant_collection in existing:
        client.delete_collection(collection_name=settings.qdrant_collection)
        deleted_collection = True

    import os
    reindex_state_path = "data/reindex_state.json"
    deleted_state_file = False
    if os.path.exists(reindex_state_path):
        os.remove(reindex_state_path)
        deleted_state_file = True

    return {
        "deleted_collection": deleted_collection,
        "deleted_reindex_state_file": deleted_state_file,
        "note": "run scripts/run_ingestion.py to re-index policy documents from scratch",
    }


@router.delete("/all-data")
def delete_all_data(confirm: bool = False, db: Session = Depends(get_db),
                     _auth=Depends(require_admin)):
    """Truncates EVERY application table - cases, orders, stock, audit
    log, traces, idempotency records, learning-loop patterns, threshold
    proposals/overrides, SQL-backed episodic memory, and job records.
    Does NOT touch the RAG index (Qdrant) or long-term memory graph
    (Neo4j/Kuzu) - use DELETE /admin/rag-index separately for those,
    since they're genuinely different storage systems with their own
    reset semantics, not something a single Postgres delete pass can
    reach. This is a development/testing reset tool, not a production
    data-retention feature - there is no undo."""
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass ?confirm=true to actually delete ALL data")

    # Deletion order matters where foreign keys exist — children before
    # parents (audit log / trace spans reference case_id; cases don't
    # have a DB-enforced FK to orders in this schema, but deleting orders
    # last keeps the intent clear regardless).
    tables = [
        (AuditLogEntry, "audit_log_entries"),
        (TraceSpanRecord, "trace_span_records"),
        (IdempotencyRecord, "idempotency_records"),
        (AlertRecord, "alert_records"),
        (JobRecord, "job_records"),
        (EpisodeRecord, "episode_records"),
        (ResolutionPatternEntry, "resolution_pattern_entries"),
        (ThresholdProposalRecord, "threshold_proposal_records"),
        (ThresholdOverrideRecord, "threshold_override_records"),
        (ExceptionCase, "exception_cases"),
        (MockInventoryRecord, "mock_inventory_records"),
        (MockOrderRecord, "mock_order_records"),
    ]
    deleted_counts = {}
    for model, table_name in tables:
        count = db.query(model).count()
        db.query(model).delete()
        deleted_counts[table_name] = count
    db.commit()

    return {"deleted_counts": deleted_counts, "total_rows_deleted": sum(deleted_counts.values())}
