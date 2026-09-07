"""
Database models and session management.

Case state machine per architecture doc Part 3/6:
  detected -> diagnosing -> decided -> executing -> verifying -> resolved
  (with escalated / reopened as side-states, per edge cases 6.2/6.3)

Runs on SQLite locally (zero setup) or Postgres in staging/prod via
DATABASE_URL — same models, same code, per app.core.config.
"""
import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, String, Float, DateTime, Enum, JSON, ForeignKey, create_engine, Text, Integer
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from app.core.config import get_settings

Base = declarative_base()


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CaseState(str, enum.Enum):
    DETECTED = "detected"
    DIAGNOSING = "diagnosing"
    DECIDED = "decided"
    ESCALATED = "escalated"          # awaiting human review (Part 5)
    EXECUTING = "executing"
    VERIFYING = "verifying"
    RESOLVED = "resolved"
    REOPENED = "reopened"            # edge case 6.3


class ExceptionCase(Base):
    """One row per order-exception case. This IS the durable state that
    lets a case survive a multi-day carrier wait or a process restart —
    see architecture doc Layer 5 / Part 1's autonomy-per-subworkflow design."""
    __tablename__ = "exception_cases"

    id = Column(String, primary_key=True, default=_uuid)
    order_id = Column(String, nullable=False, index=True)
    customer_id = Column(String, nullable=False, index=True)
    channel = Column(String, nullable=False)  # direct | amazon | walmart ... (edge case 7.1)
    exception_type = Column(String, nullable=False)  # payment | inventory | carrier | return | fraud
    state = Column(Enum(CaseState), nullable=False, default=CaseState.DETECTED, index=True)

    # Populated as the case progresses through sub-agents
    diagnosis = Column(JSON, nullable=True)
    fraud_risk_score = Column(Float, nullable=True)
    fraud_flag = Column(String, nullable=True)  # null | "flagged"
    resolution_decision = Column(JSON, nullable=True)  # {"action": "refund", "amount": 42.0, "reasoning": "...", "confidence": 0.93, "cited_policy": {...}}
    execution_result = Column(JSON, nullable=True)
    verification_result = Column(JSON, nullable=True)

    idempotency_key = Column(String, nullable=True, unique=True)  # Layer 5

    created_at = Column(DateTime(timezone=True), default=_now)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)
    resolved_at = Column(DateTime(timezone=True), nullable=True)

    audit_entries = relationship("AuditLogEntry", back_populates="case", cascade="all, delete-orphan")


class AuditLogEntry(Base):
    """Append-only audit trail per architecture doc Layer 13 / 8.8.
    Every state transition and every agent/tool decision writes one row here.
    Never updated or deleted once written."""
    __tablename__ = "audit_log_entries"

    id = Column(String, primary_key=True, default=_uuid)
    case_id = Column(String, ForeignKey("exception_cases.id"), nullable=False, index=True)
    actor = Column(String, nullable=False)  # e.g. "diagnosis_agent", "human:jane@company.com", "system"
    action = Column(String, nullable=False)  # e.g. "state_transition", "tool_call", "human_decision"
    detail = Column(JSON, nullable=False)     # structured input/output/metadata, per 8.8's span schema
    timestamp = Column(DateTime(timezone=True), default=_now)

    case = relationship("ExceptionCase", back_populates="audit_entries")


class IdempotencyRecord(Base):
    """Layer 5 reliability mechanism, enforced at the tool layer (not just
    assumed at the application layer) — every write-capable tool call
    (refund, inventory transfer, label generation) checks this table
    BEFORE executing. If the same idempotency_key was already used, the
    stored result is returned and the underlying gateway call is skipped
    entirely — this is what makes a retried webhook or a duplicate case
    escalation safe rather than a duplicate refund incident."""
    __tablename__ = "idempotency_records"

    idempotency_key = Column(String, primary_key=True)
    tool_name = Column(String, nullable=False)       # e.g. "stripe_refund", "wms_transfer"
    request_fingerprint = Column(String, nullable=False)  # hash of the request args, to detect key reuse with different args
    result = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)


class EpisodeRecord(Base):
    """Backing store for the episodic/long-term memory layer (Phase 5,
    architecture doc 8.4). Production pick is Zep/Graphiti (chosen there
    specifically for time-aware relationship reasoning over a graph DB) —
    not runnable in this sandbox (needs Neo4j/FalkorDB, no Docker/network
    access here). This table gives the same FUNCTIONAL contract for the
    subset this project actually exercises (time-ordered customer history
    queries) via plain SQL — see app/memory/episodic.py and its module
    docstring for the exact swap point and what's lost (multi-hop graph
    traversal) versus a real Graphiti-backed deployment."""
    __tablename__ = "episode_records"

    id = Column(String, primary_key=True, default=_uuid)
    customer_id = Column(String, nullable=False, index=True)
    case_id = Column(String, nullable=True, index=True)
    episode_type = Column(String, nullable=False)   # e.g. "case_resolved", "fraud_flag_raised"
    content = Column(JSON, nullable=False)            # structured episode data
    occurred_at = Column(DateTime(timezone=True), nullable=False)  # when it happened, not when logged
    created_at = Column(DateTime(timezone=True), default=_now)      # when it was logged


class TraceSpanRecord(Base):
    """Phase 10 / architecture doc 8.8 — Langfuse's span schema
    implemented directly against SQL, since this sandbox has no network
    access to Langfuse Cloud and no Docker to self-host it. One trace per
    case (trace_id == case_id), one span per agent/tool call within it —
    exactly the structure 8.8 specifies. See app/core/tracing.py for the
    Langfuse swap point."""
    __tablename__ = "trace_span_records"

    span_id = Column(String, primary_key=True, default=_uuid)
    trace_id = Column(String, nullable=False, index=True)   # == case_id
    parent_span_id = Column(String, nullable=True)
    agent_or_tool_name = Column(String, nullable=False)
    input = Column(JSON, nullable=False)
    output = Column(JSON, nullable=False)
    span_metadata = Column(JSON, nullable=False)  # model_used, latency_ms, retrieval_doc_ids_used, etc — "metadata" is reserved by SQLAlchemy's declarative API
    created_at = Column(DateTime(timezone=True), default=_now)


class AlertRecord(Base):
    """Phase 10 — alerting log. See app/core/alerting.py for the Slack
    webhook swap point (no network access to any webhook from this
    sandbox); this table is the durable record either way."""
    __tablename__ = "alert_records"

    id = Column(String, primary_key=True, default=_uuid)
    event_type = Column(String, nullable=False)   # "circuit_breaker_trip" | "idempotency_collision" | "tier1_block"
    detail = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_now)


class JobRecord(Base):
    """Phase 12 / architecture doc 8.1: the async job queue. No Redis in
    this sandbox (no Docker, no network), so this table IS the queue —
    enqueue_job() writes a row with status='queued', a worker (or, in this
    build, an explicit process_next_job() call — see app/workers/queue.py)
    picks it up. This is what makes "webhook handlers ack immediately and
    enqueue, never process synchronously inline" a real, testable
    guarantee: the webhook route handler's entire body is one INSERT,
    nothing else."""
    __tablename__ = "job_records"

    id = Column(String, primary_key=True, default=_uuid)
    job_type = Column(String, nullable=False, index=True)  # "webhook_oms" | "webhook_payment" | "webhook_carrier" | "webhook_wms" | "ingest_policy"
    payload = Column(JSON, nullable=False)
    status = Column(String, nullable=False, default="queued", index=True)  # queued | processing | done | failed
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now)
    processed_at = Column(DateTime(timezone=True), nullable=True)


class MockOrderRecord(Base):
    """Backing store for the mock OMS tool (Phase 4) — a real order table
    a real OMS would have, just self-hosted here since this project has no
    access to an actual OMS. The Diagnosis Agent (Phase 6) queries this
    exactly as it would query a real OMS's read API."""
    __tablename__ = "mock_orders"

    order_id = Column(String, primary_key=True)
    customer_id = Column(String, nullable=False, index=True)
    channel = Column(String, nullable=False)
    status = Column(String, nullable=False)          # e.g. "paid", "payment_failed", "shipped", "delivered"
    payment_intent_id = Column(String, nullable=True)  # links to the payment gateway's transaction — needed by the Execution Agent to issue a refund
    total_amount_usd = Column(Float, nullable=False)
    payment_method_breakdown = Column(JSON, nullable=True)  # edge case 2.1: split payment methods
    purchase_date = Column(DateTime(timezone=True), nullable=False)  # drives temporal policy binding (RAG 8.2.4)
    line_items = Column(JSON, nullable=False)          # [{"sku": ..., "category": ..., "qty": ..., "price": ...}]
    created_at = Column(DateTime(timezone=True), default=_now)


class MockInventoryRecord(Base):
    """Backing store for the mock WMS tool — tracks on_hand vs sellable
    quantity separately per architecture doc edge case 3.1/3.2 (phantom
    stock / damaged-quarantined stock)."""
    __tablename__ = "mock_inventory"

    id = Column(String, primary_key=True, default=_uuid)
    sku = Column(String, nullable=False, index=True)
    warehouse = Column(String, nullable=False)
    on_hand_qty = Column(Float, nullable=False, default=0)
    sellable_qty = Column(Float, nullable=False, default=0)  # <= on_hand_qty; excludes damaged/quarantined
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)


class ResolutionPatternEntry(Base):
    """Phase 9 / architecture doc 8.5 — learning-loop store.
    One row per human-reviewed case outcome, used for dynamic few-shot
    retrieval and (human-gated) confidence recalibration."""
    __tablename__ = "resolution_pattern_entries"

    id = Column(String, primary_key=True, default=_uuid)
    case_id = Column(String, ForeignKey("exception_cases.id"), nullable=False)
    cluster_key = Column(String, nullable=False, index=True)  # coarse grouping for accuracy stats, e.g. "return_low_value"
    case_feature_summary = Column(Text, nullable=False)  # embedded for similarity retrieval later
    agent_proposed_resolution = Column(JSON, nullable=False)
    human_final_resolution = Column(JSON, nullable=False)
    matched = Column(String, nullable=False)  # "match" | "mismatch"
    created_at = Column(DateTime(timezone=True), default=_now)


class ThresholdProposalRecord(Base):
    """Phase 9 — a PROPOSED confidence-threshold adjustment for one
    cluster, computed from observed accuracy. Never applied automatically
    — see ThresholdOverrideRecord below, which only gets written by an
    explicit human-accept action (app/agents/learning_loop.py's
    accept_threshold_proposal()). This separation is the actual guardrail
    the architecture doc calls for: a proposal existing in this table has
    ZERO effect on the Resolution-Policy Workflow's behavior until a human
    accepts it."""
    __tablename__ = "threshold_proposal_records"

    id = Column(String, primary_key=True, default=_uuid)
    cluster_key = Column(String, nullable=False, index=True)
    sample_size = Column(Float, nullable=False)
    overturn_rate = Column(Float, nullable=False)
    current_threshold = Column(Float, nullable=False)
    proposed_threshold = Column(Float, nullable=False)
    rationale = Column(Text, nullable=False)
    status = Column(String, nullable=False, default="pending_review")  # pending_review | accepted | rejected
    created_at = Column(DateTime(timezone=True), default=_now)
    decided_at = Column(DateTime(timezone=True), nullable=True)
    decided_by = Column(String, nullable=True)  # e.g. "human:jane@company.com" — never "system"


class ThresholdOverrideRecord(Base):
    """The ONLY table the Resolution-Policy Workflow would actually read
    an adjusted threshold from — written exclusively by
    accept_threshold_proposal(), never by the proposal-generation batch
    job itself. One row per cluster (upsert on accept)."""
    __tablename__ = "threshold_override_records"

    cluster_key = Column(String, primary_key=True)
    active_threshold = Column(Float, nullable=False)
    accepted_from_proposal_id = Column(String, nullable=False)
    accepted_by = Column(String, nullable=False)
    accepted_at = Column(DateTime(timezone=True), default=_now)


class GoldenSetRunRecord(Base):
    """One row per labeled golden-set run triggered from the UI's
    Testing page. Distinct from the golden baseline file
    (data/golden_set_baseline.json, used by drift_watch.py) — this table
    is a HISTORY of every run a person has explicitly triggered and
    named, so multiple runs (e.g. before/after a code change) can be
    compared side by side rather than each overwriting the last."""
    __tablename__ = "golden_set_run_records"

    id = Column(String, primary_key=True)
    label = Column(String, nullable=False)
    run_at = Column(DateTime(timezone=True), default=_now)
    pass_count = Column(Integer, nullable=False)
    total_count = Column(Integer, nullable=False)
    results = Column(JSON, nullable=False)  # list of {scenario_name, passed, detail}
    triggered_by = Column(String, nullable=False, default="ui")


# --- Engine / session ---
settings = get_settings()
_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(
    settings.database_url,
    connect_args=_connect_args,
    # pool_pre_ping + pool_recycle: found necessary in production against
    # a real cloud Postgres (Neon/Supabase). Managed/serverless Postgres
    # providers aggressively close idle connections server-side to save
    # resources — SQLAlchemy's default connection pool has no way to
    # know a pooled connection has gone stale until it actually tries to
    # use it, which surfaces as "psycopg2.OperationalError: server closed
    # the connection unexpectedly" on whatever query happens to run
    # first after an idle period. This hit in exactly that shape: an RQ
    # worker sitting idle waiting for jobs, then failing on the very
    # first query once a job finally arrived.
    #   - pool_pre_ping=True: runs a lightweight "is this connection
    #     still alive" check before handing a pooled connection to a
    #     real query, transparently reconnecting if it's dead. Small
    #     latency cost per checkout, in exchange for never seeing this
    #     error again — the standard, documented fix for this exact
    #     class of issue with any serverless/managed Postgres.
    #   - pool_recycle=280: proactively discards and replaces a pooled
    #     connection older than this many seconds, rather than relying
    #     solely on pre_ping to catch every case. 280s is comfortably
    #     under most managed providers' typical idle-connection-close
    #     windows (often 5 minutes / 300s) — recycling just under that
    #     avoids racing the server's own timeout.
    # Harmless no-ops for local SQLite (no idle-connection-close
    # behavior to protect against there), so applied unconditionally
    # rather than branching on dialect.
    pool_pre_ping=True,
    pool_recycle=280,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create tables if they don't exist. Fine for SQLite/dev; use Alembic
    migrations once this moves to Postgres in staging (documented in a later phase)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
