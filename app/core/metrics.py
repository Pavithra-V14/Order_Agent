"""
Metrics - architecture doc 8.7's full list, computed from TraceSpanRecord,
ExceptionCase, IdempotencyRecord, and AlertRecord. Feeds the /metrics/{scope}
API endpoints which the Metrics Dashboard frontend page will render.
"""
from __future__ import annotations

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.db import ExceptionCase, CaseState, TraceSpanRecord, IdempotencyRecord, AlertRecord


def compute_agent_metrics(db: Session) -> dict:
    total_cases = db.query(ExceptionCase).count()
    by_state = dict(
        db.query(ExceptionCase.state, func.count(ExceptionCase.id)).group_by(ExceptionCase.state).all()
    )
    escalated = by_state.get(CaseState.ESCALATED, 0)
    resolved = by_state.get(CaseState.RESOLVED, 0)

    diagnosis_spans = db.query(TraceSpanRecord).filter(TraceSpanRecord.agent_or_tool_name == "diagnosis_agent").all()
    avg_steps = None
    if diagnosis_spans:
        step_counts = [len(s.output.get("steps_taken", [])) for s in diagnosis_spans if isinstance(s.output, dict)]
        if step_counts:
            avg_steps = sum(step_counts) / len(step_counts)

    return {
        "total_cases": total_cases,
        "cases_by_state": {(k.value if hasattr(k, "value") else str(k)): v for k, v in by_state.items()},
        "escalation_rate": (escalated / total_cases) if total_cases else 0.0,
        "resolution_rate": (resolved / total_cases) if total_cases else 0.0,
        "avg_diagnosis_steps": avg_steps,
    }


def compute_tool_metrics(db: Session) -> dict:
    idempotency_by_tool = dict(
        db.query(IdempotencyRecord.tool_name, func.count(IdempotencyRecord.idempotency_key))
        .group_by(IdempotencyRecord.tool_name).all()
    )
    circuit_trips = db.query(AlertRecord).filter(AlertRecord.event_type == "circuit_breaker_trip").count()
    idempotency_collisions = db.query(AlertRecord).filter(AlertRecord.event_type == "idempotency_collision").count()

    return {
        "successful_write_calls_by_tool": idempotency_by_tool,
        "circuit_breaker_trip_count": circuit_trips,
        "idempotency_collision_count": idempotency_collisions,
    }


def compute_rag_metrics(db: Session) -> dict:
    """The groundedness metric: for every resolution_decision span that
    cites a policy doc_id, was that doc_id actually present in a
    rag_retrieval span's retrieval_doc_ids_used within the SAME trace
    (case)? Computed from trace data, not asserted."""
    retrieval_spans = db.query(TraceSpanRecord).filter(TraceSpanRecord.agent_or_tool_name == "rag_retrieval").all()
    decision_spans = db.query(TraceSpanRecord).filter(TraceSpanRecord.agent_or_tool_name == "resolution_decision").all()

    retrieved_doc_ids_by_trace = {}
    retrieved_versions_by_trace = {}
    for s in retrieval_spans:
        retrieved_doc_ids_by_trace.setdefault(s.trace_id, set()).update(
            s.span_metadata.get("retrieval_doc_ids_used", [])
        )
        retrieved_versions_by_trace.setdefault(s.trace_id, set()).update(
            s.span_metadata.get("retrieval_doc_versions_used", [])
        )

    grounded_count = 0
    total_citing_decisions = 0
    per_case_groundedness = []

    for s in decision_spans:
        cited_doc_id = s.span_metadata.get("cited_doc_id")
        if not cited_doc_id:
            continue
        total_citing_decisions += 1
        retrieved_for_trace = retrieved_doc_ids_by_trace.get(s.trace_id, set())
        retrieved_versions_for_trace = retrieved_versions_by_trace.get(s.trace_id, set())
        is_grounded = cited_doc_id in retrieved_for_trace
        if is_grounded:
            grounded_count += 1
        per_case_groundedness.append({
            "trace_id": s.trace_id,
            "cited_doc_id": cited_doc_id,
            "cited_version": s.span_metadata.get("cited_version"),
            "grounded": is_grounded,
            "actually_retrieved_doc_ids": sorted(retrieved_for_trace),
            "actually_retrieved_versions": sorted(retrieved_versions_for_trace),
        })

    groundedness_score = (grounded_count / total_citing_decisions) if total_citing_decisions else None

    return {
        "retrieval_span_count": len(retrieval_spans),
        "citing_decision_count": total_citing_decisions,
        "groundedness_score": groundedness_score,
        "per_case_groundedness": per_case_groundedness,
    }


def compute_system_metrics(db: Session) -> dict:
    all_spans = db.query(TraceSpanRecord).all()
    total_spans = len(all_spans)
    error_spans = sum(1 for s in all_spans if isinstance(s.output, dict) and "error" in s.output)
    all_alerts = db.query(AlertRecord).count()

    latencies = [
        s.span_metadata.get("latency_ms") for s in all_spans
        if isinstance(s.span_metadata, dict) and s.span_metadata.get("latency_ms") is not None
    ]
    avg_latency_ms = (sum(latencies) / len(latencies)) if latencies else None

    return {
        "total_spans_recorded": total_spans,
        "error_span_count": error_spans,
        "error_rate": (error_spans / total_spans) if total_spans else 0.0,
        "total_alerts": all_alerts,
        "avg_span_latency_ms": round(avg_latency_ms, 2) if avg_latency_ms is not None else None,
    }
