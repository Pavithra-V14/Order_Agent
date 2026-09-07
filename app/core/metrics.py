"""
Metrics - architecture doc 8.7's full list, computed from TraceSpanRecord,
ExceptionCase, IdempotencyRecord, AuditLogEntry, and AlertRecord.

HONEST STATUS (found and documented directly, not glossed over): the
guide's Part 8.7 lists ~23 distinct named metrics across 4 categories.
This file originally implemented about 6 of them, then per-tool
failure/latency/read-write-ratio were added once tool-call tracing
existed. Precision/recall@k now lives separately in app/rag/eval.py +
GET /testing/rag-eval (a live eval against a labeled query set, not a
passively-computed metric from historical trace data — it belongs with
the Testing page's other on-demand checks, not mixed into this file's
passive metrics). Several metrics below genuinely require either new
instrumentation this build hasn't added yet, or external business inputs
this system has no way to know on its own. Both kinds are named
explicitly below, not silently dropped.

STILL MISSING, and why:
- Tool selection accuracy, tool-call argument validity rate: these
  specifically (NOT per-tool failure rate/latency/read-write ratio,
  which ARE now computed below) require ground-truth labeling of
  whether the RIGHT tool was chosen and whether arguments were CORRECT
  for the situation — a call can succeed with the wrong tool chosen, or
  fail with the right tool for an unrelated reason, so success/failure
  alone can't answer either question.
- Reranker lift and index freshness lag are NO LONGER missing — see
  GET /testing/reranker-lift (pre/post-rerank rank comparison via
  hybrid_search's pre_rerank_capture hook) and GET /testing/index-freshness
  (computed from reindex_state.json's own indexed_at/source_mtime timestamps).
- Guardrail trigger accuracy (false positive/negative rate): requires
  ground-truth human feedback on whether each block/escalate was
  CORRECT, which is inherently a human-review data source, not
  something computable from telemetry alone.
- Cost per successful workflow, user adoption rate, business KPI delta:
  the guide itself flags these as "most commonly skipped" for a reason —
  they need external inputs this system doesn't model at all (LLM/API
  token pricing, a defined "eligible user" population, a pre-agent
  baseline to compare against). Not a code gap; a business-input gap
  that needs a defined data source before it's meaningful to compute.
"""
from __future__ import annotations

from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.db import ExceptionCase, CaseState, TraceSpanRecord, IdempotencyRecord, AlertRecord, AuditLogEntry


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

    # Loop/step-ceiling hit rate — computable from AuditLogEntry's
    # "diagnosis_complete" entries (app/agents/orchestrator.py already
    # writes result.terminated_reason into detail on every diagnosis run;
    # no new instrumentation needed, just a query that wasn't written yet).
    diagnosis_complete_entries = db.query(AuditLogEntry).filter(AuditLogEntry.action == "diagnosis_complete").all()
    terminated_reason_counts = {}
    for e in diagnosis_complete_entries:
        reason = (e.detail or {}).get("terminated_reason", "unknown")
        terminated_reason_counts[reason] = terminated_reason_counts.get(reason, 0) + 1
    total_diagnoses = sum(terminated_reason_counts.values())
    step_ceiling_hit_rate = (
        terminated_reason_counts.get("max_steps_reached", 0) / total_diagnoses
    ) if total_diagnoses else None

    # Escalation reason distribution — from ExceptionCase.resolution_decision's
    # routing_reasons where present (populated by resolution_policy_workflow's
    # ResolutionResult; stored on the case by whichever caller applies it).
    escalated_cases = db.query(ExceptionCase).filter(ExceptionCase.state == CaseState.ESCALATED).all()
    escalation_reason_counts = {}
    for c in escalated_cases:
        if c.fraud_flag:
            escalation_reason_counts["fraud_flag"] = escalation_reason_counts.get("fraud_flag", 0) + 1
        elif c.resolution_decision and c.resolution_decision.get("confidence", 1.0) < 0.90:
            escalation_reason_counts["low_confidence"] = escalation_reason_counts.get("low_confidence", 0) + 1
        else:
            escalation_reason_counts["other"] = escalation_reason_counts.get("other", 0) + 1

    return {
        "total_cases": total_cases,
        "cases_by_state": {(k.value if hasattr(k, "value") else str(k)): v for k, v in by_state.items()},
        "escalation_rate": (escalated / total_cases) if total_cases else 0.0,
        "resolution_rate": (resolved / total_cases) if total_cases else 0.0,
        "avg_diagnosis_steps": avg_steps,
        "diagnosis_terminated_reason_distribution": terminated_reason_counts,
        "step_ceiling_hit_rate": step_ceiling_hit_rate,
        "escalation_reason_distribution": escalation_reason_counts,
    }


def compute_tool_metrics(db: Session) -> dict:
    idempotency_by_tool = dict(
        db.query(IdempotencyRecord.tool_name, func.count(IdempotencyRecord.idempotency_key))
        .group_by(IdempotencyRecord.tool_name).all()
    )
    circuit_trips = db.query(AlertRecord).filter(AlertRecord.event_type == "circuit_breaker_trip").count()
    idempotency_collisions = db.query(AlertRecord).filter(AlertRecord.event_type == "idempotency_collision").count()

    # Per-tool failure rate/latency + read/write ratio — the previously
    # missing metrics, now computable now that every real tool call
    # (payment/wms/carrier/oms) is traced via record_tool_call(), not
    # just resolution_policy_workflow's own single span.
    tool_spans = db.query(TraceSpanRecord).filter(TraceSpanRecord.agent_or_tool_name.like("tool:%")).all()
    per_tool = {}
    for s in tool_spans:
        name = s.agent_or_tool_name[len("tool:"):]
        meta = s.span_metadata or {}
        bucket = per_tool.setdefault(name, {"calls": 0, "failures": 0, "latencies_ms": [], "is_write": meta.get("is_write", False)})
        bucket["calls"] += 1
        if meta.get("status") == "failed":
            bucket["failures"] += 1
        if meta.get("latency_ms") is not None:
            bucket["latencies_ms"].append(meta["latency_ms"])

    per_tool_summary = {}
    total_read_calls = 0
    total_write_calls = 0
    for name, bucket in per_tool.items():
        latencies = bucket["latencies_ms"]
        per_tool_summary[name] = {
            "call_count": bucket["calls"],
            "failure_rate": (bucket["failures"] / bucket["calls"]) if bucket["calls"] else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "is_write": bucket["is_write"],
        }
        if bucket["is_write"]:
            total_write_calls += bucket["calls"]
        else:
            total_read_calls += bucket["calls"]

    total_tool_calls = total_read_calls + total_write_calls
    read_write_ratio = (total_read_calls / total_write_calls) if total_write_calls else None

    return {
        "successful_write_calls_by_tool": idempotency_by_tool,
        "circuit_breaker_trip_count": circuit_trips,
        "idempotency_collision_count": idempotency_collisions,
        "per_tool_metrics": per_tool_summary,
        "total_tool_calls": total_tool_calls,
        "read_call_count": total_read_calls,
        "write_call_count": total_write_calls,
        "read_write_ratio": round(read_write_ratio, 2) if read_write_ratio is not None else None,
        # Tool selection accuracy and argument validity rate still
        # require ground-truth labeling (was the RIGHT tool chosen for
        # this situation, were these the CORRECT arguments) that isn't
        # computable from call success/failure alone — a call can
        # succeed with the wrong tool, or fail with the right one for
        # an unrelated reason. Not computed here, not faked.
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

    # Retrieval hit rate — % of retrieval spans that returned at least
    # one result. Computable directly from num_results already recorded
    # on every rag_retrieval span (app/rag/traced_retrieval.py) — no new
    # instrumentation needed.
    retrievals_with_results = sum(
        1 for s in retrieval_spans if isinstance(s.output, dict) and s.output.get("num_results", 0) > 0
    )
    retrieval_hit_rate = (retrievals_with_results / len(retrieval_spans)) if retrieval_spans else None

    return {
        "retrieval_span_count": len(retrieval_spans),
        "retrieval_hit_rate": retrieval_hit_rate,
        "citing_decision_count": total_citing_decisions,
        "groundedness_score": groundedness_score,
        "per_case_groundedness": per_case_groundedness,
        # See module docstring: precision/recall@k, reranker lift, and
        # index freshness lag all require new instrumentation/eval
        # artifacts not yet built.
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
    p95_latency_ms = None
    if latencies:
        sorted_latencies = sorted(latencies)
        p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]

    # Audit log completeness % — every case must have at least one
    # AuditLogEntry (its own state_transition into DETECTED, at minimum).
    # Computable directly from existing data, catches a real regression
    # class: a case created via some code path that skips writing its
    # initial audit row.
    total_cases = db.query(ExceptionCase).count()
    cases_with_audit_entries = db.query(AuditLogEntry.case_id).distinct().count()
    audit_completeness_pct = (cases_with_audit_entries / total_cases * 100) if total_cases else None

    return {
        "total_spans_recorded": total_spans,
        "error_span_count": error_spans,
        "error_rate": (error_spans / total_spans) if total_spans else 0.0,
        "total_alerts": all_alerts,
        "avg_span_latency_ms": round(avg_latency_ms, 2) if avg_latency_ms is not None else None,
        "p95_span_latency_ms": round(p95_latency_ms, 2) if p95_latency_ms is not None else None,
        "audit_log_completeness_pct": round(audit_completeness_pct, 1) if audit_completeness_pct is not None else None,
        # See module docstring: cost per successful workflow, guardrail
        # trigger accuracy, user adoption rate, and business KPI delta
        # all require external inputs (pricing, human review labels, a
        # defined user population, a pre-agent baseline) this system has
        # no way to know from telemetry alone — not computed here, and
        # deliberately not faked with a placeholder number.
    }
