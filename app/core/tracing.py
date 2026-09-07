"""
Tracing - architecture doc 8.8. Production pick is Langfuse (self-hosted
or cloud). No network access to Langfuse Cloud and no Docker to self-host
in this sandbox, so spans are recorded directly to TraceSpanRecord
(app/core/db.py) with the exact same shape 8.8 specifies: one trace per
case, one span per agent/tool call, input/output/metadata on every span.

PII redaction happens BEFORE persist, not after - per 8.8's explicit
requirement, since logs are a permanent record and can't be safely
scrubbed retroactively with the same confidence. Reuses the same
regex-based scanner from Tier 2 guardrails rather than a separate
mechanism.

Swap point: replace record_span()'s body with a Langfuse SDK call once
self-hosted Langfuse or a cloud API key is available - the calling code
in every agent/tool wrapper stays identical.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.core.db import TraceSpanRecord
from app.guardrails.tier2_structural import scan_for_pii


def _redact_pii(value):
    """Recursively redacts PII from strings within a nested dict/list
    structure, using the same regex scanner Tier 2 guardrails uses."""
    if isinstance(value, str):
        findings = scan_for_pii(value)
        redacted = value
        for f in sorted(findings, key=lambda x: x["span"][0], reverse=True):
            start, end = f["span"]
            redacted = redacted[:start] + f"[REDACTED_{f['type'].upper()}]" + redacted[end:]
        return redacted
    if isinstance(value, dict):
        return {k: _redact_pii(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_pii(v) for v in value]
    return value


def record_span(
    db: Session,
    trace_id: str,
    agent_or_tool_name: str,
    input_data: dict,
    output_data: dict,
    metadata: dict = None,
    parent_span_id: str = None,
) -> TraceSpanRecord:
    """Records one span. PII-redacts input/output before persist.

    Dual-writes to Langfuse Cloud when configured (LANGFUSE_PUBLIC_KEY /
    LANGFUSE_SECRET_KEY + TRACING_ENABLED=true) — SQL remains the primary
    store this project's own metrics (app/core/metrics.py, including the
    RAG groundedness computation) query directly, since that needs
    structured SQL querying Langfuse's hosted UI doesn't expose an API
    for; Langfuse Cloud is an ADDITIONAL sink giving a real hosted trace
    UI on top of the same data. A Langfuse failure (network, bad
    credentials) is caught and logged, never allowed to break the
    business logic that's actually being traced — observability must not
    become a new failure mode for the system it's observing.
    """
    span = TraceSpanRecord(
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        agent_or_tool_name=agent_or_tool_name,
        input=_redact_pii(input_data),
        output=_redact_pii(output_data),
        span_metadata=metadata or {},
    )
    db.add(span)
    db.commit()

    _push_to_langfuse(trace_id, agent_or_tool_name, span.input, span.output, span.span_metadata, parent_span_id)

    return span


def record_tool_call(db: Session, trace_id: str, tool_name: str, is_write: bool, fn, *args, **kwargs):
    """Wraps a single tool call (payment/wms/carrier/oms) with tracing —
    the previously-missing per-tool-call instrumentation that
    app/core/metrics.py's docstring documented as needed for per-tool
    failure rate/latency, read/write ratio, and argument validity rate.
    Before this, only resolution_policy_workflow.py ever called
    record_span() directly; no individual tool call was traced at all.

    Also prints a live line to the terminal via app.core.console_log —
    a real, separately-requested piece of operational visibility: the
    DB-persisted trace span is queryable after the fact, but gives no
    indication while WATCHING a running process of what's happening in
    real time.

    Records success or failure, latency, and read/write classification
    on every call — the call itself is never suppressed or altered:
    exceptions still propagate normally to the caller after being
    recorded, so this never changes the actual behavior of a failing
    tool call, only what gets observed about it.
    """
    from app.core.console_log import log_tool_call
    start = time.monotonic()
    try:
        result = fn(*args, **kwargs)
        latency_ms = (time.monotonic() - start) * 1000
        record_span(
            db, trace_id=trace_id, agent_or_tool_name=f"tool:{tool_name}",
            input_data={"args_repr": repr(args)[:500], "kwargs_repr": repr(kwargs)[:500]},
            output_data={"success": True},
            metadata={"latency_ms": round(latency_ms, 2), "is_write": is_write, "status": "success"},
        )
        log_tool_call(tool_name, is_write, "success", latency_ms, case_id=trace_id)
        return result
    except Exception as e:
        latency_ms = (time.monotonic() - start) * 1000
        try:
            record_span(
                db, trace_id=trace_id, agent_or_tool_name=f"tool:{tool_name}",
                input_data={"args_repr": repr(args)[:500], "kwargs_repr": repr(kwargs)[:500]},
                output_data={"error": str(e)},
                metadata={"latency_ms": round(latency_ms, 2), "is_write": is_write, "status": "failed"},
            )
        except Exception:
            pass  # tracing must never mask the original tool failure below
        log_tool_call(tool_name, is_write, "failed", latency_ms, case_id=trace_id)
        raise


def _push_to_langfuse(trace_id: str, name: str, input_data: dict, output_data: dict,
                       metadata: dict, parent_span_id: str = None) -> None:
    from app.core.config import get_settings
    settings = get_settings()
    if not (settings.tracing_enabled and settings.langfuse_public_key and settings.langfuse_secret_key):
        return

    try:
        client = _get_langfuse_client()
        langfuse_trace_id = client.create_trace_id(seed=trace_id)
        trace_context = {"trace_id": langfuse_trace_id}
        if parent_span_id:
            trace_context["parent_span_id"] = parent_span_id
        span = client.start_observation(
            trace_context=trace_context, name=name,
            input=input_data, output=output_data, metadata=metadata,
        )
        span.end()
        # Found necessary directly: Langfuse's v4 SDK is OpenTelemetry-
        # based and batches spans internally, only actually sending them
        # to the server on flush() or process shutdown. Without this
        # call, a span created inside a normal request-handling flow (or
        # a short-lived script) can sit in an internal buffer and simply
        # never reach Langfuse's backend — which is exactly why nothing
        # showed up in the Langfuse dashboard despite this code
        # appearing to run without error. Flushing per-span trades a
        # little batching efficiency for guaranteed delivery, an
        # acceptable tradeoff given this project's actual per-case span
        # volume is low.
        client.flush()
    except Exception as e:
        # Never let an observability-layer failure break the traced
        # operation itself — logged, not raised.
        import logging
        logging.getLogger("tracing").warning("Langfuse push failed (non-fatal): %s", e)


_langfuse_client = None


def _get_langfuse_client():
    global _langfuse_client
    if _langfuse_client is None:
        from langfuse import Langfuse
        from app.core.config import get_settings
        settings = get_settings()
        _langfuse_client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    return _langfuse_client


def reset_langfuse_client() -> None:
    """Test helper."""
    global _langfuse_client
    _langfuse_client = None


@dataclass
class SpanTimer:
    """Context-manager helper: captures latency automatically and lets the
    caller fill in output/metadata after the wrapped call completes."""
    db: Session
    trace_id: str
    agent_or_tool_name: str
    input_data: dict
    parent_span_id: str = None
    output_data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    _start: float = field(default=0.0, init=False)

    def __enter__(self):
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        latency_ms = (time.monotonic() - self._start) * 1000
        self.metadata["latency_ms"] = round(latency_ms, 2)
        if exc_type is not None:
            self.output_data = {"error": str(exc_val)}
        record_span(self.db, self.trace_id, self.agent_or_tool_name,
                    self.input_data, self.output_data, self.metadata, self.parent_span_id)
        return False


@contextmanager
def traced_span(db: Session, trace_id: str, agent_or_tool_name: str, input_data: dict,
                 parent_span_id: str = None):
    timer = SpanTimer(db=db, trace_id=trace_id, agent_or_tool_name=agent_or_tool_name,
                       input_data=input_data, parent_span_id=parent_span_id)
    with timer:
        yield timer


def get_trace(db: Session, trace_id: str) -> list:
    """Returns all spans for one trace (case), in chronological order."""
    spans = (
        db.query(TraceSpanRecord)
        .filter(TraceSpanRecord.trace_id == trace_id)
        .order_by(TraceSpanRecord.created_at.asc())
        .all()
    )
    return [
        {
            "span_id": s.span_id, "parent_span_id": s.parent_span_id,
            "agent_or_tool_name": s.agent_or_tool_name,
            "input": s.input, "output": s.output, "metadata": s.span_metadata,
            "created_at": s.created_at.isoformat(),
        }
        for s in spans
    ]
