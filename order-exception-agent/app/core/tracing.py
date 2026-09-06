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
