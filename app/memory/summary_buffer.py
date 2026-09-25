"""
Short-term per-case working memory - architecture doc 8.4's Summary
Buffer: recent tool outputs kept verbatim (bounded), older context
collapsed into a running summary. This is deliberately a lightweight,
custom implementation, not a heavy framework - the architecture doc is
explicit that per-case working memory doesn't need Zep/Graphiti's
machinery, only the long-term/cross-case layer does (episodic.py).

Stage 2 memory upgrade — two real gaps closed, found during a direct
audit after Stage 1's cross-customer fraud work:

  1. LLM summarization was a permanent stub (_summarize did a plain
     string concatenation, never a real fold) even after Phase 6 wired
     an LLM client into the rest of the diagnosis loop - this class's
     own docstring said "Phase 6 plugs in a real summarization call
     without changing this class's interface", but that follow-up never
     happened. Fixed: add() now accepts an optional llm (any
     BaseLLMClient, real or FakeLLMClient) and calls its
     summarize_context() - a genuine LLM fold when a real client is
     configured, the same deterministic behavior as before when
     FakeLLMClient is used or no llm is given at all (backward
     compatible - existing direct SummaryBuffer.add() callers with no
     llm argument, e.g. tests/test_phase5_memory.py, are unaffected).
  2. The buffer was pure in-process state (_buffers dict) with no
     persistence at all - a process restart mid-diagnosis silently lost
     it, despite Layer 5's durable-state requirement covering every
     other piece of case state (the case row itself, the audit log).
     Fixed: load_buffer_state()/persist_buffer_state() read/write
     ExceptionCase.working_memory_summary (a JSON field, no schema
     migration needed - same pattern already used for
     resolution_decision). HONEST SCOPE: this persists the buffer's
     last-known state so it's recoverable/inspectable after a crash
     (e.g. via the case detail page) - it does NOT make run_diagnosis()
     itself resumable mid-loop from a saved buffer. Nothing in this
     project's orchestration currently re-enters a diagnosis loop
     partway through, so true resumption is a separate, larger change;
     this closes the "silently lost" gap without overclaiming more.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from app.agents.llm_client import BaseLLMClient


@dataclass
class SummaryBuffer:
    case_id: str
    max_verbatim_items: int = 6
    verbatim_items: list[dict] = field(default_factory=list)
    running_summary: str = ""
    # True-TTL eviction (requested follow-up to Stage 2's event-triggered
    # evict_buffer(), which only fires when a case actually resolves - a
    # case that gets abandoned/stuck mid-diagnosis and never resolves
    # would otherwise sit in _buffers forever). Updated on every add().
    last_touched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add(self, item: dict, llm: "BaseLLMClient | None" = None) -> None:
        """Adds one tool-call result (or agent step) to the buffer. When
        the verbatim window overflows, the oldest item is folded into
        running_summary rather than dropped outright - this is the
        "recent verbatim + summarized older context" shape from 8.4.

        llm (Stage 2, optional): when given, folding goes through a real
        summarize_context() call (see app/agents/llm_client.py) instead
        of the deterministic fallback. A failed real call degrades to
        the same deterministic fold rather than losing the item or
        raising - summarization quality is never allowed to block the
        diagnosis loop it's attached to.
        """
        self.verbatim_items.append(item)
        self.last_touched_at = datetime.now(timezone.utc)
        if len(self.verbatim_items) > self.max_verbatim_items:
            oldest = self.verbatim_items.pop(0)
            self.running_summary = self._summarize(self.running_summary, oldest, llm)

    def _summarize(self, existing_summary: str, new_item: dict, llm: "BaseLLMClient | None" = None) -> str:
        """When llm is given, delegates to its summarize_context() -
        every BaseLLMClient implementation (including FakeLLMClient)
        provides one, so this never needs to branch on client type. When
        llm is None (no client threaded through, e.g. existing direct
        callers/tests), or a real call raises, falls back to the same
        deterministic text fold this class has always used - kept here,
        not deleted, specifically so behavior is identical to before
        Stage 2 for every caller that doesn't opt into passing an llm."""
        if llm is not None:
            try:
                return llm.summarize_context(existing_summary, new_item)
            except Exception as e:
                import logging
                logging.getLogger("summary_buffer").warning(
                    "summarize_context failed, falling back to deterministic fold (non-fatal): %s", e)

        item_desc = f"[{new_item.get('agent', 'unknown')}] {new_item.get('summary', str(new_item))}"
        if not existing_summary:
            return item_desc
        return f"{existing_summary}; {item_desc}"

    def get_context(self) -> dict:
        """What the orchestrator/agents actually read: recent items in
        full plus the folded summary of everything older."""
        return {
            "case_id": self.case_id,
            "running_summary": self.running_summary,
            "recent_items": list(self.verbatim_items),
        }


# In-memory registry keyed by case_id - the fast path within a single
# process/run. Stage 2 adds DB persistence (below) alongside this, not
# instead of it: the in-process dict remains the source of truth WITHIN
# one diagnosis run (cheap, no DB round-trip per step needed to keep
# using it), while persist_buffer_state() gives a case's LAST KNOWN
# buffer state a durable home that survives this dict being cleared or
# the process restarting.
_buffers: dict[str, SummaryBuffer] = {}


def get_or_create_buffer(case_id: str) -> SummaryBuffer:
    if case_id not in _buffers:
        _buffers[case_id] = SummaryBuffer(case_id=case_id)
    return _buffers[case_id]


def reset_buffers() -> None:
    _buffers.clear()


def evict_buffer(case_id: str) -> None:
    """Stage 2 memory upgrade: drops a case's in-process buffer once it's
    no longer needed (a resolved/blocked case has no further diagnosis
    steps coming). Without this, _buffers accumulates one entry per case
    ever diagnosed for the lifetime of the process - a slow, real memory
    leak in a long-running deployment. Safe to call even if the case_id
    was never buffered (e.g. a case that never ran diagnosis).

    HONEST SCOPE: this is EVENT-triggered (fires from
    resolution_completion.py when a case actually resolves), not time-
    based. A case that gets abandoned or stuck mid-diagnosis and never
    reaches resolution would never hit this path and would sit in
    _buffers indefinitely - see evict_stale_buffers() below for the
    genuine time-based backstop that covers exactly that case."""
    _buffers.pop(case_id, None)


def evict_stale_buffers(max_age_seconds: float) -> int:
    """Real TTL eviction, as a follow-up to evict_buffer()'s event-
    triggered version above: removes any buffer whose last_touched_at
    is older than max_age_seconds, regardless of whether its case ever
    resolves. This is the backstop for a case that gets abandoned mid-
    diagnosis (a crashed worker, a case a human simply never comes back
    to) - evict_buffer() alone would never fire for it.

    Deliberately NOT a background thread/scheduler this module starts
    itself - this project has no in-process scheduler anywhere else
    (the confidence-recalibration batch job in this same file is
    exposed the identical way: a function meant to be triggered
    periodically by something external, not by a thread this codebase
    spins up on import). Wired the same way here: POST
    /admin/evict-stale-buffers (app/api/v1/admin.py), meant to be
    called on a schedule (cron, the ops runbook) exactly like
    /threshold-proposals/run-batch-job already is. Returns the number
    of buffers evicted, for that endpoint to report."""
    now = datetime.now(timezone.utc)
    stale_case_ids = [
        case_id for case_id, buf in _buffers.items()
        if (now - buf.last_touched_at).total_seconds() > max_age_seconds
    ]
    for case_id in stale_case_ids:
        _buffers.pop(case_id, None)
    return len(stale_case_ids)


def persist_buffer_state(db: "Session", case_id: str, buffer: SummaryBuffer) -> None:
    """Writes buffer.get_context() onto ExceptionCase.working_memory_summary
    - the JSON field this Stage 2 upgrade adds (app/core/db.py), no
    migration needed (same create_all() pattern as every other JSON
    field in this project). A no-op, not an error, when case_id doesn't
    correspond to a real persisted case (common in tests that call
    run_diagnosis() directly without first creating an ExceptionCase
    row) - this must never be the reason a diagnosis step fails."""
    from app.core.db import ExceptionCase
    case = db.get(ExceptionCase, case_id)
    if case is None:
        return
    case.working_memory_summary = buffer.get_context()
    db.commit()


def load_buffer_state(db: "Session", case_id: str) -> SummaryBuffer:
    """Reconstructs a SummaryBuffer from ExceptionCase.working_memory_summary
    - the counterpart read to persist_buffer_state(). Returns a fresh,
    empty buffer (not an error) when there's no case row or no saved
    state yet, which is the normal state for a case that hasn't started
    diagnosis. NOTE: run_diagnosis() deliberately does NOT call this at
    the start of a run (see that function's own reset-at-start
    docstring) - a fresh diagnosis run always starts with an empty
    buffer, matching existing, tested behavior
    (tests/test_summary_buffer_wiring.py's reset test). This function
    exists for genuine crash-recovery/inspection use (e.g. a future
    admin endpoint reading "what was the buffer's last known state
    before this case's process died"), not for seeding a new run.
    """
    from app.core.db import ExceptionCase
    case = db.get(ExceptionCase, case_id)
    buffer = SummaryBuffer(case_id=case_id)
    if case is not None and case.working_memory_summary:
        state = case.working_memory_summary
        buffer.running_summary = state.get("running_summary", "")
        buffer.verbatim_items = list(state.get("recent_items", []))
    return buffer
