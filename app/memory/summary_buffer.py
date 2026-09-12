"""
Short-term per-case working memory - architecture doc 8.4's Summary
Buffer: recent tool outputs kept verbatim (bounded), older context
collapsed into a running summary. This is deliberately a lightweight,
custom implementation, not a heavy framework - the architecture doc is
explicit that per-case working memory doesn't need Zep/Graphiti's
machinery, only the long-term/cross-case layer does (episodic.py).

The actual LLM summarization call is stubbed here (_summarize) since
Phase 6 hasn't wired up an LLM client yet - this module is complete and
testable on the mechanical bounded-buffer behavior now; Phase 6 plugs in
a real summarization call without changing this class's interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SummaryBuffer:
    case_id: str
    max_verbatim_items: int = 6
    verbatim_items: list[dict] = field(default_factory=list)
    running_summary: str = ""

    def add(self, item: dict) -> None:
        """Adds one tool-call result (or agent step) to the buffer. When
        the verbatim window overflows, the oldest item is folded into
        running_summary rather than dropped outright - this is the
        "recent verbatim + summarized older context" shape from 8.4."""
        self.verbatim_items.append(item)
        if len(self.verbatim_items) > self.max_verbatim_items:
            oldest = self.verbatim_items.pop(0)
            self.running_summary = self._summarize(self.running_summary, oldest)

    def _summarize(self, existing_summary: str, new_item: dict) -> str:
        """STUB for Phase 6: real implementation calls the generation-tier
        LLM (per architecture doc 8.10) to fold new_item into
        existing_summary. For now, a deterministic text fold keeps this
        class fully testable without a live LLM call."""
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


# In-memory registry keyed by case_id - Phase 6 will persist this to the
# ExceptionCase row (or a dedicated table) so it survives a process
# restart mid-case, matching the durable-state requirement from Layer 5.
# Kept in-memory here since Phase 5's scope is the buffer's BEHAVIOR, not
# its persistence, which is a Phase 6 orchestrator-state concern.
_buffers: dict[str, SummaryBuffer] = {}


def get_or_create_buffer(case_id: str) -> SummaryBuffer:
    if case_id not in _buffers:
        _buffers[case_id] = SummaryBuffer(case_id=case_id)
    return _buffers[case_id]


def reset_buffers() -> None:
    _buffers.clear()
