"""
Captions image/chart elements so they become searchable text (architecture
doc 8.2.1). Production path: a vision-capable LLM call (Gemini 2.0 Flash
supports vision on its free tier, per 8.10) describing the chart/image.
Not callable from this sandbox (no network access to generativelanguage.
googleapis.com from the bash tool's allowed domains) — so this ships with
a heuristic fallback that extracts real signal from the chart's source
data when available (this project generates its own charts in Phase 2, so
we actually have the underlying numbers) and falls back to a generic
placeholder caption otherwise, clearly logged as such so it's never
mistaken for a real vision-model caption.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def caption_chart_heuristic(image_path: str, known_title: str | None = None,
                             known_categories: list[str] | None = None,
                             known_values: list[float] | None = None) -> str:
    """Used when the chart's underlying data is already known (e.g., this
    project's own Phase 2 generation script kept the source numbers) —
    produces a genuinely accurate caption without needing a vision model call
    at all. This is the right choice when available, not just a fallback:
    a caption derived from ground-truth data is more reliable than a vision
    model's read of a rendered chart image."""
    if known_categories and known_values:
        pairs = ", ".join(f"{c}: {v}" for c, v in zip(known_categories, known_values))
        title = known_title or "Chart"
        return f"{title}. Data shown — {pairs}."
    return caption_chart_via_vision_llm(image_path)


def caption_chart_via_vision_llm(image_path: str) -> str:
    """Production path — calls a vision-capable free-tier LLM (Gemini 2.0
    Flash, per architecture doc 8.10) to describe the chart. Not reachable
    from this sandbox's network allowlist, so it raises clearly rather than
    silently returning a fake caption — callers should prefer
    caption_chart_heuristic() with known data when it's available (as it is
    for every chart in this project's own policy corpus, Phase 2)."""
    raise RuntimeError(
        "caption_chart_via_vision_llm requires network access to a vision-capable "
        "LLM API (e.g. Gemini 2.0 Flash) not available in this sandbox. "
        "Use caption_chart_heuristic() with known chart data instead, or run this "
        "in an environment with real API access and a configured GOOGLE_API_KEY."
    )
