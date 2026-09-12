"""
Metadata schema per architecture doc 8.2.3 — this is the piece that makes
temporal correctness (the return-policy-changed edge case) structural
rather than hoped-for.

Rather than hardcoding each PDF's effective dates in a side config, this
parses them directly out of the document's own header line (every policy
PDF in this project's corpus states "Document ID: X | Version: Y |
Effective: ... | ..." on page 1 — see scripts/generate_policy_pdfs.py) —
so if a policy document doesn't declare its own version metadata clearly,
that's a data-quality problem to fix at the source, not something to
paper over with an out-of-band mapping file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date


@dataclass
class PolicyMetadata:
    doc_id: str
    version: str
    effective_start: date
    effective_end: date | None       # None means "currently active"
    superseded_by: str | None
    supersedes: str | None
    doc_type: str                     # "return_policy" | "fraud_policy" | ...
    product_category: list[str] = field(default_factory=lambda: ["all"])
    channel: list[str] = field(default_factory=lambda: ["all"])
    return_window_days_by_category: dict[str, int] | None = None


_HEADER_RE = re.compile(
    r"Document ID:\s*(?P<doc_id>[A-Z0-9\-]+).*?"
    r"Version:\s*(?P<version>\d+).*?"
    r"Effective:\s*(?P<eff_start>\d{4}-\d{2}-\d{2})\s*(?:to|-)\s*(?P<eff_end>\d{4}-\d{2}-\d{2}|present)",
    re.IGNORECASE | re.DOTALL,
)
_SUPERSEDES_RE = re.compile(r"Supersedes:\s*([A-Z0-9\-]+)", re.IGNORECASE)
_SUPERSEDED_BY_RE = re.compile(r"Superseded by:\s*([A-Z0-9\-]+)", re.IGNORECASE)
_RETURN_WINDOWS_RE = re.compile(r"Return Windows \(days\):\s*(.+?)(?:\n|$)", re.IGNORECASE)


def _parse_return_windows(header_text: str) -> dict[str, int] | None:
    """Parses the structured 'Return Windows (days): apparel=180,
    electronics=30, ...' line every return_policy PDF states on page 1
    (see scripts/generate_policy_pdfs.py) — the machine-readable
    counterpart to the human-readable category table on the same page.

    This exists specifically to make return-window decisions
    (does this order qualify for an automatic refund vs. denial)
    programmatically checkable, rather than requiring an LLM to
    interpret prose text or a fragile regex against a formatted table —
    the return window genuinely differs BY PRODUCT CATEGORY within the
    same policy (apparel=180 days, electronics=30 days in the same
    document), so a single per-policy number would be wrong.

    Returns None (not an error) if this line is absent — not every
    policy type has return windows (fraud policies don't), and this
    field is optional precisely because of that.
    """
    match = _RETURN_WINDOWS_RE.search(header_text)
    if not match:
        return None
    windows = {}
    for pair in match.group(1).split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        category, days_str = pair.split("=", 1)
        try:
            windows[category.strip().lower()] = int(days_str.strip())
        except ValueError:
            continue
    return windows or None


def _infer_doc_type(doc_id: str) -> str:
    if doc_id.startswith("RET-"):
        return "return_policy"
    if doc_id.startswith("FRAUD-"):
        return "fraud_policy"
    return "general_policy"


def parse_policy_metadata(header_text: str) -> PolicyMetadata:
    """Parses the version-header block that appears on page 1 of every
    policy PDF in this corpus. Raises a clear error rather than guessing
    if the expected fields aren't present — silently defaulting temporal
    fields is exactly the class of bug the whole versioning design exists
    to prevent."""
    match = _HEADER_RE.search(header_text.replace("\n", " "))
    if not match:
        raise ValueError(
            "Could not parse required version-header fields (Document ID / "
            "Version / Effective dates) from this policy document's page 1 text. "
            "Refusing to ingest with guessed/default temporal metadata — fix "
            "the source document's header instead. Extracted text was:\n"
            f"{header_text[:300]}"
        )

    doc_id = match.group("doc_id")
    eff_end_raw = match.group("eff_end")
    eff_end = None if eff_end_raw.lower() == "present" else date.fromisoformat(eff_end_raw)

    supersedes_match = _SUPERSEDES_RE.search(header_text)
    superseded_by_match = _SUPERSEDED_BY_RE.search(header_text)

    return PolicyMetadata(
        doc_id=doc_id,
        version=match.group("version"),
        effective_start=date.fromisoformat(match.group("eff_start")),
        effective_end=eff_end,
        supersedes=supersedes_match.group(1) if supersedes_match else None,
        superseded_by=superseded_by_match.group(1) if superseded_by_match else None,
        doc_type=_infer_doc_type(doc_id),
        return_window_days_by_category=_parse_return_windows(header_text),
    )
