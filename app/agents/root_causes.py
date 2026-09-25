"""
Root-cause contract between the Diagnosis Agent (LLM, free text) and the
Resolution-Policy Workflow (deterministic rules).

Two failure modes this module exists to close, both confirmed against a
real Groq model and the real pipeline (see tests/test_review_evidence.py):

1. Substring matching on free text. `"payment_issue" in causes_text`
   matched "payment_issue: none found, payment looks fine" and routed it
   to an automatic refund. Causes are now PARSED into a fixed category
   set; anything outside it is `unrecognized`, never silently dropped.

2. LLM claims with no tool evidence behind them. A real model turned an
   "unavailable" placeholder (no tracking number on the order) into
   "carrier_issue: carrier_unavailable". A category now only counts as
   CONFIRMED when the structured tool findings support it.
"""
from __future__ import annotations

from dataclasses import dataclass

PAYMENT = "payment_issue"
INVENTORY = "inventory_issue"
CARRIER = "carrier_issue"
NO_ANOMALY = "no_anomaly_detected"
INCOMPLETE = "diagnosis_incomplete"
TIMEOUT = "diagnosis_timeout"
UNRECOGNIZED = "unrecognized"

KNOWN_CATEGORIES = (PAYMENT, INVENTORY, CARRIER, NO_ANOMALY, INCOMPLETE, TIMEOUT)
# Categories that mean "the system does not actually know what happened" -
# never grounds for moving money.
INCONCLUSIVE_CATEGORIES = frozenset({INCOMPLETE, TIMEOUT, UNRECOGNIZED})

# Carrier statuses that are NOT a delivery fault. Anything else reported
# by the carrier gateway (lost, damaged, returned_to_sender, ...) is.
_CARRIER_OK_STATUSES = frozenset({"delivered", "in_transit", "label_created", "pre_transit",
                                   "out_for_delivery", "unknown"})


@dataclass(frozen=True)
class RootCause:
    category: str
    detail: str
    raw: str


def parse_root_causes(raw_causes) -> list[RootCause]:
    """Parses 'category: detail' strings. A non-string item, or one whose
    prefix isn't a known category, becomes UNRECOGNIZED."""
    parsed = []
    for item in raw_causes or []:
        if not isinstance(item, str):
            parsed.append(RootCause(UNRECOGNIZED, repr(item), repr(item)))
            continue
        head, sep, tail = item.partition(":")
        category = head.strip().lower()
        if sep and category in KNOWN_CATEGORIES:
            parsed.append(RootCause(category, tail.strip(), item))
        else:
            parsed.append(RootCause(UNRECOGNIZED, item.strip(), item))
    return parsed


def carrier_fault_confirmed(diagnosis_findings: dict | None) -> bool | None:
    """True/False when carrier findings exist; None when the caller supplied
    no findings at all (a direct caller of the rules, not the pipeline)."""
    if diagnosis_findings is None:
        return None
    carrier = diagnosis_findings.get("carrier")
    if not isinstance(carrier, dict) or carrier.get("unavailable") or carrier.get("error"):
        return False
    status = str(carrier.get("status") or "").lower()
    return bool(status) and status not in _CARRIER_OK_STATUSES
