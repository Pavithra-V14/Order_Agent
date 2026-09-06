"""
Tier 1 guardrails - architecture doc 8.6: deterministic (sync, blocking),
~1-5ms, plain Python. The non-negotiable policy ceilings from Part 5 -
never LLM-decided, never bypassable by a confident-sounding model output.

This is the single most safety-critical file in the whole system: no
matter what decision.reasoning says, no matter how high
decision.confidence is, a decision that violates a Tier 1 ceiling is
BLOCKED. Tested explicitly with a maximally persuasive fake decision
(confidence=1.0, elaborate reasoning) that still gets blocked purely on
the numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.guardrails.schema import ResolutionDecision, ResolutionAction


@dataclass
class Tier1Result:
    passed: bool
    violations: list = field(default_factory=list)


def check_tier1_ceilings(
    decision: ResolutionDecision,
    auto_execute_value_ceiling_usd: float,
    max_single_action_ceiling_usd: float = 1000.0,
) -> Tier1Result:
    """Checks a proposed decision against hard, numeric-only ceilings.
    IMPORTANT: this function reads NOTHING from decision.reasoning or
    decision.confidence to decide pass/fail — those fields are for audit
    and for the ESCALATE-vs-AUTO_EXECUTE routing decision (a separate,
    softer judgment made downstream in resolution_policy_workflow.py),
    never for whether a Tier 1 ceiling applies. A confidence of 1.0 does
    not, and must not, unlock a bigger ceiling.

    Deliberately scoped to absolute, non-negotiable numeric/structural
    limits only — mandatory-escalation triggers that depend on case
    CONTEXT rather than the decision object itself (e.g. "a fraud flag is
    present on this case") belong at the ROUTING layer
    (resolution_policy_workflow.py), not here. Putting them here would
    make "blocked outright" and "escalated for review" indistinguishable,
    which loses real information: a fraud-flagged case still gets a human
    who reviews a real proposed resolution, while a Tier 1 block means the
    proposal itself was structurally/numerically invalid and must be
    regenerated, not just reviewed.
    """
    violations = []

    if decision.amount_usd > max_single_action_ceiling_usd:
        violations.append(
            f"amount_usd={decision.amount_usd} exceeds the absolute hard ceiling "
            f"of {max_single_action_ceiling_usd} — no single action may exceed this "
            f"regardless of confidence or auto-execute eligibility."
        )

    if decision.action in (ResolutionAction.REFUND, ResolutionAction.PARTIAL_CREDIT) and decision.cited_policy is None:
        violations.append(
            "A monetary decision (refund/partial_credit) with no cited_policy is "
            "blocked outright — every dollar amount must be traceable to a specific "
            "policy clause, not just a model's stated confidence."
        )

    return Tier1Result(passed=len(violations) == 0, violations=violations)
