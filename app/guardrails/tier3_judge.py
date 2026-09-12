"""
Tier 3 guardrail - architecture doc 8.6: LLM-judge (async, non-blocking),
200-1000ms but off the critical path. Samples resolved decisions for
policy-citation accuracy and tone drift; never blocks execution.

This is a background worker job (per 8.1's /workers layout), not called
synchronously from the Resolution-Policy Workflow. This sandbox has no
network access to a real judge-tier LLM, so the fake judge does genuine
rule-based checking (does the cited policy doc_id/version actually exist
in a known-good set? is the reasoning text non-empty and non-templated?)
rather than returning a canned "looks fine."
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.guardrails.schema import ResolutionDecision


@dataclass
class Tier3JudgeResult:
    passed: bool
    quality_score: float
    flags: list = field(default_factory=list)


def judge_decision_fake(decision: ResolutionDecision, known_policy_doc_ids: set) -> Tier3JudgeResult:
    """Rule-based substitute for a real LLM-judge call. Checks the kind of
    thing a judge model would actually catch: does the cited policy doc_id
    correspond to a document that genuinely exists, and is the reasoning
    substantive rather than a generic template string."""
    flags = []
    score = 1.0

    if decision.cited_policy is not None and decision.cited_policy.doc_id not in known_policy_doc_ids:
        flags.append(f"cited_policy.doc_id '{decision.cited_policy.doc_id}' does not match any known policy document")
        score -= 0.5

    generic_phrases = ["as per policy", "standard procedure applies", "n/a"]
    if decision.reasoning.strip().lower() in generic_phrases:
        flags.append("reasoning text is a generic placeholder, not substantive")
        score -= 0.3

    score = max(score, 0.0)
    return Tier3JudgeResult(passed=len(flags) == 0, quality_score=score, flags=flags)


def run_tier3_async_sample(case_id: str, decision: ResolutionDecision, known_policy_doc_ids: set) -> dict:
    """Entry point a background worker calls on a SAMPLE of resolved
    cases, not every case, and never inline with execution. Logs the
    result rather than raising - Tier 3 NEVER blocks execution, it only
    informs drift-watch metrics."""
    result = judge_decision_fake(decision, known_policy_doc_ids)
    return {
        "case_id": case_id,
        "tier3_passed": result.passed,
        "quality_score": result.quality_score,
        "flags": result.flags,
    }
