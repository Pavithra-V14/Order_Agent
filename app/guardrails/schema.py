"""
Resolution decision schema - architecture doc Part 5 / 8.6. This is the
object the Resolution-Policy Workflow produces, that Tier 1/2 guardrails
validate, and that the (Phase 8) Execution Agent will consume. Defining
it once here means every layer validates against the SAME contract.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class ResolutionAction(str, Enum):
    REFUND = "refund"
    RESHIP = "reship"
    PARTIAL_CREDIT = "partial_credit"
    DENY = "deny"


class CitedPolicy(BaseModel):
    doc_id: str
    version: str
    clause_summary: str = Field(description="Short summary of the specific clause relied on, not the full text")


class ResolutionDecision(BaseModel):
    """The structured decision object - validated by Tier 2 guardrails
    (schema) before anything downstream ever sees it, and checked against
    Tier 1's hard ceilings (non-negotiable, non-LLM) regardless of what
    values appear here."""
    action: ResolutionAction
    amount_usd: float = Field(ge=0, description="0 for reship/deny actions")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=10, description="Why this action was chosen, in plain language")
    cited_policy: CitedPolicy | None = None
    # Set by the rules when the evidence can't support an automated
    # action (inconclusive diagnosis, LLM claim contradicted by tool
    # data). Routing always ESCALATES such a decision, at any threshold.
    requires_human_review: bool = False


class RoutingOutcome(str, Enum):
    AUTO_EXECUTE = "auto_execute"
    ESCALATE = "escalate"
    BLOCKED = "blocked"   # Tier 1 hard-ceiling rejection - never reaches a human OR auto-executes as proposed


class ResolutionResult(BaseModel):
    decision: ResolutionDecision
    routing: RoutingOutcome
    routing_reasons: list = Field(default_factory=list)
    tier1_passed: bool
    tier2_passed: bool
    # Few-shot retrieval (app/agents/learning_loop.py) — found fully
    # implemented but never actually called from anywhere during a
    # direct audit. Surfaced here as CONTEXT for a human reviewer
    # during escalation (real value even without an LLM in the
    # decision loop: "here's how similar past cases were resolved"
    # directly serves the stated goal of improving consistency), not
    # used to alter the decision itself — this stays additive, never
    # silently changing what gets proposed.
    similar_past_cases: list = Field(default_factory=list)
