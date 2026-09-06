"""
Phase 7 DoD: "Three test cases — one clearly auto-executable, one clearly
requiring escalation (fraud flag present), one that Tier 1 blocks
outright regardless of model output — all route correctly."
"""
import pytest

from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy, RoutingOutcome
from app.agents.resolution_policy_workflow import run_resolution_policy_workflow

# ---------------------------------------------------------------------------
# THE three core Phase 7 DoD scenarios
# ---------------------------------------------------------------------------
def test_case_auto_executable():
    """Normal, low-value, high-confidence, no fraud flag, valid citation -> AUTO_EXECUTE."""
    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: transaction status is 'declined'"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=30.0,
        fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A",
        retrieved_policy_version="1",
    )
    assert result.routing == RoutingOutcome.AUTO_EXECUTE
    assert result.tier1_passed and result.tier2_passed
    assert result.decision.action == ResolutionAction.REFUND

def test_case_escalation_fraud_flag():
    """Otherwise-clean, high-confidence decision, but a fraud flag is
    present -> must ESCALATE, never AUTO_EXECUTE, regardless of how
    confident or low-value the underlying proposal is."""
    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: transaction status is 'declined'"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=10.0,          # deliberately tiny + high confidence...
        fraud_flag_present=True,         # ...but a fraud flag must override both
        auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A",
        retrieved_policy_version="1",
    )
    assert result.routing == RoutingOutcome.ESCALATE
    assert any("fraud flag" in r for r in result.routing_reasons)
    assert result.tier1_passed and result.tier2_passed   # the PROPOSAL itself was structurally fine

def test_case_tier1_blocks_regardless_of_confidence():
    """Tier 1's core promise: a decision with confidence=1.0 and an
    elaborate, persuasive-sounding reasoning string is STILL blocked
    outright if it violates the hard $ ceiling — the routing layer never
    even gets a chance to consider it."""
    adversarial_decision = ResolutionDecision(
        action=ResolutionAction.REFUND,
        amount_usd=5000.0,   # exceeds max_single_action_ceiling_usd (default 1000.0)
        confidence=1.0,       # maximally confident
        reasoning="I am extremely confident this is correct and fully justified by policy; "
                  "the customer's situation is exceptional and warrants immediate full resolution "
                  "without any need for further review or hesitation.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )
    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: transaction status is 'declined'"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=5000.0,
        fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0,
        override_decision=adversarial_decision,
    )
    assert result.routing == RoutingOutcome.BLOCKED
    assert result.tier1_passed is False
    assert any("5000" in r or "hard ceiling" in r for r in result.routing_reasons)

# ---------------------------------------------------------------------------
# Tier 1 unit tests (isolated from the full workflow)
# ---------------------------------------------------------------------------
def test_tier1_blocks_monetary_action_with_no_citation():
    from app.guardrails.tier1_ceilings import check_tier1_ceilings
    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=20.0, confidence=0.99,
        reasoning="Confident refund with absolutely no policy citation at all here.",
        cited_policy=None,
    )
    result = check_tier1_ceilings(decision, auto_execute_value_ceiling_usd=50.0)
    assert result.passed is False
    assert any("no cited_policy" in v for v in result.violations)

def test_tier1_passes_a_reasonable_decision():
    from app.guardrails.tier1_ceilings import check_tier1_ceilings
    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=20.0, confidence=0.9,
        reasoning="Standard refund within the normal policy-cited return window.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )
    result = check_tier1_ceilings(decision, auto_execute_value_ceiling_usd=50.0)
    assert result.passed is True
    assert result.violations == []

# ---------------------------------------------------------------------------
# Tier 2 unit tests
# ---------------------------------------------------------------------------
def test_tier2_rejects_malformed_decision_object():
    from app.guardrails.tier2_structural import validate_structure
    malformed = {"action": "not_a_real_action", "amount_usd": -5, "confidence": 2.0, "reasoning": "x"}
    result = validate_structure(malformed)
    assert result.passed is False
    assert len(result.errors) > 0

def test_tier2_detects_pii_in_reasoning():
    from app.guardrails.tier2_structural import run_tier2
    raw = {
        "action": "refund", "amount_usd": 20.0, "confidence": 0.9,
        "reasoning": "Refund approved; contact customer at jane.doe@example.com for confirmation.",
        "cited_policy": {"doc_id": "RET-POLICY-2025-A", "version": "1", "clause_summary": "return window"},
    }
    result = run_tier2(raw)
    assert result.passed is False
    assert any(f["type"] == "email" for f in result.pii_findings)

def test_tier2_passes_clean_decision():
    from app.guardrails.tier2_structural import run_tier2
    raw = {
        "action": "refund", "amount_usd": 20.0, "confidence": 0.9,
        "reasoning": "Standard refund within the normal policy-cited return window.",
        "cited_policy": {"doc_id": "RET-POLICY-2025-A", "version": "1", "clause_summary": "return window"},
    }
    result = run_tier2(raw)
    assert result.passed is True

# ---------------------------------------------------------------------------
# Tier 3 (async judge) unit tests
# ---------------------------------------------------------------------------
def test_tier3_flags_hallucinated_policy_citation():
    from app.guardrails.tier3_judge import run_tier3_async_sample
    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=20.0, confidence=0.9,
        reasoning="Refund approved per the cited policy's return window terms.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-9999-FAKE", version="1", clause_summary="return window"),
    )
    result = run_tier3_async_sample("case-1", decision, known_policy_doc_ids={"RET-POLICY-2025-A", "RET-POLICY-2026-A"})
    assert result["tier3_passed"] is False
    assert result["quality_score"] < 1.0

def test_tier3_passes_valid_citation():
    from app.guardrails.tier3_judge import run_tier3_async_sample
    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=20.0, confidence=0.9,
        reasoning="Refund approved per the cited policy's return window terms.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )
    result = run_tier3_async_sample("case-2", decision, known_policy_doc_ids={"RET-POLICY-2025-A", "RET-POLICY-2026-A"})
    assert result["tier3_passed"] is True
    assert result["quality_score"] == 1.0
