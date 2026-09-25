"""
Phase 7 DoD: "Three test cases — one clearly auto-executable, one clearly
requiring escalation (fraud flag present), one that Tier 1 blocks
outright regardless of model output — all route correctly."
"""
import pytest

from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy, RoutingOutcome
from app.agents.resolution_policy_workflow import run_resolution_policy_workflow, propose_resolution_decision

# ---------------------------------------------------------------------------
# THE three core Phase 7 DoD scenarios
# ---------------------------------------------------------------------------
def test_case_auto_executable():
    """Normal, low-value, no fraud flag, valid citation, a plain return
    inside the policy window on a succeeded payment -> AUTO_EXECUTE.
    (A "payment_issue" on a SUCCEEDED charge used to auto-refund here; the
    gateway contradicts that claim, so it now escalates - see
    tests/test_review_evidence.py::test_R3_*.)"""
    from datetime import date, timedelta
    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["no_anomaly_detected: all checked systems report normal state"],
        purchase_date=(date.today() - timedelta(days=10)).isoformat(), product_category="apparel",
        return_window_days_by_category={"apparel": 180, "all": 180},
        inventory_result={"any_shortfall": False},
        order_amount_usd=30.0,
        fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A",
        retrieved_policy_version="1",
        payment_status="succeeded",
    )
    assert result.routing == RoutingOutcome.AUTO_EXECUTE
    assert result.tier1_passed and result.tier2_passed
    assert result.decision.action == ResolutionAction.REFUND


def test_payment_issue_with_no_successful_charge_escalates_not_refunds():
    """THE regression test for a real bug found by actually running
    against real Stripe: 'This PaymentIntent does not have a successful
    charge to refund.' A payment that was NEVER successfully charged
    (declined, requires_payment_method, canceled, etc.) has nothing to
    refund — the original decision logic's own reasoning text even
    admitted this ('refunding... since the customer was never
    successfully charged') while still proposing a refund anyway, an
    internally contradictory decision that only a real payment gateway
    enforcing the real-world rule ever caught."""
    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: payment method required or payment failed"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=45.0,
        fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A",
        retrieved_policy_version="1",
        payment_status="requires_payment_method",  # a real Stripe status meaning never charged
    )
    assert result.decision.action == ResolutionAction.DENY
    assert result.decision.amount_usd == 0.0
    assert result.routing == RoutingOutcome.ESCALATE, (
        "must escalate for human review, not silently auto-deny either — this needs a person to "
        "decide the actual next step (new payment method, cancel, etc.), not an automated non-decision"
    )


def test_carrier_issue_with_stock_available_proposes_reship_not_refund():
    """THE regression test for a real gap found directly: RESHIP was a
    fully-supported execution action (a real carrier label actually
    gets generated) but this decision logic never selected it — a lost/
    damaged-in-transit package always got refunded instead, even when
    reshipping the same item was the more appropriate real response."""
    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["carrier_issue: tracking status is 'lost'"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=45.0,
        fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A",
        retrieved_policy_version="1",
    )
    assert result.decision.action == ResolutionAction.RESHIP
    assert result.decision.amount_usd == 0.0
    assert result.routing == RoutingOutcome.AUTO_EXECUTE


def test_carrier_issue_with_shortfall_falls_back_to_refund():
    """The sensible fallback: you can't reship an item you don't have in
    stock — a carrier issue with an inventory shortfall must still
    refund, not attempt a reship that would immediately fail."""
    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["carrier_issue: tracking status is 'lost'"],
        inventory_result={"any_shortfall": True},
        order_amount_usd=45.0,
        fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A",
        retrieved_policy_version="1",
    )
    assert result.decision.action == ResolutionAction.REFUND
    assert result.decision.amount_usd == 45.0


def test_no_anomaly_within_return_window_approves_refund_not_denial():
    """THE regression test for a real, meaningful gap found by actually
    running this system: "no_anomaly_detected" (the MOST COMMON real
    case — a customer returning an item they simply don't want, nothing
    operationally broken) was unconditionally DENIED, which is backwards
    from how return policies actually work. This proves a plain return
    within the cited policy's window now correctly gets approved."""
    from datetime import date, timedelta
    purchase_date = (date.today() - timedelta(days=10)).isoformat()

    decision = propose_resolution_decision(
        diagnosis_root_causes=["no_anomaly_detected: all checked systems report normal state"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=45.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A",
        retrieved_policy_version="1",
        purchase_date=purchase_date,
        product_category="apparel",
        return_window_days_by_category={"apparel": 180, "all": 180},
    )
    assert decision.action == ResolutionAction.REFUND
    assert decision.amount_usd == 45.0
    assert decision.cited_policy is not None


def test_no_anomaly_outside_return_window_denies_with_clear_reason():
    """The other half: a plain return request made LONG after the
    return window closed must still be denied - this fix approves
    returns within policy, it doesn't remove the window check entirely."""
    from datetime import date, timedelta
    purchase_date = (date.today() - timedelta(days=200)).isoformat()

    decision = propose_resolution_decision(
        diagnosis_root_causes=["no_anomaly_detected: all checked systems report normal state"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=45.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A",
        retrieved_policy_version="1",
        purchase_date=purchase_date,
        product_category="apparel",
        return_window_days_by_category={"apparel": 180, "all": 180},
    )
    assert decision.action == ResolutionAction.DENY
    assert "200" in decision.reasoning
    assert "180" in decision.reasoning


def test_no_anomaly_uses_category_specific_window_not_generic_all():
    """The window genuinely differs by category within the SAME policy
    (electronics=30 days vs apparel=180 days) - this proves the
    category-specific lookup is actually used, not just the 'all'
    catch-all regardless of what category the order actually is."""
    from datetime import date, timedelta
    purchase_date = (date.today() - timedelta(days=45)).isoformat()  # within apparel's window, outside electronics'

    decision_electronics = propose_resolution_decision(
        diagnosis_root_causes=["no_anomaly_detected: all checked systems report normal state"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=200.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A",
        retrieved_policy_version="1",
        purchase_date=purchase_date,
        product_category="electronics",
        return_window_days_by_category={"apparel": 180, "electronics": 30, "all": 180},
    )
    assert decision_electronics.action == ResolutionAction.DENY, (
        "electronics has a 30-day window - 45 days ago must be denied, even though "
        "the generic 'all' window (180) would have approved it"
    )

    decision_apparel = propose_resolution_decision(
        diagnosis_root_causes=["no_anomaly_detected: all checked systems report normal state"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=45.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A",
        retrieved_policy_version="1",
        purchase_date=purchase_date,
        product_category="apparel",
        return_window_days_by_category={"apparel": 180, "electronics": 30, "all": 180},
    )
    assert decision_apparel.action == ResolutionAction.REFUND


def test_no_anomaly_without_window_data_denies_with_low_confidence_not_a_guess():
    """When purchase_date/category/window data genuinely isn't available
    (e.g. a caller invoking propose_resolution_decision() directly
    without them), this must NOT silently guess either way - a
    low-confidence denial that would correctly route to ESCALATE rather
    than auto-execute, since the confidence sits below any reasonable
    auto-execute threshold."""
    decision = propose_resolution_decision(
        diagnosis_root_causes=["no_anomaly_detected: all checked systems report normal state"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=45.0,
    )
    assert decision.action == ResolutionAction.DENY
    assert decision.confidence < 0.90, "must be low-confidence enough to force ESCALATE, not auto-execute a guess"

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
