"""
Resolution-Policy Workflow - architecture doc Part 1/5: a WORKFLOW, not an
open agent loop (per the autonomy calibration in Part 1 - the decision
step is bounded and auditable, unlike the Diagnosis Agent's open-ended
planning). Takes Diagnosis + Fraud/Risk + Inventory + Customer Context
outputs plus a RAG-retrieved policy citation, produces a structured
ResolutionDecision, runs it through Tier 1 (hard ceilings) and Tier 2
(structural + PII) guardrails, then routes to AUTO_EXECUTE, ESCALATE, or
BLOCKED.

Decision generation itself is deterministic rule-based logic (not an open
LLM call) - matching Part 1's explicit design that even the "LLM
reasoning" here operates within a fixed band, never with authority to
exceed Tier 1's ceilings regardless of what it concludes.
"""
from __future__ import annotations

from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy, ResolutionResult, RoutingOutcome
from app.guardrails.tier1_ceilings import check_tier1_ceilings
from app.guardrails.tier2_structural import run_tier2


def propose_resolution_decision(
    diagnosis_root_causes: list,
    inventory_result: dict,
    order_amount_usd: float,
    retrieved_policy_doc_id: str = None,
    retrieved_policy_version: str = None,
) -> ResolutionDecision:
    """Rule-based decision proposal - bounded logic, not open reasoning.
    This is the piece that would eventually call a reasoning-tier LLM
    (per architecture doc 8.10) to draft `reasoning` in more natural
    language, but the ACTION and AMOUNT selection logic stays rule-based
    regardless - the LLM (when wired in) drafts explanation text within
    constraints this function already determined, never picks the action
    or amount itself."""
    causes_text = " | ".join(diagnosis_root_causes)
    has_inventory_issue = "inventory_issue" in causes_text
    has_payment_issue = "payment_issue" in causes_text
    has_no_anomaly = "no_anomaly_detected" in causes_text
    any_shortfall = bool(inventory_result.get("any_shortfall"))

    cited = None
    if retrieved_policy_doc_id:
        cited = CitedPolicy(
            doc_id=retrieved_policy_doc_id,
            version=retrieved_policy_version or "unknown",
            clause_summary="Return window and refund processing terms",
        )

    if has_no_anomaly:
        return ResolutionDecision(
            action=ResolutionAction.DENY,
            amount_usd=0.0,
            confidence=0.95,
            reasoning="No anomaly was found across order, payment, inventory, or carrier checks - "
                      "the exception could not be substantiated against current system state.",
            cited_policy=None,
        )

    if has_inventory_issue and any_shortfall:
        return ResolutionDecision(
            action=ResolutionAction.PARTIAL_CREDIT,
            amount_usd=round(order_amount_usd * 0.5, 2),
            confidence=0.85,
            reasoning="Inventory shortfall confirmed for one or more line items - offering partial "
                      "credit for the unfulfillable portion per standard policy.",
            cited_policy=cited,
        )

    if has_payment_issue:
        return ResolutionDecision(
            action=ResolutionAction.REFUND,
            amount_usd=order_amount_usd,
            confidence=0.92,
            reasoning="Payment gateway confirms the transaction did not succeed - refunding the full "
                      "order amount since the customer was never successfully charged for a completed order.",
            cited_policy=cited,
        )

    return ResolutionDecision(
        action=ResolutionAction.REFUND,
        amount_usd=order_amount_usd,
        confidence=0.93,
        reasoning="Standard return request within the applicable policy's return window - approving "
                  "a full refund per the cited policy terms.",
        cited_policy=cited,
    )


def run_resolution_policy_workflow(
    diagnosis_root_causes: list,
    inventory_result: dict,
    order_amount_usd: float,
    fraud_flag_present: bool,
    auto_execute_confidence_threshold: float,
    auto_execute_value_ceiling_usd: float,
    retrieved_policy_doc_id: str = None,
    retrieved_policy_version: str = None,
    max_single_action_ceiling_usd: float = 1000.0,
    override_decision: ResolutionDecision = None,
    db=None,
    case_id: str = None,
) -> ResolutionResult:
    """Full workflow: propose -> Tier 1 -> Tier 2 -> route.

    override_decision lets tests inject an adversarial decision directly
    (e.g. confidence=1.0, elaborate reasoning, amount exceeding the hard
    ceiling) to prove Tier 1 blocks it regardless of how "confident" the
    proposal looks - bypassing propose_resolution_decision()'s own
    rule-based generation, which would never produce such a decision
    itself. This is exactly what the Phase 7 DoD's third test case needs.
    """
    decision = override_decision or propose_resolution_decision(
        diagnosis_root_causes, inventory_result, order_amount_usd,
        retrieved_policy_doc_id, retrieved_policy_version,
    )

    if db is not None and case_id is not None:
        from app.core.tracing import record_span
        record_span(db, trace_id=case_id, agent_or_tool_name="resolution_decision",
                    input_data={"diagnosis_root_causes": diagnosis_root_causes,
                                "order_amount_usd": order_amount_usd},
                    output_data=decision.model_dump(mode="json"),
                    metadata={"confidence": decision.confidence,
                              "cited_doc_id": decision.cited_policy.doc_id if decision.cited_policy else None,
                              "cited_version": decision.cited_policy.version if decision.cited_policy else None})

    tier2 = run_tier2(decision.model_dump(mode="json"))
    if not tier2.passed:
        return ResolutionResult(
            decision=decision, routing=RoutingOutcome.BLOCKED,
            routing_reasons=[f"Tier 2 (structural/PII) failed: {tier2.errors}"],
            tier1_passed=False, tier2_passed=False,
        )

    tier1 = check_tier1_ceilings(decision, auto_execute_value_ceiling_usd, max_single_action_ceiling_usd)
    if not tier1.passed:
        if db is not None:
            from app.core.alerting import send_alert
            send_alert(db, "tier1_block", {"case_id": case_id, "violations": tier1.violations,
                                            "decision_amount_usd": decision.amount_usd})
        return ResolutionResult(
            decision=decision, routing=RoutingOutcome.BLOCKED,
            routing_reasons=[f"Tier 1 (hard ceiling) failed: {tier1.violations}"],
            tier1_passed=False, tier2_passed=True,
        )

    reasons = []
    if fraud_flag_present:
        reasons.append("fraud flag present on this case - mandatory human review regardless of confidence or value")
        return ResolutionResult(decision=decision, routing=RoutingOutcome.ESCALATE,
                                 routing_reasons=reasons, tier1_passed=True, tier2_passed=True)

    from app.core.config import get_settings
    if not get_settings().auto_execution_enabled:
        reasons.append("auto-execution is globally disabled (rollback switch, Phase 16) — "
                        "routing to human review regardless of confidence or value")
        return ResolutionResult(decision=decision, routing=RoutingOutcome.ESCALATE,
                                 routing_reasons=reasons, tier1_passed=True, tier2_passed=True)

    if decision.confidence < auto_execute_confidence_threshold:
        reasons.append(f"confidence {decision.confidence} below auto-execute threshold {auto_execute_confidence_threshold}")

    if decision.amount_usd >= auto_execute_value_ceiling_usd:
        reasons.append(f"amount_usd {decision.amount_usd} at or above auto-execute ceiling {auto_execute_value_ceiling_usd}")

    if reasons:
        return ResolutionResult(decision=decision, routing=RoutingOutcome.ESCALATE,
                                 routing_reasons=reasons, tier1_passed=True, tier2_passed=True)

    return ResolutionResult(decision=decision, routing=RoutingOutcome.AUTO_EXECUTE,
                             routing_reasons=["confidence and value within auto-execute band, no fraud flag"],
                             tier1_passed=True, tier2_passed=True)
