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

from datetime import date, datetime

from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy, ResolutionResult, RoutingOutcome
from app.guardrails.tier1_ceilings import check_tier1_ceilings
from app.guardrails.tier2_structural import run_tier2


def _resolve_window_days(return_window_days_by_category: dict | None, product_category: str | None) -> int | None:
    """Looks up the applicable return window for a specific product
    category, falling back to the policy's 'all' catch-all entry if that
    exact category isn't listed separately — matches how the real policy
    documents are structured (most categories get their own window, but
    "all" covers anything not explicitly listed)."""
    if not return_window_days_by_category:
        return None
    if product_category:
        category_key = product_category.strip().lower()
        if category_key in return_window_days_by_category:
            return return_window_days_by_category[category_key]
    return return_window_days_by_category.get("all")


def _days_since(purchase_date: str | date | None) -> int | None:
    """Computes whole days between a purchase date and today. Accepts
    either a date object or an ISO-format string (get_order() returns
    the latter) - returns None rather than raising if the input is
    missing or malformed, since an unparseable date should route to the
    "genuinely missing data" fallback in propose_resolution_decision(),
    not crash the whole resolution workflow."""
    if purchase_date is None:
        return None
    try:
        if isinstance(purchase_date, str):
            purchase_date = date.fromisoformat(purchase_date.split("T")[0])
        elif isinstance(purchase_date, datetime):
            purchase_date = purchase_date.date()
        return (date.today() - purchase_date).days
    except (ValueError, TypeError):
        return None


def propose_resolution_decision(
    diagnosis_root_causes: list,
    inventory_result: dict,
    order_amount_usd: float,
    retrieved_policy_doc_id: str = None,
    retrieved_policy_version: str = None,
    purchase_date: str = None,
    product_category: str = None,
    return_window_days_by_category: dict = None,
    payment_status: str = None,
) -> ResolutionDecision:
    """Rule-based decision proposal - bounded logic, not open reasoning.
    This is the piece that would eventually call a reasoning-tier LLM
    (per architecture doc 8.10) to draft `reasoning` in more natural
    language, but the ACTION and AMOUNT selection logic stays rule-based
    regardless - the LLM (when wired in) drafts explanation text within
    constraints this function already determined, never picks the action
    or amount itself.

    purchase_date/product_category/return_window_days_by_category:
    added to fix a real, previously-undiscovered gap found by actually
    running this system end to end: "no_anomaly_detected" (the MOST
    COMMON real-world case — a customer returning an item they simply
    don't want, with nothing operationally broken about the order) was
    unconditionally treated as grounds for DENIAL. That's backwards from
    how return policies actually work: most legitimate returns have no
    system fault to find and should be approved within the return
    window, not denied for lacking a diagnosable problem. This now
    checks the CITED policy's actual, category-specific return window
    (return_window_days_by_category — see app/rag/metadata.py, parsed
    from a structured line every real policy PDF states, not guessed)
    against how long ago the order was purchased, and approves a refund
    if within it — falling back to denial only when genuinely outside
    the window, or when the window data isn't available at all (a
    missing citation is still refused rather than silently approved).
    """
    causes_text = " | ".join(diagnosis_root_causes)
    has_inventory_issue = "inventory_issue" in causes_text
    has_payment_issue = "payment_issue" in causes_text
    has_carrier_issue = "carrier_issue" in causes_text
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
        window_days = _resolve_window_days(return_window_days_by_category, product_category)
        days_since_purchase = _days_since(purchase_date)

        if window_days is not None and days_since_purchase is not None:
            if days_since_purchase <= window_days:
                return ResolutionDecision(
                    action=ResolutionAction.REFUND,
                    amount_usd=order_amount_usd,
                    confidence=0.93,
                    reasoning=f"No operational anomaly was found (payment, inventory, and carrier all "
                              f"report normal state) — this is a standard return request, not a system "
                              f"fault. Order was purchased {days_since_purchase} day(s) ago, within the "
                              f"cited policy's {window_days}-day window for this product category, so "
                              f"approving a full refund per policy terms.",
                    cited_policy=cited,
                )
            return ResolutionDecision(
                action=ResolutionAction.DENY,
                amount_usd=0.0,
                confidence=0.91,
                reasoning=f"No operational anomaly was found, and this order was purchased "
                          f"{days_since_purchase} day(s) ago — outside the cited policy's "
                          f"{window_days}-day return window for this product category. Denying per "
                          f"policy terms, not for lack of a diagnosable system fault.",
                cited_policy=cited,
            )

        # Genuinely missing the data needed to check the window at all
        # (no citation, or no purchase_date/category supplied) — refuse
        # rather than silently approve OR silently deny without a real
        # basis either way. This should be rare in real usage (the full
        # pipeline always supplies these), but a caller invoking this
        # function directly without them gets an honest, low-confidence
        # denial that clearly explains why, not a guess.
        return ResolutionDecision(
            action=ResolutionAction.DENY,
            amount_usd=0.0,
            confidence=0.60,
            reasoning="No operational anomaly was found, but this order's purchase date, product "
                      "category, or the cited policy's return-window data was not available to check "
                      "whether it falls within the applicable return window — denying pending a proper "
                      "policy-window check rather than guessing either way.",
            cited_policy=cited,
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

    if has_carrier_issue:
        # A genuine gap found and fixed: RESHIP was a fully-supported
        # execution action (a real carrier label gets generated) but was
        # NEVER actually selected by this decision logic — a lost/
        # damaged-in-transit package would always be refunded instead,
        # even when reshipping the same item (with stock on hand) is the
        # more appropriate real-world response. Only reship when
        # inventory is actually available; fall back to refund if it
        # isn't, since you can't reship what you don't have.
        if not any_shortfall:
            return ResolutionDecision(
                action=ResolutionAction.RESHIP,
                amount_usd=0.0,
                confidence=0.90,
                reasoning="Carrier tracking confirms a delivery exception (lost/damaged/returned to "
                          "sender) and sufficient stock is available — reshipping the same item rather "
                          "than refunding, since the customer's actual intent was to receive the product.",
                cited_policy=cited,
            )
        return ResolutionDecision(
            action=ResolutionAction.REFUND,
            amount_usd=order_amount_usd,
            confidence=0.88,
            reasoning="Carrier tracking confirms a delivery exception, but insufficient stock is "
                      "available to reship the same item — refunding the full order amount instead.",
            cited_policy=cited,
        )

    if has_payment_issue:
        # Only a payment that was genuinely, successfully charged is
        # refundable at all — Stripe's real Refund API confirms this
        # directly ("This PaymentIntent does not have a successful
        # charge to refund"). The ORIGINAL version of this branch
        # unconditionally proposed REFUND for ANY payment_issue root
        # cause, including its own literal reasoning text admitting
        # "the customer was never successfully charged" — an internally
        # contradictory decision that only surfaced as broken once run
        # against a real payment gateway that actually enforces the
        # real-world rule a fake gateway never checked. Real Stripe
        # PaymentIntent statuses that mean "never actually charged":
        # requires_payment_method, requires_confirmation, requires_action,
        # canceled, processing (not yet settled), requires_capture
        # (authorized but not captured — not confirmed yet either).
        # "succeeded" is the only status that means money was genuinely
        # taken and is refundable.
        if payment_status == "succeeded":
            return ResolutionDecision(
                action=ResolutionAction.REFUND,
                amount_usd=order_amount_usd,
                confidence=0.92,
                reasoning="Payment gateway confirms this payment DID succeed, but a payment_issue "
                          "was still flagged (e.g. an incorrect charge amount or a separate billing "
                          "dispute) - refunding the full order amount since the customer was "
                          "genuinely, successfully charged.",
                cited_policy=cited,
            )
        return ResolutionDecision(
            action=ResolutionAction.DENY,
            amount_usd=0.0,
            confidence=0.40,  # deliberately low - this needs a human, not an automated denial either
            reasoning=f"Payment gateway confirms this payment was NEVER successfully charged "
                      f"(status={payment_status!r}) - there is nothing to refund. This needs human "
                      f"review to determine the right next step (contact the customer for a new "
                      f"payment method, cancel the order, or something else) rather than either an "
                      f"automated refund of money that was never taken or a silent denial.",
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
    purchase_date: str = None,
    product_category: str = None,
    return_window_days_by_category: dict = None,
    payment_status: str = None,
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
        purchase_date, product_category, return_window_days_by_category,
        payment_status,
    )

    similar_past_cases = []
    if db is not None:
        # Few-shot retrieval, surfaced as CONTEXT only — never allowed
        # to alter `decision` itself, which was already computed above.
        # Wrapped so a retrieval failure (e.g. no past cases yet, or a
        # genuine error) never blocks a real resolution from completing.
        try:
            from app.agents.learning_loop import retrieve_similar_past_resolutions
            query_summary = f"root_causes={diagnosis_root_causes}, fraud_flag={fraud_flag_present}"
            similar = retrieve_similar_past_resolutions(db, query_summary, k=3)
            similar_past_cases = [
                {"case_id": s.case_id, "case_feature_summary": s.case_feature_summary,
                 "human_final_resolution": s.human_final_resolution, "similarity": s.similarity}
                for s in similar
            ]
        except Exception:
            pass

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
            tier1_passed=False, tier2_passed=False, similar_past_cases=similar_past_cases,
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
            tier1_passed=False, tier2_passed=True, similar_past_cases=similar_past_cases,
        )

    reasons = []
    if fraud_flag_present:
        reasons.append("fraud flag present on this case - mandatory human review regardless of confidence or value")
        return ResolutionResult(decision=decision, routing=RoutingOutcome.ESCALATE,
                                 routing_reasons=reasons, tier1_passed=True, tier2_passed=True,
                                 similar_past_cases=similar_past_cases)

    from app.core.config import get_settings
    if not get_settings().auto_execution_enabled:
        reasons.append("auto-execution is globally disabled (rollback switch, Phase 16) — "
                        "routing to human review regardless of confidence or value")
        return ResolutionResult(decision=decision, routing=RoutingOutcome.ESCALATE,
                                 routing_reasons=reasons, tier1_passed=True, tier2_passed=True,
                                 similar_past_cases=similar_past_cases)

    if decision.confidence < auto_execute_confidence_threshold:
        reasons.append(f"confidence {decision.confidence} below auto-execute threshold {auto_execute_confidence_threshold}")

    if decision.amount_usd >= auto_execute_value_ceiling_usd:
        reasons.append(f"amount_usd {decision.amount_usd} at or above auto-execute ceiling {auto_execute_value_ceiling_usd}")

    if reasons:
        return ResolutionResult(decision=decision, routing=RoutingOutcome.ESCALATE,
                                 routing_reasons=reasons, tier1_passed=True, tier2_passed=True,
                                 similar_past_cases=similar_past_cases)

    return ResolutionResult(decision=decision, routing=RoutingOutcome.AUTO_EXECUTE,
                             routing_reasons=["confidence and value within auto-execute band, no fraud flag"],
                             tier1_passed=True, tier2_passed=True, similar_past_cases=similar_past_cases)
