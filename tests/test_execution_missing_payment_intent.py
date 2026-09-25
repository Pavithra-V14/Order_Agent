"""
Test for a real production bug: execute_resolution() previously attempted
a REFUND/PARTIAL_CREDIT even when no payment_intent_id existed at all,
wasting 3 real retries against a call that could never succeed (Stripe
correctly rejects a refund request with no payment_intent or charge).
Found from an actual run: a real Groq LLM diagnosed
"payment_issue: missing payment information" from the ABSENCE of a
payment record on a delivery-focused case that never had one, and
resolution routed that straight to REFUND.
"""


def test_refund_with_no_payment_intent_id_fails_fast_not_after_three_wasted_retries():
    from app.agents.execution_agent import execute_resolution, ExecutionStatus
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy
    from app.core.db import SessionLocal, init_db

    init_db()
    db = SessionLocal()

    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=45.0, confidence=0.9,
        reasoning="A refund decision with no real payment to refund against.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )

    result = execute_resolution(
        db, case_id="case-no-payment-intent-test", decision=decision,
        payment_intent_id=None,  # the exact condition that caused the real bug
    )

    # A missing payment is permanent, not transient: FAILED sends the case
    # to a human instead of the retry queue, where it could never succeed.
    assert result.status == ExecutionStatus.FAILED
    assert result.attempts_made == 0, (
        "must fail immediately (0 attempts) rather than wasting 3 real retries against a gateway "
        "call that has no payment_intent_id to work with and can never succeed"
    )
    assert "no payment_intent_id" in result.error.lower()
    db.close()


def test_partial_credit_with_no_payment_intent_id_also_fails_fast():
    """The same precondition applies to PARTIAL_CREDIT, not just REFUND —
    both actions genuinely require a real payment to act against."""
    from app.agents.execution_agent import execute_resolution, ExecutionStatus
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy
    from app.core.db import SessionLocal, init_db

    init_db()
    db = SessionLocal()

    decision = ResolutionDecision(
        action=ResolutionAction.PARTIAL_CREDIT, amount_usd=20.0, confidence=0.85,
        reasoning="A partial credit decision with no real payment to credit against.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )

    result = execute_resolution(
        db, case_id="case-no-payment-intent-test-2", decision=decision, payment_intent_id=None,
    )

    # A missing payment is permanent, not transient: FAILED sends the case
    # to a human instead of the retry queue, where it could never succeed.
    assert result.status == ExecutionStatus.FAILED
    assert result.attempts_made == 0
    db.close()
