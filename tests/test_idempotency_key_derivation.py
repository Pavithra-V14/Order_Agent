"""
Tests for app/agents/execution_agent.py's _derive_idempotency_key() -
the regression test for a real production bug found from an actual
demo run.
"""


def test_same_case_id_with_a_genuinely_different_payment_gets_a_different_idempotency_key():
    """THE regression test for a real production bug: reusing the SAME
    case_id across two genuinely DIFFERENT underlying requests (a
    different payment_intent_id each time - exactly what happens when a
    demo script creates a fresh Stripe test payment each run but reuses
    the same case_id) used to derive the IDENTICAL idempotency key for
    both, since the key was based on case_id+action alone. Stripe
    correctly rejected the second request with "Keys for idempotent
    requests can only be used with the same parameters they were first
    used with" - a real, reproducible failure, not a fluke. The key
    must now differ when the underlying payment_intent_id genuinely
    differs, even for the same case_id and action.
    """
    from app.agents.execution_agent import _derive_idempotency_key
    from app.guardrails.schema import ResolutionAction

    key1 = _derive_idempotency_key("case-return-demo", ResolutionAction.REFUND,
                                    payment_intent_id="pi_first_run", amount_usd=45.0)
    key2 = _derive_idempotency_key("case-return-demo", ResolutionAction.REFUND,
                                    payment_intent_id="pi_second_run", amount_usd=45.0)

    assert key1 != key2, (
        "the same case_id with a genuinely different payment_intent_id must produce a different "
        "idempotency key - reusing the identical key for two different underlying requests is "
        "exactly what caused Stripe's real 'idempotent requests' rejection"
    )


def test_same_case_id_same_payment_and_amount_produces_the_same_key_every_time():
    """The other half: a TRUE retry (identical case, action,
    payment_intent_id, and amount) must still deterministically produce
    the SAME key every time - this fix must not break real idempotent
    retries of the identical operation."""
    from app.agents.execution_agent import _derive_idempotency_key
    from app.guardrails.schema import ResolutionAction

    key1 = _derive_idempotency_key("case-x", ResolutionAction.REFUND,
                                    payment_intent_id="pi_same", amount_usd=45.0)
    key2 = _derive_idempotency_key("case-x", ResolutionAction.REFUND,
                                    payment_intent_id="pi_same", amount_usd=45.0)
    assert key1 == key2


def test_different_amount_alone_also_produces_a_different_key():
    """A partial refund followed by a different-amount partial refund
    for the same case must not collide either - amount is part of what
    makes a request genuinely different, not just the payment_intent_id."""
    from app.agents.execution_agent import _derive_idempotency_key
    from app.guardrails.schema import ResolutionAction

    key1 = _derive_idempotency_key("case-y", ResolutionAction.PARTIAL_CREDIT,
                                    payment_intent_id="pi_same", amount_usd=20.0)
    key2 = _derive_idempotency_key("case-y", ResolutionAction.PARTIAL_CREDIT,
                                    payment_intent_id="pi_same", amount_usd=30.0)
    assert key1 != key2
