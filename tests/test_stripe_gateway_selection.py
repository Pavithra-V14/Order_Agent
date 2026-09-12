"""
Tests for app/tools/payment.py's get_payment_gateway() factory - the
regression test for a real, previously-silent bug: this function never
checked for STRIPE_API_KEY at all, unconditionally returning the fake
gateway regardless of what was configured. Every refund in every
environment running this project used the fake gateway the whole time.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def reset_singletons():
    from app.tools.payment import reset_fake_gateway
    reset_fake_gateway()
    os.environ.pop("STRIPE_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    reset_fake_gateway()
    os.environ.pop("STRIPE_API_KEY", None)
    get_settings.cache_clear()


def test_get_payment_gateway_returns_fake_when_no_stripe_key():
    from app.tools.payment import get_payment_gateway, FakePaymentGateway
    gateway = get_payment_gateway()
    assert isinstance(gateway, FakePaymentGateway)


def test_get_payment_gateway_returns_stripe_when_key_configured():
    """THE regression test for the actual bug: with STRIPE_API_KEY set,
    this MUST return a real StripeGateway instance, not silently fall
    back to the fake one - which is exactly what happened before this
    fix, unconditionally, regardless of configuration."""
    os.environ["STRIPE_API_KEY"] = "sk_test_fake_key_for_this_test"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.tools.payment import get_payment_gateway, StripeGateway
    gateway = get_payment_gateway()
    assert isinstance(gateway, StripeGateway), (
        "get_payment_gateway() must select StripeGateway when STRIPE_API_KEY is configured - "
        "this factory previously never checked for it at all"
    )


def test_get_payment_gateway_is_a_stable_singleton():
    from app.tools.payment import get_payment_gateway
    gateway1 = get_payment_gateway()
    gateway2 = get_payment_gateway()
    assert gateway1 is gateway2


def test_stripe_gateway_seed_transaction_is_a_safe_noop(monkeypatch):
    """THE regression test for a real crash risk found directly: every
    demo/seed script in this project calls
    get_payment_gateway().seed_transaction(...) unconditionally — a
    method that only ever existed on FakePaymentGateway. The moment a
    real STRIPE_API_KEY gets configured (making get_payment_gateway()
    correctly select StripeGateway, per the OTHER fix in this same
    file), every one of those scripts would crash with an
    AttributeError. This proves calling it on a real gateway is a safe,
    explained no-op instead."""
    os.environ["STRIPE_API_KEY"] = "sk_test_fake_for_noop_check"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.tools.payment import get_payment_gateway, StripeGateway
    gateway = get_payment_gateway()
    assert isinstance(gateway, StripeGateway)

    # Must not raise
    gateway.seed_transaction("pi_fake_id", amount_usd=45.0, status="declined")
