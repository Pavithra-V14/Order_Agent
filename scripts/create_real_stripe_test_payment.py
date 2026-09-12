"""
Creates a REAL Stripe test-mode PaymentIntent, immediately confirms it
using Stripe's built-in test payment method (pm_card_visa — a special
ID Stripe provides specifically for test mode, requires no real card),
so it ends up in a genuinely "succeeded" state that can then actually
be refunded.

This exists because StripeGateway.issue_refund() calls Stripe's real
Refund.create() API against a payment_intent_id — which only works if
that PaymentIntent genuinely exists and succeeded in Stripe's system.
Every demo/seed script in this project uses a fake string like
"pi_pipeline_demo" for this ID, which works fine against
FakePaymentGateway (nothing validates it) but fails against a real
StripeGateway with an actual "No such payment_intent" error from
Stripe's API, since that ID was never really created there.

HONEST NOTE: this uses Stripe's documented test-mode pattern
(pm_card_visa + confirm=True), but has not been run against a real
Stripe account from this environment (no network access here to
api.stripe.com) — same limitation as StripeGateway itself. If the exact
parameters below don't match your Stripe API version, you'll see the
real error, and we can fix it together the same way the Shippo
integration got fixed: one real error at a time.

Usage:
    python3 scripts/create_real_stripe_test_payment.py
    # prints a real payment_intent_id you can then use in a demo script
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import get_settings


def create_real_stripe_test_payment(amount_usd: float = 45.0) -> str:
    """Creates a real Stripe test-mode PaymentIntent using Stripe's
    built-in test payment method (pm_card_visa — a special ID Stripe
    provides specifically for test mode, requires no real card),
    confirmed immediately so it ends up genuinely "succeeded" and
    refundable. Returns the real payment_intent_id.

    Extracted into a reusable function (rather than only existing as
    this script's own main()) specifically so demo scripts needing a
    real refundable payment (return_pipeline_demo.py, refund_demo.py)
    can create one automatically when needed, instead of requiring a
    person to run this script separately, copy the printed ID, and
    paste it as a command-line argument to a different script — a real,
    unnecessary manual step found not to be fundamental at all once
    someone asked why it existed.

    Raises the real Stripe exception on failure — callers should decide
    how to present that, this function doesn't swallow or reformat it.
    """
    settings = get_settings()
    if not settings.stripe_api_key:
        raise RuntimeError("STRIPE_API_KEY is not set in .env - nothing to create.")

    import stripe
    stripe.api_key = settings.stripe_api_key

    intent = stripe.PaymentIntent.create(
        amount=int(round(amount_usd * 100)),
        currency="usd",
        payment_method="pm_card_visa",
        confirm=True,
        automatic_payment_methods={"enabled": True, "allow_redirects": "never"},
    )
    return intent.id


def create_real_stripe_declined_payment(amount_usd: float = 45.0) -> str:
    """Creates a real Stripe test-mode PaymentIntent that genuinely
    FAILS, using pm_card_visa_chargeDeclined — Stripe's own documented
    reserved test payment method specifically for simulating a real
    card decline (a real API round-trip, a real PaymentIntent object
    created with a real ID, genuinely ending in a failed/declined
    state) — not a fabricated string that was never actually created
    in Stripe's system at all.

    Built specifically to fix a real bug: full_pipeline_demo.py's
    scenario is "payment declined", but previously used a fake, never-
    created payment_intent_id string ("pi_pipeline_demo") — which fails
    with "No such payment_intent" when run against real Stripe, a
    DIFFERENT failure mode than the intended "payment exists but was
    declined" scenario. This creates a real PaymentIntent object that
    genuinely exists and genuinely has a failed status, matching the
    scenario the demo is actually meant to represent.

    Note: confirm=True on a guaranteed-to-fail payment method raises a
    real stripe.error.CardError - this is caught and the PaymentIntent's
    own ID is still returned, since the object exists in Stripe's system
    (in a 'requires_payment_method' state) even though confirmation failed.
    """
    settings = get_settings()
    if not settings.stripe_api_key:
        raise RuntimeError("STRIPE_API_KEY is not set in .env - nothing to create.")

    import stripe
    stripe.api_key = settings.stripe_api_key

    try:
        intent = stripe.PaymentIntent.create(
            amount=int(round(amount_usd * 100)),
            currency="usd",
            payment_method="pm_card_visa_chargeDeclined",
            confirm=True,
            automatic_payment_methods={"enabled": True, "allow_redirects": "never"},
        )
        return intent.id
    except stripe.CardError as e:
        # Expected — this payment method is Stripe's own reserved
        # "always declines" test token. The PaymentIntent object still
        # genuinely exists in Stripe's system even though confirmation
        # failed; Stripe's documented decline-error format includes it
        # nested under e.error.payment_intent (a StripeObject, attribute
        # access, not a plain dict).
        #
        # Found and fixed TWO real bugs here, in sequence, each only
        # discoverable by actually running this against real Stripe:
        # (1) originally caught stripe.error.CardError, which doesn't
        # match what this SDK version actually raises (confirmed:
        # stripe.CardError IS the exact same class object Stripe raises
        # internally as stripe._error.CardError — the except clause
        # simply never matched, so the real CardError propagated
        # straight through uncaught). (2) this function then assumed
        # e.payment_intent was set directly on the exception (it isn't —
        # CardError's own __init__ never sets that attribute at all);
        # the real data lives nested under e.error.payment_intent
        # instead, confirmed by reading the SDK's own error-construction
        # source rather than guessing a second time.
        payment_intent = getattr(e.error, "payment_intent", None) if e.error else None
        intent_id = getattr(payment_intent, "id", None) if payment_intent else None
        if not intent_id:
            # If this is STILL the wrong path (a real possibility, since
            # this couldn't be verified against a live Stripe decline
            # from this environment), fail with a diagnostic showing
            # exactly what IS available, rather than the original
            # opaque CardError traceback pointing into this file.
            available = dir(e.error) if e.error else dir(e)
            raise RuntimeError(
                f"CardError caught correctly, but could not find a payment_intent.id on it via "
                f"e.error.payment_intent.id. This function's assumption about the error's exact "
                f"structure may be wrong for your Stripe SDK version. "
                f"Available attributes on e.error: {available}. "
                f"Raw e.error: {e.error!r}. Original message: {e}"
            ) from e
        return intent_id


def main():
    settings = get_settings()
    if not settings.stripe_api_key:
        print("STRIPE_API_KEY is not set in .env - nothing to test.")
        sys.exit(1)

    import stripe
    stripe.api_key = settings.stripe_api_key

    print("Creating a real Stripe test-mode PaymentIntent...")
    try:
        intent = stripe.PaymentIntent.create(
            amount=4500,  # $45.00, in cents
            currency="usd",
            payment_method="pm_card_visa",  # Stripe's built-in test payment method
            confirm=True,
            automatic_payment_methods={"enabled": True, "allow_redirects": "never"},
        )
        print(f"SUCCESS")
        print(f"  id: {intent.id}")
        print(f"  status: {intent.status}")
        print(f"  amount: ${intent.amount / 100}")
        print()
        if intent.status == "succeeded":
            print("This payment_intent_id is now genuinely refundable. Use it in a demo:")
            print(f"  python3 scripts/refund_demo.py {intent.id}")
        else:
            print(f"Status is '{intent.status}', not 'succeeded' - a refund would likely fail.")
            print("This can happen depending on your Stripe account's payment method configuration.")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        print()
        print("Common causes:")
        print("  - STRIPE_API_KEY is a live key, not a test key (must start with sk_test_)")
        print("  - Your Stripe account has specific payment method restrictions")
        sys.exit(1)


if __name__ == "__main__":
    main()
