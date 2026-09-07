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
