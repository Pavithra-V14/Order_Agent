"""
Refund demo - exercises the PAYMENT gateway path end to end against
whichever backend your .env configures (Stripe or FakePaymentGateway),
mirroring what reship_demo.py does for the carrier gateway.

Unlike the carrier path, a refund needs a REAL, ALREADY-SUCCEEDED
payment to refund against — you can't refund something that was never
charged. Run scripts/create_real_stripe_test_payment.py first to create
one, then pass its ID here.

Usage:
    uvicorn app.main:app --reload   # separate terminal

    # Fake gateway (no STRIPE_API_KEY set) - works immediately:
    python3 scripts/refund_demo.py

    # Real Stripe - create a real refundable payment first:
    python3 scripts/create_real_stripe_test_payment.py
    python3 scripts/refund_demo.py pi_xxxxxxxxxxxxx
"""
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import SessionLocal, ExceptionCase, CaseState, init_db
from app.tools.oms import create_order, get_order
from app.tools.payment import get_payment_gateway
from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy
from app.agents.resolution_completion import complete_resolution


def main():
    init_db()
    db = SessionLocal()

    payment_intent_id = sys.argv[1] if len(sys.argv) > 1 else "pi_refund_demo_fake"

    print("Checking which payment backend is active...")
    from app.core.config import get_settings
    settings = get_settings()
    if settings.stripe_api_key:
        print("  -> Stripe (real)")
        if payment_intent_id == "pi_refund_demo_fake":
            print("\nERROR: STRIPE_API_KEY is configured, but no real payment_intent_id was given.")
            print("A fake ID will fail with a real Stripe 'No such payment_intent' error.")
            print("Run this first:")
            print("  python3 scripts/create_real_stripe_test_payment.py")
            print("Then:")
            print(f"  python3 scripts/refund_demo.py <the real pi_... it prints>")
            sys.exit(1)
    else:
        print("  -> FakePaymentGateway (local) — set STRIPE_API_KEY to use real Stripe")
        get_payment_gateway().seed_transaction(payment_intent_id, amount_usd=45.0, status="succeeded")

    order_id = "ORD-REFUND-DEMO"
    existing = get_order(db, order_id)
    if not existing:
        create_order(
            db, order_id=order_id, customer_id="CUST-REFUND-DEMO", channel="direct",
            status="paid", total_amount_usd=45.0,
            purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
            line_items=[{"sku": "SKU-REFUND-DEMO", "category": "apparel", "qty": 1, "price": 45.0}],
            payment_intent_id=payment_intent_id,
        )
        print(f"Created order {order_id}")
    else:
        print(f"Order {order_id} already exists — reusing it")

    case_id = "case-refund-demo"
    case = db.get(ExceptionCase, case_id)
    if case is None:
        case = ExceptionCase(
            id=case_id, order_id=order_id, customer_id="CUST-REFUND-DEMO",
            channel="direct", exception_type="return", state=CaseState.DETECTED,
        )
        db.add(case)
        db.commit()
        print(f"Created case {case_id}")
    else:
        print(f"Case {case_id} already exists — reusing it")

    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=45.0, confidence=0.94,
        reasoning="Item returned within the standard return window, condition verified.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )

    print(f"\nExecuting refund against payment_intent_id={payment_intent_id}...")
    result = complete_resolution(
        db, case=case, proposed_decision=decision, final_decision=decision,
        decided_by="system:refund_demo", action_label="refund_demo",
    )
    print(f"\nOutcome: {result['outcome']}")
    print(f"Execution result: {result.get('execution')}")
    if result.get("error"):
        print(f"Error detail: {result['error']}")

    db.close()
    print(f"\nView it: http://127.0.0.1:8000/cases/{case_id}")


if __name__ == "__main__":
    main()
