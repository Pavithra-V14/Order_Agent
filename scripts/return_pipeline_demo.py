"""
Full pipeline demo for a genuine RETURN - runs the REAL, complete
run_full_case_pipeline(), but for a situation that's meaningfully
DIFFERENT from payment/delivery exceptions: nothing is actually broken.
No payment declined, no inventory shortfall, no lost package — the
customer simply wants to return an item they already received, paid
for, and got successfully.

This is worth seeing precisely BECAUSE it's different: diagnosis runs
the exact same checks (order/payment/inventory/carrier) as every other
case, finds nothing wrong with any of them, and concludes with
"no_anomaly_detected" as the root cause.

FIXED, previously a real gap: "no_anomaly_detected" used to be
unconditionally treated as grounds for DENIAL — backwards from how
return policies actually work, since a plain return with no system
fault is the MOST COMMON legitimate case, not a suspicious one. This
now checks the cited policy's actual, category-specific return window
(parsed from a structured line in the real policy PDF — see
app/rag/metadata.py) against how long ago the order was purchased, and
approves a refund when within it. This demo uses a purchase date 10
days ago specifically to land inside every real policy version's
window, so you should see APPROVE here, not deny — if you want to see
the denial path instead, edit the `timedelta(days=10)` below to
something larger than 180.

Usage:
    uvicorn app.main:app --reload   # separate terminal

    # With real Stripe configured, this creates a real refundable test
    # payment automatically — no manual step needed:
    python3 scripts/return_pipeline_demo.py

    # Optional: reuse a specific existing payment_intent_id instead:
    python3 scripts/return_pipeline_demo.py pi_xxxxxxxxxxxxx
"""
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import SessionLocal, ExceptionCase, CaseState, init_db
from app.tools.oms import create_order, get_order
from app.tools.wms import seed_stock
from app.tools.payment import get_payment_gateway
from app.agents.orchestrator import run_full_case_pipeline


def main():
    init_db()
    db = SessionLocal()

    order_id = "ORD-RETURN-DEMO"

    from app.core.config import get_settings
    settings = get_settings()
    if settings.stripe_api_key:
        if len(sys.argv) >= 2:
            # Optional manual override — reuse a specific existing
            # payment_intent_id instead of creating a fresh one.
            payment_intent_id = sys.argv[1]
            print(f"Using provided Stripe payment_intent_id: {payment_intent_id}")
        else:
            # Creates a real, genuinely refundable Stripe test payment
            # automatically — no separate manual step required. This
            # used to require running scripts/create_real_stripe_test_payment.py
            # by hand and pasting its printed ID as an argument here;
            # there was no real reason that had to be a manual step, so
            # it isn't one anymore.
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from create_real_stripe_test_payment import create_real_stripe_test_payment
            print("Creating a real Stripe test-mode payment automatically...")
            payment_intent_id = create_real_stripe_test_payment(amount_usd=45.0)
            print(f"Created real Stripe payment_intent_id: {payment_intent_id}")
    else:
        payment_intent_id = "pi_return_demo"

    from datetime import timedelta
    # A REALISTIC recent purchase date, not a fixed historical one — the
    # return-window check this demo now exercises is genuinely evaluated
    # relative to today's actual date (correctly: a return window
    # expires relative to when the request is being processed NOW, not
    # relative to some other reference point), so a hardcoded date like
    # "2025-06-15" would show a DENIAL a year+ later regardless of which
    # policy applies, rather than demonstrating the intended APPROVAL path.
    purchase_date = datetime.now(timezone.utc) - timedelta(days=10)
    existing = get_order(db, order_id)
    if not existing:
        create_order(
            db, order_id=order_id, customer_id="CUST-RETURN-DEMO", channel="direct",
            status="delivered",  # order arrived fine — this is a plain return, not a delivery problem
            total_amount_usd=45.0,
            purchase_date=purchase_date,
            line_items=[{"sku": "SKU-RETURN-DEMO", "category": "apparel", "qty": 1, "price": 45.0}],
            payment_intent_id=payment_intent_id,
        )
        print(f"Created order {order_id} (status=delivered, purchased {(datetime.now(timezone.utc) - purchase_date).days} days ago)")
    else:
        print(f"Order {order_id} already exists — reusing it")

    get_payment_gateway().seed_transaction(payment_intent_id, amount_usd=45.0, status="succeeded")
    seed_stock(db, sku="SKU-RETURN-DEMO", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    from app.core.config import get_settings
    settings = get_settings()
    using_real_shippo = bool(settings.shippo_api_key) and not settings.easypost_api_key

    if settings.easypost_api_key:
        print(f"\n{'!' * 70}")
        print("STOPPING: EasyPost is configured. EasyPost doesn't document an")
        print("equivalent test-mode magic-tracking-number mechanism the way")
        print("Shippo does — only Shippo's real API genuinely supports this")
        print("demo's needs. Temporarily comment out EASYPOST_API_KEY, leaving")
        print("SHIPPO_API_KEY active if you have one, or unset both to use the")
        print("fake gateway.")
        print(f"{'!' * 70}\n")
        db.close()
        sys.exit(1)

    # A "delivered" order status correctly triggers diagnosis's carrier
    # check (per the real planner logic in app/agents/llm_client.py) —
    # without a real tracking number to actually check, that check can
    # never be fulfilled, and the planner keeps re-requesting it until
    # hitting the step ceiling instead of concluding cleanly. Found
    # directly: an earlier version of this script omitted this and
    # diagnosis terminated with "diagnosis_incomplete: step ceiling
    # reached" instead of the intended "no_anomaly_detected". A real
    # delivered order naturally has tracking info showing delivered
    # anyway, so this is also just more realistic, not a workaround.
    if using_real_shippo:
        # Shippo's own documented test-mode mechanism: carrier="shippo"
        # (their reserved test-mode token, not a real carrier) combined
        # with a magic tracking number genuinely simulates that exact
        # status via a real API call — no real physical package needed.
        # Found and fixed directly after initially, incorrectly,
        # assuming no such capability existed at all (see
        # app/tools/carrier.py's corrected seed_tracking() docstring).
        tracking_number = "SHIPPO_DELIVERED"
        carrier = "shippo"
        print(f"Using REAL Shippo test-mode tracking: carrier={carrier!r}, tracking_number={tracking_number!r}")
    else:
        tracking_number = "TRACK-RETURN-DEMO-DELIVERED"
        carrier = None
        from app.tools.carrier import get_carrier_gateway
        get_carrier_gateway().seed_tracking(tracking_number, "delivered")

    case_id = "case-return-demo"
    case = db.get(ExceptionCase, case_id)
    if case is None:
        case = ExceptionCase(
            id=case_id, order_id=order_id, customer_id="CUST-RETURN-DEMO",
            channel="direct", exception_type="return", state=CaseState.DETECTED,
        )
        db.add(case)
        db.commit()
        print(f"Created case {case_id}")
    else:
        print(f"Case {case_id} already exists — reusing it")

    print("\nRunning the FULL pipeline (diagnosis -> RAG retrieval -> resolution -> execution)...\n")
    result = run_full_case_pipeline(
        db, case_id=case_id, order_id=order_id, customer_id="CUST-RETURN-DEMO",
        order_amount_usd=45.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0, payment_intent_id=payment_intent_id,
        tracking_number=tracking_number, carrier=carrier,
    )

    print(f"\nDiagnosis root causes: {result['diagnosis']['diagnosis_root_causes']}")
    print(f"Routing: {result['routing']}")
    if result["completion"]:
        outcome = result["completion"]["outcome"]
        print(f"Outcome: {outcome}")
        if outcome == "resolved":
            print(f"Final action: {result['completion']['final_action']}")
        print(f"Execution: {result['completion'].get('execution')}")
    else:
        print("Case escalated for human review — check /escalations")

    db.close()
    print(f"\nView it: http://127.0.0.1:8000/cases/{case_id}")


if __name__ == "__main__":
    main()
