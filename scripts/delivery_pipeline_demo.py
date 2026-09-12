"""
Full pipeline demo for a DELIVERY exception - runs the REAL, complete
run_full_case_pipeline() (diagnosis -> RAG retrieval -> resolution ->
execution), same depth as full_pipeline_demo.py, but for a genuinely
different underlying situation: a SHIPPED order whose carrier tracking
shows a real delivery problem, rather than a declined payment.

Works against REAL Shippo too, using their documented test-mode magic
tracking numbers (carrier="shippo", tracking_number="SHIPPO_RETURNED")
— a genuine API call to Shippo's real backend that simulates that exact
status, no physical package required. An earlier version of this
comment incorrectly claimed no such capability existed on any real
carrier; that was checked and corrected directly (see
app/tools/carrier.py's ShippoGateway.get_tracking_status() docstring).

HONEST NOTE on how this project actually works, worth knowing: the
`exception_type` field on a case (payment/return/delivery/fraud) is
just a LABEL for categorization and display — diagnosis does NOT branch
on it at all (confirmed directly: grepping the whole app for
`exception_type ==` returns zero matches in real diagnosis logic).
Diagnosis runs the SAME checks (order/payment/inventory/carrier)
regardless of the label, and finds root causes empirically from what
those checks actually reveal.

Usage:
    uvicorn app.main:app --reload   # separate terminal
    python3 scripts/delivery_pipeline_demo.py
"""
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import SessionLocal, ExceptionCase, CaseState, init_db
from app.tools.oms import create_order, get_order
from app.tools.wms import seed_stock
from app.tools.carrier import get_carrier_gateway
from app.agents.orchestrator import run_full_case_pipeline


def main():
    init_db()
    db = SessionLocal()

    from app.core.config import get_settings
    settings = get_settings()

    order_id = "ORD-DELIVERY-DEMO"
    payment_intent_id = "pi_delivery_demo"
    existing = get_order(db, order_id)
    if not existing:
        create_order(
            db, order_id=order_id, customer_id="CUST-DELIVERY-DEMO", channel="direct",
            status="shipped",  # triggers the real carrier check (see module docstring)
            total_amount_usd=45.0,
            purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
            line_items=[{"sku": "SKU-DELIVERY-DEMO", "category": "apparel", "qty": 1, "price": 45.0}],
            payment_intent_id=payment_intent_id,
        )
        print(f"Created order {order_id} (status=shipped)")
    else:
        print(f"Order {order_id} already exists — reusing it")

    # A real, SUCCEEDED payment on record — not just a delivery scenario
    # with no payment at all. Found necessary directly: a case with no
    # payment_intent_id whatsoever let a real Groq LLM's diagnosis
    # planner conclude "payment_issue: missing payment information"
    # from the absence of a payment record alone, WITHOUT ever checking
    # carrier tracking — unlike FakeLLMClient's deterministic planner,
    # a real LLM has genuine latitude in what it decides to check, and
    # "no payment record at all" is legitimately unusual enough to flag
    # on its own. A real shipped order has ALWAYS been paid for by the
    # time it ships; giving this demo a normal, succeeded payment
    # matches that reality and leaves the carrier status as the one
    # genuine anomaly to find, exactly the intended scenario.
    if settings.stripe_api_key:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from create_real_stripe_test_payment import create_real_stripe_test_payment
        payment_intent_id = create_real_stripe_test_payment(amount_usd=45.0)
        print(f"Created real Stripe payment_intent_id: {payment_intent_id}")
    else:
        from app.tools.payment import get_payment_gateway
        get_payment_gateway().seed_transaction(payment_intent_id, amount_usd=45.0, status="succeeded")

    seed_stock(db, sku="SKU-DELIVERY-DEMO", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    from app.core.config import get_settings
    settings = get_settings()
    using_real_shippo = bool(settings.shippo_api_key) and not settings.easypost_api_key

    if settings.easypost_api_key:
        print(f"\n{'!' * 70}")
        print("STOPPING: EasyPost is configured. EasyPost doesn't document an")
        print("equivalent test-mode magic-tracking-number mechanism the way")
        print("Shippo does (see below) — only Shippo's real API genuinely")
        print("supports this demo's needs. Temporarily comment out")
        print("EASYPOST_API_KEY, leaving SHIPPO_API_KEY active if you have one,")
        print("or unset both to use the fake gateway.")
        print(f"{'!' * 70}\n")
        db.close()
        sys.exit(1)

    if using_real_shippo:
        # Shippo's own documented test-mode mechanism: carrier="shippo"
        # (their reserved test-mode token, not a real carrier) combined
        # with a magic tracking number genuinely simulates that exact
        # status via a real API call — no real physical package needed.
        # Found and fixed directly after initially, incorrectly,
        # assuming no such capability existed at all (see
        # app/tools/carrier.py's corrected seed_tracking() docstring for
        # the full story). SHIPPO_RETURNED (package returned to sender)
        # is the closest real equivalent to this demo's original "lost
        # package" intent.
        tracking_number = "SHIPPO_RETURNED"
        carrier = "shippo"
        print(f"Using REAL Shippo test-mode tracking: carrier={carrier!r}, tracking_number={tracking_number!r}")
    else:
        tracking_number = "TRACK-DELIVERY-DEMO-LOST"
        carrier = None
        get_carrier_gateway().seed_tracking(tracking_number, "lost")
        print(f"Seeded tracking {tracking_number} with real problem status: 'lost' (fake gateway)")

    case_id = "case-delivery-demo"
    case = db.get(ExceptionCase, case_id)
    if case is None:
        case = ExceptionCase(
            id=case_id, order_id=order_id, customer_id="CUST-DELIVERY-DEMO",
            channel="direct", exception_type="delivery", state=CaseState.DETECTED,
        )
        db.add(case)
        db.commit()
        print(f"Created case {case_id}")
    else:
        print(f"Case {case_id} already exists — reusing it")

    print("\nRunning the FULL pipeline (diagnosis -> RAG retrieval -> resolution -> execution)...\n")
    result = run_full_case_pipeline(
        db, case_id=case_id, order_id=order_id, customer_id="CUST-DELIVERY-DEMO",
        order_amount_usd=45.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0, tracking_number=tracking_number, carrier=carrier,
        payment_intent_id=payment_intent_id,
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
            print(f"Execution: {result['completion'].get('execution')}")
            print(f"Error detail: {result['completion'].get('error')}")
    else:
        print("Case escalated for human review (not auto-executed) — check /escalations")

    db.close()
    print(f"\nView it: http://127.0.0.1:8000/cases/{case_id}")


if __name__ == "__main__":
    main()
