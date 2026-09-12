"""
Full pipeline demo for a FRAUD-flagged case - runs the REAL, complete
run_full_case_pipeline(), for a customer with a genuine prior fraud
flag AND a same-day address change, together crossing the real
fraud-risk threshold.

HONEST NOTE on the exact real trigger, checked directly rather than
assumed (app/agents/llm_client.py's FakeLLMClient.assess_fraud_risk):
    +0.5 if the customer has 1+ prior fraud_flag_raised episodes
    +0.4 if the shipping address changed the same day as this request
    +0.1 if return count > 10 (weighted lightly on purpose - a
         high-volume but legitimate customer must NOT be penalized for
         return frequency alone)
    flag = True once the total score reaches >= 0.6

This demo seeds ONE prior fraud flag (+0.5) AND a same-day address
change (+0.4) = 0.9, comfortably over the threshold - two real signals
combining, not one exaggerated one, matching how the real scoring rule
is actually meant to be triggered.

Usage:
    uvicorn app.main:app --reload   # separate terminal
    python3 scripts/fraud_pipeline_demo.py
"""
import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import SessionLocal, ExceptionCase, CaseState, init_db
from app.tools.oms import create_order, get_order
from app.tools.wms import seed_stock
from app.tools.payment import get_payment_gateway
from app.memory.episodic import log_episode
from app.agents.orchestrator import run_full_case_pipeline


def main():
    init_db()
    db = SessionLocal()

    customer_id = "CUST-FRAUD-DEMO"
    order_id = "ORD-FRAUD-DEMO"
    payment_intent_id = "pi_fraud_demo"

    from app.core.config import get_settings
    settings = get_settings()
    if settings.stripe_api_key:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from create_real_stripe_test_payment import create_real_stripe_test_payment
        payment_intent_id = create_real_stripe_test_payment(amount_usd=200.0)
        print(f"Created real Stripe payment_intent_id: {payment_intent_id}")
    else:
        get_payment_gateway().seed_transaction(payment_intent_id, amount_usd=200.0, status="succeeded")

    existing = get_order(db, order_id)
    if not existing:
        create_order(
            db, order_id=order_id, customer_id=customer_id, channel="direct",
            status="delivered", total_amount_usd=200.0,
            purchase_date=datetime.now(timezone.utc) - timedelta(days=5),
            line_items=[{"sku": "SKU-FRAUD-DEMO", "category": "electronics", "qty": 1, "price": 200.0}],
            payment_intent_id=payment_intent_id,
        )
        print(f"Created order {order_id}")
    else:
        print(f"Order {order_id} already exists — reusing it")

    seed_stock(db, sku="SKU-FRAUD-DEMO", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    # status="delivered" correctly triggers diagnosis's carrier check —
    # without a real tracking number to fulfill it, the planner keeps
    # re-requesting a check it can never satisfy until hitting the step
    # ceiling instead of concluding cleanly (the exact same class of
    # issue fixed earlier in return_pipeline_demo.py). A real delivered
    # order naturally has tracking info, so this is also just accurate,
    # not a workaround.
    tracking_number = "FRAUD-DEMO-TRACKING-DELIVERED"
    if settings.easypost_api_key or settings.shippo_api_key:
        carrier = "shippo" if settings.shippo_api_key and not settings.easypost_api_key else None
        if carrier:
            tracking_number = "SHIPPO_DELIVERED"
        else:
            tracking_number = None  # EasyPost has no equivalent magic-token mechanism found
    else:
        carrier = None
        from app.tools.carrier import get_carrier_gateway
        get_carrier_gateway().seed_tracking(tracking_number, "delivered")

    # Real signal #1: a genuine prior fraud flag on this customer,
    # exactly the episode type summarize_customer_risk_profile() counts.
    log_episode(
        db, customer_id=customer_id, episode_type="fraud_flag_raised",
        content={"reason": "prior case flagged for suspicious return pattern"},
        occurred_at=datetime.now(timezone.utc) - timedelta(days=20),
    )
    print("Seeded 1 prior fraud_flag_raised episode for this customer")

    case_id = "case-fraud-demo"
    case = db.get(ExceptionCase, case_id)
    if case is None:
        case = ExceptionCase(
            id=case_id, order_id=order_id, customer_id=customer_id,
            channel="direct", exception_type="fraud", state=CaseState.DETECTED,
        )
        db.add(case)
        db.commit()
        print(f"Created case {case_id}")
    else:
        print(f"Case {case_id} already exists — reusing it")

    print("\nRunning the FULL pipeline (diagnosis -> RAG retrieval -> resolution -> execution)...\n")
    # address_changed_same_day=True is real signal #2 - combined with
    # the prior fraud flag above, this crosses the real 0.6 threshold.
    result = run_full_case_pipeline(
        db, case_id=case_id, order_id=order_id, customer_id=customer_id,
        order_amount_usd=200.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0,  # deliberately LOW - forces escalation even if approved
        payment_intent_id=payment_intent_id,
        tracking_number=tracking_number, carrier=carrier,
        address_changed_same_day=True,
    )

    print(f"\nDiagnosis root causes: {result['diagnosis']['diagnosis_root_causes']}")
    print(f"Fraud risk score: {result['diagnosis']['fraud_result']['risk_score']}")
    print(f"Fraud flag: {result['diagnosis']['fraud_result']['flag']}")
    print(f"Fraud reasons: {result['diagnosis']['fraud_result']['reasons']}")
    print(f"Routing: {result['routing']}")
    if result["completion"]:
        outcome = result["completion"]["outcome"]
        print(f"Outcome: {outcome}")
        if outcome == "resolved":
            print(f"Final action: {result['completion']['final_action']}")
        print(f"Execution: {result['completion'].get('execution')}")
    else:
        print("Case escalated for human review (expected — fraud flag forces this) — check /escalations")

    db.close()
    print(f"\nView it: http://127.0.0.1:8000/cases/{case_id}")


if __name__ == "__main__":
    main()
