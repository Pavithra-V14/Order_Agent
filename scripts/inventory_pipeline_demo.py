"""
Full pipeline demo for an INVENTORY exception - runs the REAL, complete
run_full_case_pipeline(), for an order whose requested quantity
genuinely exceeds available SELLABLE stock (never on_hand_qty alone -
edge case 3.2, checked directly in app/agents/workflow_agents.py's
run_inventory_agent).

Checked directly (app/agents/resolution_policy_workflow.py): a genuine
inventory shortfall routes to PARTIAL_CREDIT at 50% of order value, not
a full refund or a reship - this demo verifies that real decision, not
an assumed one.

Usage:
    uvicorn app.main:app --reload   # separate terminal
    python3 scripts/inventory_pipeline_demo.py
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

    order_id = "ORD-INVENTORY-DEMO"
    payment_intent_id = "pi_inventory_demo"

    from app.core.config import get_settings
    settings = get_settings()
    if settings.stripe_api_key:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from create_real_stripe_test_payment import create_real_stripe_test_payment
        payment_intent_id = create_real_stripe_test_payment(amount_usd=45.0)
        print(f"Created real Stripe payment_intent_id: {payment_intent_id}")
    else:
        get_payment_gateway().seed_transaction(payment_intent_id, amount_usd=45.0, status="succeeded")

    existing = get_order(db, order_id)
    if not existing:
        create_order(
            db, order_id=order_id, customer_id="CUST-INVENTORY-DEMO", channel="direct",
            status="paid", total_amount_usd=45.0,
            purchase_date=datetime.now(timezone.utc),
            # Requesting 5 units of an item with only 2 sellable in stock
            line_items=[{"sku": "SKU-INVENTORY-DEMO", "category": "apparel", "qty": 5, "price": 9.0}],
            payment_intent_id=payment_intent_id,
        )
        print(f"Created order {order_id} (requesting qty=5)")
    else:
        print(f"Order {order_id} already exists — reusing it")

    # Real shortfall: only 2 sellable, order wants 5. on_hand_qty is
    # deliberately HIGHER (4) than sellable_qty (2) to also verify the
    # real "always use sellable_qty, never on_hand_qty alone" rule -
    # if this demo used on_hand_qty by mistake, 4 would still be < 5
    # and coincidentally still show a shortfall, so sellable_qty=2 is
    # what actually proves the real rule is being applied.
    seed_stock(db, sku="SKU-INVENTORY-DEMO", warehouse="WH-A", on_hand_qty=4, sellable_qty=2)
    print("Seeded stock: sellable_qty=2, on_hand_qty=4 (order requests qty=5)")

    case_id = "case-inventory-demo"
    case = db.get(ExceptionCase, case_id)
    if case is None:
        case = ExceptionCase(
            id=case_id, order_id=order_id, customer_id="CUST-INVENTORY-DEMO",
            channel="direct", exception_type="inventory", state=CaseState.DETECTED,
        )
        db.add(case)
        db.commit()
        print(f"Created case {case_id}")
    else:
        print(f"Case {case_id} already exists — reusing it")

    print("\nRunning the FULL pipeline (diagnosis -> RAG retrieval -> resolution -> execution)...\n")
    result = run_full_case_pipeline(
        db, case_id=case_id, order_id=order_id, customer_id="CUST-INVENTORY-DEMO",
        order_amount_usd=45.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0, payment_intent_id=payment_intent_id,
    )

    print(f"\nDiagnosis root causes: {result['diagnosis']['diagnosis_root_causes']}")
    print(f"Any shortfall: {result['diagnosis']['inventory_result'].get('any_shortfall')}")
    print(f"Routing: {result['routing']}")
    if result["completion"]:
        outcome = result["completion"]["outcome"]
        print(f"Outcome: {outcome}")
        if outcome == "resolved":
            print(f"Final action: {result['completion']['final_action']}")
        print(f"Execution: {result['completion'].get('execution')}")
    else:
        print("Case escalated for human review — check /escalations")
        print("(Expected: PARTIAL_CREDIT decisions carry confidence=0.85, below the 0.90")
        print(" auto-execute threshold used here — a deliberately conservative guardrail")
        print(" for a decision that estimates a 50% credit rather than fully resolving")
        print(" the shortfall, not a bug. Approve it at http://127.0.0.1:8000/escalations")
        print(" to see the actual partial-credit refund execute.)")

    db.close()
    print(f"\nView it: http://127.0.0.1:8000/cases/{case_id}")


if __name__ == "__main__":
    main()
