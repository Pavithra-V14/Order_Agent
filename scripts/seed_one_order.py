"""
Seeds ONE order + payment transaction into the live database, so you can
test webhooks via curl/Postman yourself without running the full
scripts/full_pipeline_demo.py end-to-end script.

There is no HTTP endpoint to create an order - in a real deployment,
orders originate upstream in an actual OMS; this system only ever
RECEIVES status-change webhooks about orders that already exist. This
script is that "already exists" step, done once, manually.

Usage:
    python3 scripts/seed_one_order.py
    # then test webhooks against it (Windows cmd example):
    curl -X POST http://127.0.0.1:8000/api/v1/webhooks/oms -H "Content-Type: application/json" -d "{\"order_id\": \"ORD-PIPELINE-DEMO\", \"new_status\": \"payment_failed\"}"
"""
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import SessionLocal, init_db
from app.tools.oms import create_order, get_order
from app.tools.wms import seed_stock
from app.tools.payment import get_payment_gateway


ORDER_ID = "ORD-PIPELINE-DEMO"
CUSTOMER_ID = "CUST-PIPELINE-DEMO"
PAYMENT_INTENT_ID = "pi_pipeline_demo"
SKU = "SKU-PIPELINE-DEMO"


def main():
    init_db()
    db = SessionLocal()

    existing = get_order(db, ORDER_ID)
    if existing:
        print(f"Order {ORDER_ID} already exists (status={existing['status']}) - reusing it.")
    else:
        create_order(
            db, order_id=ORDER_ID, customer_id=CUSTOMER_ID, channel="direct",
            status="paid", total_amount_usd=45.0,
            purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
            line_items=[{"sku": SKU, "category": "apparel", "qty": 1, "price": 45.0}],
            payment_intent_id=PAYMENT_INTENT_ID,
        )
    # Re-seeded unconditionally even if the order already existed — the
    # payment gateway's in-memory transaction data does NOT persist
    # across separate script runs the way the order row does. Confirmed
    # as a real bug in full_pipeline_demo.py by actually running it twice.
    get_payment_gateway().seed_transaction(PAYMENT_INTENT_ID, amount_usd=45.0, status="declined")
    seed_stock(db, sku=SKU, warehouse="WH-A", on_hand_qty=5, sellable_qty=5)
    db.close()

    print(f"Seeded order {ORDER_ID} (customer={CUSTOMER_ID}, status=paid).")
    print(f"Payment intent {PAYMENT_INTENT_ID} is already 'declined' on the gateway side")
    print("(so a payment_failed webhook + diagnosis will find a real root cause).\n")
    print("Now test the webhook, e.g. (Windows cmd):")
    print('  curl -X POST http://127.0.0.1:8000/api/v1/webhooks/oms -H "Content-Type: application/json" '
          '-d "{\\"order_id\\": \\"' + ORDER_ID + '\\", \\"new_status\\": \\"payment_failed\\"}"')


if __name__ == "__main__":
    main()
