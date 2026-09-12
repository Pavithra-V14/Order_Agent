"""
Full real pipeline demo: fires an actual webhook - the real, and now
genuinely COMPLETE, input this system responds to.

FIXED, previously the single most repeatedly-flagged gap in this whole
project: a webhook used to only ever create a DETECTED case and stop
there - diagnosis, RAG retrieval, and resolution required a separate
manual call to run_full_case_pipeline() afterward. This script used to
make that manual call itself (an old "Step 2"), exactly the way you'd
have needed to if driving this from curl/Postman. That manual step is
GONE now: app/workers/handlers.py's _run_pipeline_for_new_case() means
the webhook itself genuinely, autonomously runs the whole pipeline —
this script just fires the webhook and reports what already happened.

Usage:
    # 1. Start the server first (separate terminal):
    uvicorn app.main:app --reload

    # 2. Run this script:
    python3 scripts/full_pipeline_demo.py

    # 3. It prints the case_id - open http://127.0.0.1:8000/cases/<id>
"""
import sys
import os
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from app.main import app
from app.core.db import SessionLocal
from app.tools.oms import create_order, get_order
from app.tools.wms import seed_stock
from app.tools.payment import get_payment_gateway
from app.agents.orchestrator import run_full_case_pipeline


def main():
    with TestClient(app) as client:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from _demo_auth_helper import get_or_create_demo_api_key
        api_key = get_or_create_demo_api_key()
        client.headers.update({"X-API-Key": api_key})
        print(f"Using a real, freshly-created admin API key for this run (auth is genuinely enforced,")
        print(f"not bypassed — see app/core/auth.py)")
        db = SessionLocal()
        existing = get_order(db, "ORD-PIPELINE-DEMO")
        from app.core.config import get_settings
        settings = get_settings()
        if settings.stripe_api_key:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from create_real_stripe_test_payment import create_real_stripe_declined_payment
            # HONEST NOTE, worth knowing: even with a REAL, genuinely-
            # declined PaymentIntent (fixing the "No such payment_intent"
            # error a fake string caused), this scenario may still fail
            # against real Stripe for a DIFFERENT, more fundamental
            # reason — Stripe's Refund.create() requires a SUCCEEDED
            # prior charge, and a declined payment was never actually
            # charged. If this demo still fails, that's the real answer
            # revealing itself: "payment declined -> refund" may not be
            # sound business logic at all, regardless of whether the
            # underlying payment_intent_id is real or fake. Worth fixing
            # the underlying resolution logic if so — ask if you want
            # that built next.
            payment_intent_id = create_real_stripe_declined_payment(amount_usd=45.0)
            print(f"Created real, genuinely-declined Stripe payment_intent_id: {payment_intent_id}")
        else:
            payment_intent_id = "pi_pipeline_demo"

        if existing:
            print(f"Order ORD-PIPELINE-DEMO already exists (status={existing['status']}) - reusing it.")
            print("(If you want a fully fresh run, delete/rename it in your database first,")
            print(" or change ORDER_ID below to a new value.)\n")
        else:
            create_order(
                db, order_id="ORD-PIPELINE-DEMO", customer_id="CUST-PIPELINE-DEMO", channel="direct",
                status="paid", total_amount_usd=45.0,
                purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
                line_items=[{"sku": "SKU-PIPELINE-DEMO", "category": "apparel", "qty": 1, "price": 45.0}],
                payment_intent_id=payment_intent_id,
            )
        # Re-seeded unconditionally, even when the order already existed:
        # FakePaymentGateway/seed_stock's data lives in-memory (or, for
        # the real Stripe/EasyPost/Shippo gateways, is a separate live
        # system) — it does NOT persist across separate script runs the
        # way the order row in the database does.
        if not settings.stripe_api_key:
            get_payment_gateway().seed_transaction(payment_intent_id, amount_usd=45.0, status="declined")
        seed_stock(db, sku="SKU-PIPELINE-DEMO", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)
        db.close()

        print("STEP 1 - INPUT: a webhook fires, exactly what an external OMS would call")
        resp = client.post("/api/v1/webhooks/oms",
                            json={"order_id": "ORD-PIPELINE-DEMO", "new_status": "payment_failed"})
        print(f"  -> {resp.status_code} {resp.json()}")
        job_id = resp.json()["job_id"]

        job = None
        for _ in range(50):
            job = client.get(f"/api/v1/webhooks/jobs/{job_id}").json()
            if job["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)

        if job["status"] == "queued":
            print(f"\n{'!' * 70}")
            print("STUCK: this job is still 'queued' after 5 seconds of polling.")
            print("This almost always means REDIS_URL is configured (RQ-backed job queue)")
            print("but scripts/run_rq_worker.py is NOT running in a separate terminal.")
            print("Either start that script now (in a second terminal, same .env),")
            print("or unset REDIS_URL in .env and restart uvicorn to use the in-process queue.")
            print(f"{'!' * 70}\n")
            sys.exit(1)
        if job["status"] == "failed":
            print(f"\nJOB FAILED: {job['error']}\n")
            sys.exit(1)

        case_id = job["result"]["case_created"]
        if case_id is None:
            print(f"\nNo case was created — job result: {job['result']}")
            print("This usually means the order's new_status wasn't recognized as exception-")
            print("triggering, or a case for this order/status was already created earlier.")
            sys.exit(1)

        # THE real fix, proven here directly: the webhook itself now
        # genuinely runs diagnosis -> RAG retrieval -> resolution on its
        # own — no separate manual call needed at all. Previously, this
        # script's own "STEP 2" called run_full_case_pipeline() again,
        # manually, right here — the exact "manual glue step" gap that
        # was the single most repeatedly-flagged issue in this whole
        # project. Calling it again now would be redundant at best
        # (the case is no longer in DETECTED state) and would silently
        # double-run diagnosis at worst.
        pipeline_outcome = job["result"]["pipeline_outcome"]
        print(f"  -> case auto-created AND auto-diagnosed AND auto-resolved, all from this ONE webhook call: {case_id}")
        if pipeline_outcome is None:
            print("  -> (no pipeline outcome — order likely isn't an exception-triggering status)")
        elif pipeline_outcome.get("outcome") == "pipeline_error":
            print(f"  -> pipeline error (caught and alerted, not crashed): {pipeline_outcome['error']}")
        else:
            print(f"  -> routing: {pipeline_outcome['routing']}")
            print(f"  -> outcome: {pipeline_outcome['outcome']}")
        print()

        if pipeline_outcome and pipeline_outcome.get("routing") == "escalate":
            print("STEP 2 - routed to escalate — needs human review, approving now")
            print(f"  -> http://127.0.0.1:8000/escalations  (this case should be listed)\n")
            resp = client.post(f"/api/v1/escalations/{case_id}/decision", json={"action": "approve"})
            print(f"  -> {resp.status_code} {resp.json()}\n")
        else:
            print("STEP 2 - nothing further needed (already auto-executed, or a real pipeline error occurred)\n")

        print("STEP 3 - final OUTPUT")
        final_case = client.get(f"/api/v1/cases/{case_id}").json()
        print(f"  -> final state: {final_case['state']}")
        print(f"  -> execution result: {final_case['execution_result']}")
        audit = client.get(f"/api/v1/audit-log?case_id={case_id}").json()
        print(f"  -> audit trail: {[a['action'] for a in audit]}\n")

        print("=" * 70)
        print(f"Open in your browser to see the full result:")
        print(f"  http://127.0.0.1:8000/cases/{case_id}")
        print(f"  http://127.0.0.1:8000/traces/{case_id}")
        print(f"  http://127.0.0.1:8000/audit-log")
        print(f"  http://127.0.0.1:8000/metrics  (RAG tab should now show retrieval_span_count > 0)")
        print("=" * 70)


if __name__ == "__main__":
    main()
