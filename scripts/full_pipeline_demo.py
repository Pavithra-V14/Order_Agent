"""
Full real pipeline demo: fires an actual webhook (the real INPUT this
system responds to), runs the real Diagnosis + Fraud/Inventory agents,
the real Resolution-Policy Workflow, escalates, approves, and shows the
final OUTPUT - all through the same code paths the live API/frontend use.

HONEST NOTE, worth knowing: the webhook handler creates a case
automatically, but does NOT yet auto-trigger the orchestrator
(diagnosis/resolution) - that wiring isn't built. This script does that
"missing glue" step manually (Step 2 below), exactly the way you'd need
to today if driving this from curl/Postman instead of this script.

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
from app.core.db import SessionLocal, ExceptionCase, CaseState
from app.tools.oms import create_order, get_order
from app.tools.wms import seed_stock
from app.tools.payment import get_payment_gateway
from app.agents.orchestrator import run_orchestrator_for_case
from app.agents.resolution_policy_workflow import run_resolution_policy_workflow


def main():
    with TestClient(app) as client:
        db = SessionLocal()
        existing = get_order(db, "ORD-PIPELINE-DEMO")
        if existing:
            print(f"Order ORD-PIPELINE-DEMO already exists (status={existing['status']}) — reusing it.")
            print("(If you want a fully fresh run, delete/rename it in your database first,")
            print(" or change ORDER_ID below to a new value.)\n")
        else:
            create_order(
                db, order_id="ORD-PIPELINE-DEMO", customer_id="CUST-PIPELINE-DEMO", channel="direct",
                status="paid", total_amount_usd=45.0,
                purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
                line_items=[{"sku": "SKU-PIPELINE-DEMO", "category": "apparel", "qty": 1, "price": 45.0}],
                payment_intent_id="pi_pipeline_demo",
            )
        # Re-seeded unconditionally, even when the order already existed:
        # FakePaymentGateway/seed_stock's data lives in-memory (or, for
        # the real Stripe/EasyPost/Shippo gateways, is a separate live
        # system) — it does NOT persist across separate script runs the
        # way the order row in the database does. Skipping this on the
        # "order already exists" branch was a real bug: the second run of
        # this exact script failed with "No such payment_intent_id" for
        # exactly that reason, caught by actually re-running the script
        # twice in a row, not by reasoning about it in the abstract.
        get_payment_gateway().seed_transaction("pi_pipeline_demo", amount_usd=45.0, status="declined")
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
        print(f"  -> case auto-created (DETECTED state): {case_id}\n")

        print("STEP 2 - run the orchestrator (diagnosis + fraud/inventory/customer-context agents)")
        print("         NOT auto-triggered by the webhook yet - this is the manual glue step")
        final_state = run_orchestrator_for_case(
            case_id=case_id, order_id="ORD-PIPELINE-DEMO", customer_id="CUST-PIPELINE-DEMO",
            payment_intent_id="pi_pipeline_demo",
        )
        print(f"  -> diagnosis root causes: {final_state['diagnosis_root_causes']}")
        print(f"  -> fraud risk score: {final_state['fraud_result']['risk_score']}\n")

        print("STEP 3 - run the Resolution-Policy Workflow (guardrails + routing)")
        db2 = SessionLocal()
        result = run_resolution_policy_workflow(
            diagnosis_root_causes=final_state["diagnosis_root_causes"],
            inventory_result=final_state["inventory_result"],
            order_amount_usd=45.0,
            fraud_flag_present=bool(final_state["fraud_result"]["flag"]),
            auto_execute_confidence_threshold=0.90,
            auto_execute_value_ceiling_usd=50.0,
            retrieved_policy_doc_id="RET-POLICY-2025-A",
            retrieved_policy_version="1",
            db=db2, case_id=case_id,
        )
        print(f"  -> routing: {result.routing.value}")
        print(f"  -> proposed: {result.decision.action.value} ${result.decision.amount_usd}\n")

        case = db2.get(ExceptionCase, case_id)
        case.resolution_decision = result.decision.model_dump(mode="json")
        case.state = CaseState.ESCALATED
        db2.commit()
        db2.close()

        print(f"STEP 4 - check the OUTPUT so far via the frontend:")
        print(f"  -> http://127.0.0.1:8000/cases/{case_id}")
        print(f"  -> http://127.0.0.1:8000/escalations  (this case should be listed)\n")

        print("STEP 5 - approve it (this is exactly the API call the Approve button makes)")
        resp = client.post(f"/api/v1/escalations/{case_id}/decision",
                            json={"action": "approve", "decided_by": "human:you@company.com"})
        print(f"  -> {resp.status_code} {resp.json()}\n")

        print("STEP 6 - final OUTPUT")
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
        print("=" * 70)


if __name__ == "__main__":
    main()
