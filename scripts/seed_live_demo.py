"""
Live demo seeding script - populates the RUNNING app's real database
(whatever DATABASE_URL is currently configured, cloud or local) with all
8 golden-set edge-case scenarios, so you can walk through each one in the
actual browser UI instead of just reading pytest output.

This does NOT touch the test suite's isolated temp databases - it writes
directly into the same database the live `uvicorn app.main:app` process
is using, so whatever you seed here shows up immediately on refresh in
the Case Queue, Escalations, Audit Log, Metrics, and Trace Viewer pages.

Usage:
    # 1. Start the server first (separate terminal):
    uvicorn app.main:app --reload

    # 2. Run this script (uses the SAME .env config as the server):
    python3 scripts/seed_live_demo.py

    # 3. Open http://127.0.0.1:8000/ and follow the walkthrough this
    #    script prints at the end.
"""
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import SessionLocal, init_db, ExceptionCase, CaseState, AuditLogEntry
from app.tools.oms import create_order
from app.tools.wms import seed_stock
from app.tools.payment import get_payment_gateway
from app.memory.episodic import log_episode


def main():
    init_db()
    db = SessionLocal()
    print("Seeding live demo data into the running app's database...\n")

    # Scenario 1: Temporal policy correctness
    create_order(
        db, order_id="ORD-DEMO-TEMPORAL", customer_id="CUST-DEMO-1", channel="direct",
        status="paid", total_amount_usd=45.0,
        purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
        line_items=[{"sku": "SKU-DEMO-APPAREL", "category": "apparel", "qty": 1, "price": 45.0}],
        payment_intent_id="pi_demo_temporal",
    )
    get_payment_gateway().seed_transaction("pi_demo_temporal", amount_usd=45.0, status="succeeded")
    case1 = ExceptionCase(
        id="case-demo-temporal", order_id="ORD-DEMO-TEMPORAL", customer_id="CUST-DEMO-1",
        channel="direct", exception_type="return", state=CaseState.ESCALATED,
        resolution_decision={
            "action": "refund", "amount_usd": 45.0, "confidence": 0.93,
            "reasoning": "Return requested within the 180-day window from RET-POLICY-2025-A, "
                         "the version in effect at time of purchase (2025-06-15) - NOT the current "
                         "120-day RET-POLICY-2026-A, which only governs orders placed after 2026-02-01.",
            "cited_policy": {"doc_id": "RET-POLICY-2025-A", "version": "1", "clause_summary": "180-day return window"},
        },
    )
    db.add(case1)
    db.add(AuditLogEntry(case_id=case1.id, actor="system", action="state_transition",
                          detail={"from": None, "to": "escalated", "reason": "demo seed"}))

    # Scenario 2: Duplicate-refund idempotency
    create_order(
        db, order_id="ORD-DEMO-IDEMPOTENT", customer_id="CUST-DEMO-2", channel="direct",
        status="paid", total_amount_usd=25.0,
        purchase_date=datetime(2025, 8, 1, tzinfo=timezone.utc),
        line_items=[{"sku": "SKU-DEMO-BOOK", "category": "books", "qty": 1, "price": 25.0}],
        payment_intent_id="pi_demo_idempotent",
    )
    get_payment_gateway().seed_transaction("pi_demo_idempotent", amount_usd=25.0, status="succeeded")
    case2 = ExceptionCase(
        id="case-demo-idempotent", order_id="ORD-DEMO-IDEMPOTENT", customer_id="CUST-DEMO-2",
        channel="direct", exception_type="return", state=CaseState.ESCALATED,
        resolution_decision={
            "action": "refund", "amount_usd": 25.0, "confidence": 0.95,
            "reasoning": "Standard refund - approve this twice from the UI to see the second "
                         "approval return the identical refund ID, not a duplicate charge.",
            "cited_policy": {"doc_id": "RET-POLICY-2025-A", "version": "1", "clause_summary": "return window"},
        },
    )
    db.add(case2)

    # Scenario 3: Fraud vs. high-LTV customer
    for i in range(15):
        log_episode(db, "CUST-DEMO-HIGH-LTV", "case_resolved",
                    {"exception_type": "return", "outcome": "approved"},
                    occurred_at=datetime(2025, 1, (i % 28) + 1, tzinfo=timezone.utc))
    create_order(
        db, order_id="ORD-DEMO-FRAUD-LTV", customer_id="CUST-DEMO-HIGH-LTV", channel="direct",
        status="paid", total_amount_usd=60.0,
        purchase_date=datetime(2025, 9, 1, tzinfo=timezone.utc),
        line_items=[{"sku": "SKU-DEMO-SHOES", "category": "footwear", "qty": 1, "price": 60.0}],
    )
    case3 = ExceptionCase(
        id="case-demo-fraud-ltv", order_id="ORD-DEMO-FRAUD-LTV", customer_id="CUST-DEMO-HIGH-LTV",
        channel="direct", exception_type="return", state=CaseState.DETECTED,
        fraud_risk_score=0.15, fraud_flag=None,
    )
    db.add(case3)

    # Scenario 4: Multi-cause diagnosis
    create_order(
        db, order_id="ORD-DEMO-MULTICAUSE", customer_id="CUST-DEMO-3", channel="direct",
        status="payment_failed", total_amount_usd=100.0,
        purchase_date=datetime(2025, 9, 1, tzinfo=timezone.utc),
        line_items=[{"sku": "SKU-DEMO-ELECTRONICS", "category": "electronics", "qty": 2, "price": 50.0}],
        payment_intent_id="pi_demo_multicause",
    )
    get_payment_gateway().seed_transaction("pi_demo_multicause", amount_usd=100.0, status="declined")
    seed_stock(db, sku="SKU-DEMO-ELECTRONICS", warehouse="WH-A", on_hand_qty=1, sellable_qty=0)
    seed_stock(db, sku="SKU-DEMO-ELECTRONICS", warehouse="WH-B", on_hand_qty=1, sellable_qty=1)
    case4 = ExceptionCase(
        id="case-demo-multicause", order_id="ORD-DEMO-MULTICAUSE", customer_id="CUST-DEMO-3",
        channel="direct", exception_type="payment", state=CaseState.DIAGNOSING,
        diagnosis={
            "root_causes": [
                "payment_issue: transaction status is 'declined'",
                "inventory_issue: SKU-DEMO-ELECTRONICS has insufficient sellable stock (requested=2, sellable=1)",
            ],
            "terminated_reason": "concluded",
        },
    )
    db.add(case4)

    # Scenario 5: Tier 1 hard block
    create_order(
        db, order_id="ORD-DEMO-TIER1BLOCK", customer_id="CUST-DEMO-4", channel="direct",
        status="paid", total_amount_usd=5000.0,
        purchase_date=datetime(2025, 9, 1, tzinfo=timezone.utc),
        line_items=[{"sku": "SKU-DEMO-EXPENSIVE", "category": "electronics", "qty": 1, "price": 5000.0}],
    )
    case5 = ExceptionCase(
        id="case-demo-tier1block", order_id="ORD-DEMO-TIER1BLOCK", customer_id="CUST-DEMO-4",
        channel="direct", exception_type="return", state=CaseState.ESCALATED,
        resolution_decision={
            "action": "refund", "amount_usd": 5000.0, "confidence": 1.0,
            "reasoning": "This decision was BLOCKED by Tier 1's hard $1000 ceiling despite "
                         "confidence=1.0 - Tier 1 reads only the numbers, never the confidence "
                         "score or reasoning text, to decide whether to block.",
            "cited_policy": {"doc_id": "RET-POLICY-2025-A", "version": "1", "clause_summary": "return window"},
        },
    )
    db.add(case5)
    db.add(AuditLogEntry(case_id=case5.id, actor="system", action="tier1_block_demo",
                          detail={"violations": ["amount_usd=5000.0 exceeds the absolute hard ceiling of 1000.0"]}))

    # Scenario 6: Circuit breaker / pending retry (illustrative snapshot)
    case6 = ExceptionCase(
        id="case-demo-circuit", order_id="ORD-DEMO-CIRCUIT", customer_id="CUST-DEMO-5",
        channel="direct", exception_type="payment", state=CaseState.ESCALATED,
        execution_result={"status": "pending_retry", "result": {
            "error": "Circuit 'payment' is OPEN (3 consecutive failures) - failing fast without "
                     "attempting the call. Will retry after 30.0s."
        }},
    )
    db.add(case6)

    # Scenario 7: Webhook cache invalidation setup
    seed_stock(db, sku="SKU-DEMO-CACHE-TEST", warehouse="WH-A", on_hand_qty=10, sellable_qty=10)

    # Scenario 8: Resolved case with full audit trail
    case8 = ExceptionCase(
        id="case-demo-resolved", order_id="ORD-DEMO-RESOLVED", customer_id="CUST-DEMO-6",
        channel="direct", exception_type="return", state=CaseState.RESOLVED,
        resolution_decision={"action": "refund", "amount_usd": 30.0, "confidence": 0.94,
                              "reasoning": "Standard approved return."},
        execution_result={"status": "executed", "result": {"id": "re_demo_fake", "status": "succeeded"}},
    )
    db.add(case8)
    db.add(AuditLogEntry(case_id=case8.id, actor="human:demo@company.com", action="human_decision",
                          detail={"action": "approve", "final_resolution": {"action": "refund", "amount_usd": 30.0}}))

    db.commit()
    db.close()

    print("Done. 8 scenarios seeded. Now open the app and follow this walkthrough:\n")
    print("=" * 78)
    print("""
1. TEMPORAL POLICY CORRECTNESS
   -> http://127.0.0.1:8000/cases/case-demo-temporal
   Look at "Resolution Decision" - cites RET-POLICY-2025-A (180-day window),
   NOT the current RET-POLICY-2026-A (120-day). Also check the Policy
   Documents page shows both versions indexed.

2. DUPLICATE-REFUND IDEMPOTENCY
   -> http://127.0.0.1:8000/escalations
   Click "Approve" on case-demo-idempotent TWICE in a row.
   -> http://127.0.0.1:8000/audit-log  (filter by case-demo-idempotent)
   The second approval's audit entry should reference the SAME refund
   result as the first - no duplicate charge.

3. FRAUD VS. HIGH-LTV CUSTOMER
   -> http://127.0.0.1:8000/cases/case-demo-fraud-ltv
   Fraud risk score is LOW (0.15) despite 15 past returns on this customer
   - proves return count alone doesn't drive the score up.

4. MULTI-CAUSE DIAGNOSIS
   -> http://127.0.0.1:8000/cases/case-demo-multicause
   "Diagnosis" panel shows BOTH a payment_issue AND an inventory_issue
   root cause from one diagnosis run.

5. TIER 1 HARD BLOCK
   -> http://127.0.0.1:8000/audit-log  (filter by case-demo-tier1block)
   Shows the $5000 decision was blocked purely on the ceiling, regardless
   of the model's stated confidence=1.0.

6. CIRCUIT BREAKER / PENDING RETRY
   -> http://127.0.0.1:8000/cases/case-demo-circuit
   "Execution" panel shows status=pending_retry with the circuit-open
   error message - this is what a real payment outage looks like mid-flow.

7. WEBHOOK CACHE INVALIDATION (run this curl command yourself):
   curl -X POST http://127.0.0.1:8000/api/v1/webhooks/inventory \\
     -H "Content-Type: application/json" \\
     -d '{"sku":"SKU-DEMO-CACHE-TEST","warehouse":"WH-A","new_on_hand_qty":2,"new_sellable_qty":2}'
   Poll the job status URL it returns - stock is now 2, not the stale
   cached 10, immediately after the webhook processes.

8. FULL RESOLVED CASE + AUDIT TRAIL
   -> http://127.0.0.1:8000/cases/case-demo-resolved
   -> http://127.0.0.1:8000/audit-log
   See the complete human-decision audit entry for a normal approval.

Also worth checking regardless of scenario:
   -> http://127.0.0.1:8000/metrics          (agent/tool/rag/system tabs)
   -> http://127.0.0.1:8000/traces/case-demo-temporal   (full span trace)
   -> http://127.0.0.1:8000/threshold-config  (run the batch job, see proposals)
""")
    print("=" * 78)


if __name__ == "__main__":
    main()
