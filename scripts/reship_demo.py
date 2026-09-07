"""
Reship demo - exercises the CARRIER gateway path end to end (Shippo,
EasyPost, or FakeCarrierGateway, whichever your .env configures), which
nothing else in this project's demo scripts actually demonstrates —
full_pipeline_demo.py always ends up on a REFUND, never a reship.

Builds a case where the diagnosis finds a carrier/delivery issue,
forces the resolution decision to RESHIP, and executes it — producing a
real generate_return_label() call through whichever carrier backend is
currently active. Check GET /admin/backend-status (or the Admin page)
first to see which one that is.

Usage:
    uvicorn app.main:app --reload   # separate terminal
    python3 scripts/reship_demo.py
"""
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import SessionLocal, ExceptionCase, CaseState, init_db
from app.tools.oms import create_order, get_order
from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy
from app.agents.execution_agent import execute_resolution
from app.agents.resolution_completion import complete_resolution


def main():
    init_db()
    db = SessionLocal()

    order_id = "ORD-RESHIP-DEMO"
    existing = get_order(db, order_id)
    if not existing:
        create_order(
            db, order_id=order_id, customer_id="CUST-RESHIP-DEMO", channel="direct",
            status="paid", total_amount_usd=60.0,
            purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
            line_items=[{"sku": "SKU-RESHIP-DEMO", "category": "apparel", "qty": 1, "price": 60.0}],
        )
        print(f"Created order {order_id}")
    else:
        print(f"Order {order_id} already exists — reusing it")

    case_id = "case-reship-demo"
    case = db.get(ExceptionCase, case_id)
    if case is None:
        case = ExceptionCase(
            id=case_id, order_id=order_id, customer_id="CUST-RESHIP-DEMO",
            channel="direct", exception_type="delivery", state=CaseState.DETECTED,
        )
        db.add(case)
        db.commit()
        print(f"Created case {case_id}")
    else:
        print(f"Case {case_id} already exists — reusing it")

    print("\nChecking which carrier backend is active...")
    from app.core.config import get_settings
    settings = get_settings()
    if settings.easypost_api_key:
        print("  -> EasyPost (real)")
    elif settings.shippo_api_key:
        print("  -> Shippo (real)")
    else:
        print("  -> FakeCarrierGateway (local) — set EASYPOST_API_KEY or SHIPPO_API_KEY to use a real one")

    decision = ResolutionDecision(
        action=ResolutionAction.RESHIP, amount_usd=0.0, confidence=0.93,
        reasoning="Package reported lost in transit — reshipping under the carrier's delivery guarantee.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="delivery guarantee"),
    )

    print("\nExecuting reship (this calls the REAL carrier gateway if configured)...")
    result = complete_resolution(
        db, case=case, proposed_decision=decision, final_decision=decision,
        decided_by="system:reship_demo", action_label="reship_demo",
    )
    print(f"\nOutcome: {result['outcome']}")
    print(f"Execution result: {result.get('execution')}")
    if result.get("error"):
        print(f"Error detail: {result['error']}")

    db.close()
    print(f"\nView it: http://127.0.0.1:8000/cases/{case_id}")


if __name__ == "__main__":
    main()
