"""
MANUAL CHECK - Stage 3, made VISIBLE on a real case page (not just the
console-printed fit()-count proof in manual_check_stage3_retrieval_caching.py).

Seeds several past resolved "no anomaly" return cases directly into the
resolution-pattern table, matching the EXACT feature-summary text a
real no-anomaly return case produces (see
app/agents/resolution_policy_workflow.py's query_summary format), then
runs ONE new real case through the full pipeline. That new case's
GET /api/v1/cases/{case_id} response, and its /cases/{case_id} browser
page, should show a non-empty "Similar past cases" list under its
Resolution Decision panel - this is Stage 3's cache in action, feeding
the SAME retrieve_similar_past_resolutions() call the real pipeline
already makes.

Usage:
    uvicorn app.main:app --reload   # separate terminal
    python scripts/manual_check_stage3_similar_cases_on_case_page.py
"""
import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import SessionLocal, ExceptionCase, CaseState, init_db
from app.tools.oms import create_order, get_order
from app.tools.wms import seed_stock
from app.tools.payment import get_payment_gateway
from app.tools.carrier import get_carrier_gateway
from app.agents.learning_loop import record_resolution_outcome
from app.agents.orchestrator import run_full_case_pipeline


def _ensure_payment(amount_usd: float, fallback_id: str) -> str:
    """Same real bug, same fix as manual_check_stage1_fraud_fingerprint.py
    (initially fixed there but missed here - a real oversight, this
    script also runs a case through the real pipeline and needs the
    same treatment): StripeGateway.seed_transaction() is a no-op on a
    real gateway, so a hardcoded fake payment_intent_id was never
    actually created there."""
    from app.core.config import get_settings
    settings = get_settings()
    if settings.stripe_api_key:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from create_real_stripe_test_payment import create_real_stripe_test_payment
        real_id = create_real_stripe_test_payment(amount_usd=amount_usd)
        print(f"Created real Stripe payment_intent_id: {real_id}")
        return real_id
    get_payment_gateway().seed_transaction(fallback_id, amount_usd=amount_usd, status="succeeded")
    return fallback_id


def _ensure_order_has_valid_payment(db, order_id: str, amount_usd: float) -> str:
    """Self-healing follow-up to _ensure_payment(): a SECOND real bug,
    found on a real re-run against an order created by an EARLIER,
    unpatched version of this script. Reusing an already-existing
    order's STORED payment_intent_id (to avoid creating a duplicate
    real Stripe payment every run) backfires if that stored ID is
    stale - e.g. it was written by a version of this script that
    predates _ensure_payment() and is just the old fake placeholder
    string. Fixed by actually verifying the stored ID resolves to a
    real, succeeded transaction before trusting it; if not, creates a
    fresh real payment AND updates the order row so this is a one-time
    fix, not a repeat failure on every future run."""
    from app.core.db import MockOrderRecord
    order_row = db.get(MockOrderRecord, order_id)
    try:
        status = get_payment_gateway().get_transaction_status(order_row.payment_intent_id)
        if status.get("status") == "succeeded":
            return order_row.payment_intent_id
    except Exception:
        pass  # stored ID doesn't resolve to a real, valid transaction - fall through and fix it
    print(f"Order {order_id}'s stored payment_intent_id is stale/invalid - creating a fresh one...")
    fresh_id = _ensure_payment(amount_usd=amount_usd, fallback_id=f"pi_{order_id}_refreshed")
    order_row.payment_intent_id = fresh_id
    db.commit()
    return fresh_id


def _tracking_setup():
    """Same real bug, same fix as manual_check_stage1_fraud_fingerprint.py."""
    from app.core.config import get_settings
    settings = get_settings()
    if settings.easypost_api_key:
        print("\n" + "!" * 70)
        print("STOPPING: EasyPost is configured. EasyPost doesn't document an")
        print("equivalent test-mode magic-tracking-number mechanism the way")
        print("Shippo does. Temporarily comment out EASYPOST_API_KEY, leaving")
        print("SHIPPO_API_KEY active if you have one, or unset both to use the")
        print("fake gateway.")
        print("!" * 70 + "\n")
        sys.exit(1)
    if settings.shippo_api_key:
        print("Using REAL Shippo test-mode tracking: carrier='shippo', tracking_number='SHIPPO_DELIVERED'")
        return "SHIPPO_DELIVERED", "shippo"
    tracking_number = "STAGE3-VISIBLE-TRACKING"
    get_carrier_gateway().seed_tracking(tracking_number, "delivered")
    return tracking_number, None


def _seed_history_case(db, case_id):
    """ResolutionPatternEntry.case_id has a REAL foreign key to
    exception_cases.id (app/core/db.py) - enforced by Postgres, NOT by
    SQLite. Found the hard way: this script's first version used
    made-up case_id strings with no backing row, which worked in this
    sandbox's SQLite but failed with a real ForeignKeyViolation the
    first time it ran against a real Postgres database."""
    if not db.get(ExceptionCase, case_id):
        db.add(ExceptionCase(id=case_id, order_id=f"ORD-{case_id}", customer_id=f"CUST-{case_id}",
                              channel="direct", exception_type="return", state=CaseState.RESOLVED))
        db.commit()


def main():
    init_db()
    db = SessionLocal()

    # Matches EXACTLY what a real no-anomaly return case's diagnosis
    # produces (see app/agents/diagnosis_agent.py) and how
    # resolution_policy_workflow.py formats its query text - so the new
    # case below genuinely finds these as similar, the same way a real
    # customer's case would find real past cases.
    feature_summary = "root_causes=['no_anomaly_detected: all checked systems report normal state'], fraud_flag=False"

    print("Seeding 3 past resolved 'no anomaly' return cases into resolution history...")
    for i in range(3):
        case_id = f"case-stage3-history-{i}"
        _seed_history_case(db, case_id)
        record_resolution_outcome(
            db, case_id=case_id, cluster_key="return_direct",
            case_feature_summary=feature_summary,
            agent_proposed_resolution={"action": "refund", "amount_usd": 40.0 + i},
            human_final_resolution={"action": "refund", "amount_usd": 40.0 + i},
        )

    # --- Now a NEW real case, same profile, run through the real pipeline ---
    order_id = "ORD-STAGE3-VISIBLE"
    customer_id = "CUST-STAGE3-VISIBLE"
    existing_order = get_order(db, order_id)
    if existing_order:
        payment_intent_id = _ensure_order_has_valid_payment(db, order_id, amount_usd=45.0)
    else:
        payment_intent_id = _ensure_payment(amount_usd=45.0, fallback_id="pi_stage3_visible")
        create_order(
            db, order_id=order_id, customer_id=customer_id, channel="direct",
            status="delivered", total_amount_usd=45.0,
            purchase_date=datetime.now(timezone.utc) - timedelta(days=3),
            line_items=[{"sku": "SKU-STAGE3-VIS", "category": "apparel", "qty": 1, "price": 45.0}],
            payment_intent_id=payment_intent_id,
        )
        print(f"Created order {order_id}")
    seed_stock(db, sku="SKU-STAGE3-VIS", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)
    tracking_number, carrier = _tracking_setup()

    case_id = "case-stage3-visible"
    case = db.get(ExceptionCase, case_id)
    if not case:
        case = ExceptionCase(id=case_id, order_id=order_id, customer_id=customer_id,
                              channel="direct", exception_type="return", state=CaseState.DETECTED)
        db.add(case)
        db.commit()

    print("\nRunning the real pipeline for the new case...")
    run_full_case_pipeline(
        db, case_id=case_id, order_id=order_id, customer_id=customer_id,
        order_amount_usd=45.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=200.0, payment_intent_id=payment_intent_id,
        tracking_number=tracking_number, carrier=carrier,
    )

    updated_case = db.get(ExceptionCase, case_id)
    similar = (updated_case.resolution_decision or {}).get("similar_past_cases", [])
    print(f"\nsimilar_past_cases found: {len(similar)}")
    for s in similar:
        print(f"  - {s}")

    print("\n" + "=" * 70)
    print(f"NOW OPEN: http://localhost:8000/cases/{case_id}  (in your browser, logged in)")
    print("Scroll to the 'Resolution Decision' panel - 'Similar past cases'")
    print("should list the 3 seeded history entries above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
