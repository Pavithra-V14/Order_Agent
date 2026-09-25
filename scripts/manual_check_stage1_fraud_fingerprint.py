"""
MANUAL CHECK - Stage 1: cross-customer payment fingerprint fraud signal.

What this seeds:
  - Customer A: an order with payment fingerprint fp_shared_test, plus a
    real fraud_flag_raised episode (so A is a "known bad" customer).
  - Customer B: a DIFFERENT customer, a DIFFERENT order, but the SAME
    payment fingerprint.

What to check afterward: run Customer B's case through the real
pipeline and confirm the fraud agent's reasons mention the match to
Customer A - this is the actual, real signal this session's Stage 1
work added; it did not exist before.

Usage:
    uvicorn app.main:app --reload   # separate terminal
    python scripts/manual_check_stage1_fraud_fingerprint.py
"""
import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import SessionLocal, ExceptionCase, CaseState, init_db
from app.tools.oms import create_order, get_order, compute_payment_fingerprint
from app.tools.wms import seed_stock
from app.tools.payment import get_payment_gateway
from app.memory.episodic import log_episode
from app.agents.orchestrator import run_full_case_pipeline


def _ensure_payment(amount_usd: float, fallback_id: str) -> str:
    """Real bug found running this against a real Stripe account:
    StripeGateway.seed_transaction() is a documented no-op on a real
    gateway (a real payment's status is determined by Stripe itself,
    not declared by this code) - so a hardcoded fake payment_intent_id
    like "pi_stage1_a" was never actually created there, and the
    pipeline's own payment.get_transaction_status() call later failed
    with a real "No such payment_intent" error from Stripe's API.
    Mirrors the exact real/fake detection pattern this project's own
    demo scripts already use (see scripts/fraud_pipeline_demo.py)."""
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


def _tracking_setup():
    """Same real-vs-fake issue as payment: CarrierGateway.seed_tracking()
    is a no-op on a real Shippo/EasyPost gateway. Shippo specifically
    has documented test-mode "magic" tracking numbers that genuinely
    simulate a real status (see scripts/delivery_pipeline_demo.py) -
    mirrored here, including that project's own honest stop for
    EasyPost, which has no equivalent mechanism. Returns
    (tracking_number, carrier) - carrier is None for the fake-gateway
    path (this project's default when no carrier key is set), or
    "shippo" when using the real magic number."""
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
    tracking_number = "STAGE1-DEMO-TRACKING"
    from app.tools.carrier import get_carrier_gateway
    get_carrier_gateway().seed_tracking(tracking_number, "delivered")
    return tracking_number, None


def _ensure_order_has_valid_payment(db, order_id: str, amount_usd: float) -> str:
    """Self-healing follow-up to _ensure_payment(): a real bug found on
    a real re-run against an order created by an EARLIER, unpatched
    version of this script (or manual_check_stage3_similar_cases_on_
    case_page.py, which hit exactly this). Reusing an already-existing
    order's STORED payment_intent_id (to avoid creating a duplicate
    real Stripe payment every run) backfires if that stored ID is
    stale or was never a real transaction. Fixed by verifying the
    stored ID resolves to a real, succeeded transaction before trusting
    it; if not, creates a fresh real payment AND updates the order row
    so this is a one-time fix, not a repeat failure on every future run."""
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


def main():
    init_db()
    db = SessionLocal()

    shared_fingerprint = compute_payment_fingerprint("visa", "4242")
    print(f"Shared fingerprint for both orders: {shared_fingerprint}")

    # --- Customer A: the "known bad" customer -----------------------------
    # A's own payment/tracking is never touched by CUST-B's pipeline run
    # below (only the fingerprint match matters) - no real Stripe
    # payment needed here, any placeholder ID is fine.
    order_a = "ORD-STAGE1-CUST-A"
    if not get_order(db, order_a):
        create_order(
            db, order_id=order_a, customer_id="CUST-STAGE1-A", channel="direct",
            status="delivered", total_amount_usd=100.0,
            purchase_date=datetime.now(timezone.utc) - timedelta(days=5),
            line_items=[{"sku": "SKU-STAGE1-A", "category": "electronics", "qty": 1, "price": 100.0}],
            payment_intent_id="pi_stage1_a_placeholder",
            payment_fingerprint=shared_fingerprint,
        )
        print(f"Created order {order_a} for CUST-STAGE1-A with the shared fingerprint")

    # A real fraud_flag_raised episode for Customer A, written directly
    # (synchronous log_episode, not the async version) so it's
    # immediately queryable - no need to wait for a job to process.
    log_episode(
        db, customer_id="CUST-STAGE1-A", episode_type="fraud_flag_raised",
        content={"reason": "manual test seed", "risk_score": 0.9, "payment_fingerprint": shared_fingerprint},
        occurred_at=datetime.now(timezone.utc),
    )
    print("Seeded a fraud_flag_raised episode for CUST-STAGE1-A")

    # --- Customer B: different person, same card ---------------------------
    # This IS the order the real pipeline runs against below, so its
    # payment_intent_id and tracking must be genuinely real/valid when
    # a real Stripe/carrier gateway is configured. Reuses the order's
    # ALREADY-STORED payment_intent_id on a re-run rather than creating
    # a fresh Stripe payment every time and leaving the order record
    # pointing at a stale ID.
    order_b = "ORD-STAGE1-CUST-B"
    existing_order_b = get_order(db, order_b)
    if existing_order_b:
        payment_intent_id_b = _ensure_order_has_valid_payment(db, order_b, amount_usd=75.0)
    else:
        payment_intent_id_b = _ensure_payment(amount_usd=75.0, fallback_id="pi_stage1_b")
        create_order(
            db, order_id=order_b, customer_id="CUST-STAGE1-B", channel="direct",
            status="delivered", total_amount_usd=75.0,
            purchase_date=datetime.now(timezone.utc) - timedelta(days=2),
            line_items=[{"sku": "SKU-STAGE1-B", "category": "apparel", "qty": 1, "price": 75.0}],
            payment_intent_id=payment_intent_id_b,
            payment_fingerprint=shared_fingerprint,  # <-- the same card
        )
        print(f"Created order {order_b} for CUST-STAGE1-B with the SAME fingerprint")

    seed_stock(db, sku="SKU-STAGE1-B", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)
    tracking_number, carrier = _tracking_setup()

    case_id = "case-stage1-cust-b"
    case = db.get(ExceptionCase, case_id)
    if not case:
        case = ExceptionCase(id=case_id, order_id=order_b, customer_id="CUST-STAGE1-B",
                              channel="direct", exception_type="return", state=CaseState.DETECTED)
        db.add(case)
        db.commit()

    print("\nRunning the real pipeline for CUST-STAGE1-B's case...")
    result = run_full_case_pipeline(
        db, case_id=case_id, order_id=order_b, customer_id="CUST-STAGE1-B",
        order_amount_usd=75.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=200.0, payment_intent_id=payment_intent_id_b,
        tracking_number=tracking_number, carrier=carrier,
    )

    print("\n" + "=" * 70)
    print(f"NOW CHECK: GET /api/v1/cases/{case_id} in Swagger UI (/docs)")
    print("Look at fraud_result / fraud_flag / fraud_risk_score in the response.")
    print("Expected: fraud_flag should be true/flagged, and reasons should")
    print("mention 'payment fingerprint shared with' CUST-STAGE1-A.")
    print("=" * 70)


if __name__ == "__main__":
    main()
