"""
Test for a real production bug: run_full_case_pipeline()'s own
payment_intent_id parameter was silently ignored by complete_resolution(),
which always re-derived it from the order's OWN stored DB record instead.
An order created once (with a stale/placeholder payment_intent_id) and
reused across many later runs, each passing a genuinely fresh
payment_intent_id to run_full_case_pipeline(), kept executing against
the ORIGINAL stale value forever - a real, reproducible "No such
payment_intent" error, since the fresh ID passed to the pipeline had no
actual effect on what execution used.
"""
import os
import tempfile
from datetime import datetime, timezone, timedelta

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    """Same dedicated-temp-database isolation pattern used elsewhere in
    this suite (tests/test_full_case_pipeline.py) - found necessary
    directly: this test failed with 'attempt to write a readonly
    database' when run as part of the full suite, since relying on the
    shared, module-level SessionLocal() left it vulnerable to whatever
    DB state an earlier test in the same run had left behind."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_pi_override_{os.getpid()}_{id(object())}.db")
    import app.core.db as db_module
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    fresh_engine = create_engine(f"sqlite:///{tmp_path}", connect_args={"check_same_thread": False})
    db_module.Base.metadata.create_all(bind=fresh_engine)
    db_module.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=fresh_engine)
    db_module.engine = fresh_engine

    yield db_module

    fresh_engine.dispose()
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except PermissionError:
        pass


@pytest.fixture(autouse=True)
def isolated_qdrant_with_real_policies():
    """A real, dedicated, freshly-ingested Qdrant instance — needed
    since this test's payment is genuinely "succeeded" (required to
    make the refund scenario coherent after fixing a separate bug
    where refunding a never-charged payment was proposed at all),
    meaning diagnosis reaches "no_anomaly_detected", which requires a
    REAL, retrieved policy citation with genuine return-window data to
    approve — a hardcoded retrieved_policy_doc_id bypasses retrieval
    entirely and leaves that data missing."""
    import shutil
    tmp_qdrant = tempfile.mkdtemp(prefix="test_pi_override_qdrant_")
    tmp_reindex_state = os.path.join(tempfile.gettempdir(), f"test_pi_override_reindex_{os.getpid()}_{id(object())}.json")
    os.environ["QDRANT_LOCAL_PATH"] = tmp_qdrant
    from app.core.config import get_settings
    get_settings.cache_clear()
    import app.rag.vectorstore as vectorstore_module
    if vectorstore_module._client_singleton is not None:
        vectorstore_module._client_singleton.close()
    vectorstore_module._client_singleton = None
    vectorstore_module._client_singleton_key = None

    import app.rag.ingestion as ingestion_module
    ingestion_module._REINDEX_STATE_PATH = tmp_reindex_state
    from app.rag.ingestion import ingest_policy_directory
    ingest_policy_directory("data/policies")

    yield

    if vectorstore_module._client_singleton is not None:
        vectorstore_module._client_singleton.close()
    vectorstore_module._client_singleton = None
    vectorstore_module._client_singleton_key = None
    os.environ.pop("QDRANT_LOCAL_PATH", None)
    get_settings.cache_clear()
    shutil.rmtree(tmp_qdrant, ignore_errors=True)
    if os.path.exists(tmp_reindex_state):
        os.remove(tmp_reindex_state)


def test_run_full_case_pipeline_payment_intent_id_overrides_the_orders_stale_stored_value(isolated_db):
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.tools.payment import get_payment_gateway
    from app.agents.orchestrator import run_full_case_pipeline

    db = isolated_db.SessionLocal()

    # The order is created with a STALE payment_intent_id, exactly
    # simulating an order that already existed in the database before
    # a fresh payment was ever created for a later run.
    create_order(
        db, order_id="ORD-STALE-PI-TEST", customer_id="CUST-STALE-PI-TEST", channel="direct",
        status="paid", total_amount_usd=45.0,
        purchase_date=datetime.now(timezone.utc) - timedelta(days=10),
        line_items=[{"sku": "SKU-STALE-PI-TEST", "category": "apparel", "qty": 1, "price": 45.0}],
        payment_intent_id="pi_STALE_PLACEHOLDER_NEVER_REAL",
    )
    seed_stock(db, sku="SKU-STALE-PI-TEST", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    # A genuinely DIFFERENT, fresh payment_intent_id - only THIS one is
    # seeded as a real, succeeded transaction. If complete_resolution()
    # incorrectly falls back to the order's stale stored value instead
    # of this override, the refund attempt will fail against a
    # transaction that was never seeded at all.
    fresh_payment_intent_id = "pi_FRESH_FOR_THIS_RUN"
    get_payment_gateway().seed_transaction(fresh_payment_intent_id, amount_usd=45.0, status="succeeded")

    case = isolated_db.ExceptionCase(
        id="case-stale-pi-test", order_id="ORD-STALE-PI-TEST", customer_id="CUST-STALE-PI-TEST",
        channel="direct", exception_type="payment", state=isolated_db.CaseState.DETECTED,
    )
    db.add(case)
    db.commit()

    result = run_full_case_pipeline(
        db, case_id="case-stale-pi-test", order_id="ORD-STALE-PI-TEST", customer_id="CUST-STALE-PI-TEST",
        order_amount_usd=45.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0,
        payment_intent_id=fresh_payment_intent_id,  # THE override that must actually be used
    )

    assert result["completion"] is not None
    assert result["completion"]["outcome"] == "resolved", (
        f"expected the refund to succeed against the FRESH payment_intent_id passed to "
        f"run_full_case_pipeline(), not fail against the order's stale stored value — "
        f"got: {result['completion']}"
    )
    assert result["completion"]["execution"]["payment_intent"] == fresh_payment_intent_id
    db.close()
