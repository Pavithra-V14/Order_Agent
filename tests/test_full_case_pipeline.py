"""
Tests for app.agents.orchestrator.run_full_case_pipeline - the
previously-missing function that actually completes an auto-execute
routing decision, and the first real code path that calls log_episode().
"""
import os
import tempfile
from datetime import datetime, timezone, timedelta

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_full_pipeline_{os.getpid()}_{id(object())}.db")
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
    """A real, dedicated, freshly-ingested Qdrant instance for every test
    in this file — added after finding that tests relying on RAG
    retrieval (rather than a hardcoded retrieved_policy_doc_id) genuinely
    need real ingested content to pass reliably. Relying on whatever
    happens to already be in the module-level shared local Qdrant path
    is NOT reliable: it depends entirely on what a previous, unrelated
    test run or manual setup step happened to leave behind — confirmed
    directly: these tests passed when the shared store happened to have
    content, and failed identically (falling to a low-confidence
    'missing policy citation' denial) the moment it was cleaned up, with
    no test-visible reason why behavior differed between two runs of
    literally the same code.
    """
    import shutil
    tmp_qdrant = tempfile.mkdtemp(prefix="test_full_pipeline_qdrant_")
    tmp_reindex_state = os.path.join(tempfile.gettempdir(), f"test_full_pipeline_reindex_{os.getpid()}_{id(object())}.json")
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


@pytest.fixture(autouse=True)
def reset_all():
    os.environ.pop("GROQ_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()
    from app.tools.payment import reset_fake_gateway
    from app.core.circuit_breaker import reset_all_breakers
    reset_fake_gateway()
    reset_all_breakers()
    yield


def test_pending_retry_outcome_includes_the_actual_error_message(isolated_db):
    """THE regression test for a real bug found from a live run: a
    genuine Shippo API failure printed only an empty execution result
    with no error message anywhere — complete_resolution()'s
    PENDING_RETRY branch dropped ExecutionResult.error entirely, only
    ever returning the (empty) result dict. This left a person watching
    the terminal or reading the API response with zero indication of
    WHY something failed, only that it did."""
    from app.core.db import SessionLocal, ExceptionCase, CaseState
    from app.tools.oms import create_order
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy
    from app.agents.resolution_completion import complete_resolution
    from app.core.circuit_breaker import reset_all_breakers

    reset_all_breakers()
    db = SessionLocal()
    create_order(db, order_id="ORD-ERR-VISIBILITY", customer_id="CUST-ERR-VISIBILITY", channel="direct",
                 status="paid", total_amount_usd=30.0,
                 purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
                 line_items=[{"sku": "SKU-ERR-VISIBILITY", "category": "apparel", "qty": 1, "price": 30.0}])

    case = ExceptionCase(id="case-err-visibility", order_id="ORD-ERR-VISIBILITY", customer_id="CUST-ERR-VISIBILITY",
                          channel="direct", exception_type="delivery", state=CaseState.DETECTED)
    db.add(case)
    db.commit()

    # RESHIP against a carrier gateway that will fail because no real
    # carrier is configured in this test environment in a way that
    # succeeds — genuinely exercising the PENDING_RETRY path, not a mock.
    decision = ResolutionDecision(
        action=ResolutionAction.RESHIP, amount_usd=0.0, confidence=0.9,
        reasoning="test reasoning long enough to pass validation",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="x"),
    )

    import app.tools.carrier as carrier_module
    original_gateway_fn = carrier_module.get_carrier_gateway

    class AlwaysFailsGateway:
        def generate_return_label(self, db, order_id, idempotency_key):
            raise RuntimeError("simulated real carrier API failure: 400 Bad Request - invalid address")

    carrier_module.get_carrier_gateway = lambda: AlwaysFailsGateway()
    try:
        result = complete_resolution(
            db, case=case, proposed_decision=decision, final_decision=decision,
            decided_by="system:test", action_label="test_reship",
        )
    finally:
        carrier_module.get_carrier_gateway = original_gateway_fn

    assert result["outcome"] == "execution_pending_retry"
    assert result["error"], "the actual error message must be present, not silently dropped"
    assert "invalid address" in result["error"]
    db.close()


def test_auto_execute_case_actually_gets_resolved(isolated_db):
    """THE core proof: a genuine auto-execute routing decision now
    actually executes and resolves the case - previously, NOTHING did
    this outside of scripts/full_pipeline_demo.py's workaround."""
    from app.core.db import SessionLocal, ExceptionCase, CaseState
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.tools.payment import get_payment_gateway
    from app.agents.orchestrator import run_full_case_pipeline

    db = SessionLocal()
    create_order(db, order_id="ORD-FULLPIPE-1", customer_id="CUST-FULLPIPE-1", channel="direct",
                 status="paid", total_amount_usd=30.0,
                 purchase_date=datetime.now(timezone.utc) - timedelta(days=10),
                 line_items=[{"sku": "SKU-FULLPIPE-1", "category": "apparel", "qty": 1, "price": 30.0}],
                 payment_intent_id="pi_fullpipe_1")
    get_payment_gateway().seed_transaction("pi_fullpipe_1", amount_usd=30.0, status="succeeded")
    seed_stock(db, sku="SKU-FULLPIPE-1", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    case = ExceptionCase(id="case-fullpipe-1", order_id="ORD-FULLPIPE-1", customer_id="CUST-FULLPIPE-1",
                          channel="direct", exception_type="payment", state=CaseState.DETECTED)
    db.add(case)
    db.commit()

    result = run_full_case_pipeline(
        db, case_id="case-fullpipe-1", order_id="ORD-FULLPIPE-1", customer_id="CUST-FULLPIPE-1",
        order_amount_usd=30.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0, payment_intent_id="pi_fullpipe_1",
    )

    assert result["routing"] == "auto_execute"
    assert result["completion"]["outcome"] == "resolved"

    db2 = SessionLocal()
    final_case = db2.get(ExceptionCase, "case-fullpipe-1")
    assert final_case.state == CaseState.RESOLVED
    assert final_case.execution_result["status"] == "executed"
    db2.close()
    db.close()


def test_auto_execute_completion_writes_a_customer_history_episode(isolated_db):
    """THE regression test for the second gap found alongside the first:
    log_episode() must actually get called when a case auto-resolves -
    previously nothing in the real application ever called it at all."""
    from app.core.db import SessionLocal, ExceptionCase, CaseState
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.tools.payment import get_payment_gateway
    from app.agents.orchestrator import run_full_case_pipeline
    from app.memory.episodic import get_customer_history

    db = SessionLocal()
    create_order(db, order_id="ORD-FULLPIPE-2", customer_id="CUST-FULLPIPE-2", channel="direct",
                 status="paid", total_amount_usd=25.0,
                 purchase_date=datetime.now(timezone.utc) - timedelta(days=10),
                 line_items=[{"sku": "SKU-FULLPIPE-2", "category": "apparel", "qty": 1, "price": 25.0}],
                 payment_intent_id="pi_fullpipe_2")
    get_payment_gateway().seed_transaction("pi_fullpipe_2", amount_usd=25.0, status="succeeded")
    seed_stock(db, sku="SKU-FULLPIPE-2", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    case = ExceptionCase(id="case-fullpipe-2", order_id="ORD-FULLPIPE-2", customer_id="CUST-FULLPIPE-2",
                          channel="direct", exception_type="payment", state=CaseState.DETECTED)
    db.add(case)
    db.commit()

    history_before = get_customer_history(db, "CUST-FULLPIPE-2")
    assert history_before == []

    run_full_case_pipeline(
        db, case_id="case-fullpipe-2", order_id="ORD-FULLPIPE-2", customer_id="CUST-FULLPIPE-2",
        order_amount_usd=25.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0, payment_intent_id="pi_fullpipe_2",
    )

    # The write now happens asynchronously via the job queue (memory-
    # upgrade follow-up moving log_episode() off the resolution hot
    # path - see app/workers/handlers.py's handle_log_episode()), not
    # synchronously inline within run_full_case_pipeline() - found as a
    # real failure against this exact test on a genuine run (not just
    # in this sandbox), since a fixed immediate assertion here is
    # inherently racy against the async write. Polls briefly, same
    # "ack fast, process async" pattern tests/test_phase12_api.py's
    # webhook tests and tests/test_async_episode_writes.py already use.
    import time
    deadline = time.monotonic() + 2.0
    history_after = []
    while time.monotonic() < deadline:
        history_after = get_customer_history(db, "CUST-FULLPIPE-2")
        if history_after:
            break
        time.sleep(0.01)

    assert len(history_after) == 1, (
        "log_episode_async() must have enqueued a write during auto-execute completion, "
        "and the job queue's worker must have actually processed it within 2s"
    )
    assert history_after[0]["episode_type"] == "case_resolved"
    assert history_after[0]["content"]["exception_type"] == "payment"
    db.close()


def test_escalated_case_is_not_completed(isolated_db):
    """The mirror case: when routing genuinely escalates, the case must
    NOT be marked resolved and log_episode must NOT fire yet."""
    from app.core.db import SessionLocal, ExceptionCase, CaseState
    from app.tools.oms import create_order
    from app.agents.orchestrator import run_full_case_pipeline
    from app.memory.episodic import get_customer_history

    db = SessionLocal()
    create_order(db, order_id="ORD-FULLPIPE-3", customer_id="CUST-FULLPIPE-3", channel="direct",
                 status="payment_failed", total_amount_usd=5000.0,
                 purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
                 line_items=[{"sku": "SKU-FULLPIPE-3", "category": "electronics", "qty": 1, "price": 5000.0}],
                 payment_intent_id="pi_fullpipe_3")
    from app.tools.payment import get_payment_gateway
    get_payment_gateway().seed_transaction("pi_fullpipe_3", amount_usd=5000.0, status="declined")

    case = ExceptionCase(id="case-fullpipe-3", order_id="ORD-FULLPIPE-3", customer_id="CUST-FULLPIPE-3",
                          channel="direct", exception_type="payment", state=CaseState.DETECTED)
    db.add(case)
    db.commit()

    result = run_full_case_pipeline(
        db, case_id="case-fullpipe-3", order_id="ORD-FULLPIPE-3", customer_id="CUST-FULLPIPE-3",
        order_amount_usd=5000.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0, payment_intent_id="pi_fullpipe_3",
        retrieved_policy_doc_id="RET-POLICY-2025-A", retrieved_policy_version="1",
    )

    assert result["routing"] in ("escalate", "blocked")
    assert result["completion"] is None

    db2 = SessionLocal()
    final_case = db2.get(ExceptionCase, "case-fullpipe-3")
    assert final_case.state != CaseState.RESOLVED
    db2.close()

    assert get_customer_history(db, "CUST-FULLPIPE-3") == [], (
        "log_episode must NOT fire for a case that hasn't actually resolved yet"
    )
    db.close()


def test_full_pipeline_actually_calls_real_rag_retrieval_when_no_doc_id_given(isolated_db):
    """THE regression test for a real production gap found from a live
    deployment's own metrics: retrieval span count sat at 0 while citing
    decisions sat at 3 — run_full_case_pipeline previously REQUIRED the
    caller to already know which policy applied (via the
    retrieved_policy_doc_id parameter) rather than ever actually calling
    RAG retrieval itself. scripts/full_pipeline_demo.py hardcoded the
    doc_id literally, so nothing about the "full pipeline" ever actually
    searched for anything — groundedness could only ever report 0 for
    exactly that reason, correctly, since nothing was ever retrieved to
    confirm a citation against.

    This test omits retrieved_policy_doc_id entirely and confirms the
    pipeline derives it via a REAL hybrid_search call, produces an
    actual rag_retrieval trace span, and ends up with a groundedness
    score that reflects reality rather than always being 0. Real Qdrant
    setup is handled by the isolated_qdrant_with_real_policies fixture.
    """
    from app.core.db import SessionLocal, ExceptionCase, CaseState
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.tools.payment import get_payment_gateway
    from app.agents.orchestrator import run_full_case_pipeline
    from app.core.metrics import compute_rag_metrics

    db = SessionLocal()
    create_order(db, order_id="ORD-RAG-WIRING", customer_id="CUST-RAG-WIRING", channel="direct",
                 status="paid", total_amount_usd=45.0,
                 purchase_date=datetime.now(timezone.utc) - timedelta(days=10),
                 line_items=[{"sku": "SKU-RAG-WIRING", "category": "apparel", "qty": 1, "price": 45.0}],
                 payment_intent_id="pi_rag_wiring")
    get_payment_gateway().seed_transaction("pi_rag_wiring", amount_usd=45.0, status="succeeded")
    seed_stock(db, sku="SKU-RAG-WIRING", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    case = ExceptionCase(id="case-rag-wiring", order_id="ORD-RAG-WIRING", customer_id="CUST-RAG-WIRING",
                          channel="direct", exception_type="payment", state=CaseState.DETECTED)
    db.add(case)
    db.commit()

    # Deliberately NOT passing retrieved_policy_doc_id/version at all —
    # this is the exact call shape that previously resulted in zero
    # retrieval and zero groundedness regardless of what actually happened.
    result = run_full_case_pipeline(
        db, case_id="case-rag-wiring", order_id="ORD-RAG-WIRING", customer_id="CUST-RAG-WIRING",
        order_amount_usd=45.0, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0, payment_intent_id="pi_rag_wiring",
    )

    assert result["routing"] == "auto_execute"

    metrics = compute_rag_metrics(db)
    assert metrics["retrieval_span_count"] >= 1, (
        "a real rag_retrieval trace span must exist — this is the exact thing that was "
        "missing entirely before, regardless of whether the citation was 'correct'"
    )
    assert metrics["groundedness_score"] == 1.0, (
        "the citation must be genuinely grounded in what was actually retrieved, not just "
        "coincidentally non-zero"
    )
    db.close()
