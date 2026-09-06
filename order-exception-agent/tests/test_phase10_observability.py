"""
Phase 10 DoD: "Pulling /metrics/rag after running the temporal-correctness
test case shows a groundedness score and the correct cited doc_id/version
in the trace."
"""
import os
import shutil
import tempfile

import pytest

TEST_QDRANT_PATH = "data/qdrant_local_test_phase10"
TEST_REINDEX_STATE = "data/reindex_state_test_phase10.json"


@pytest.fixture
def isolated_env():
    tmp_db = os.path.join(tempfile.gettempdir(), "test_phase10.db")
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_db}"
    os.environ["QDRANT_LOCAL_PATH"] = TEST_QDRANT_PATH
    if os.path.exists(TEST_QDRANT_PATH):
        shutil.rmtree(TEST_QDRANT_PATH)
    if os.path.exists(TEST_REINDEX_STATE):
        os.remove(TEST_REINDEX_STATE)

    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)
    db_module.init_db()

    import app.rag.vectorstore as vectorstore_module
    if vectorstore_module._client_singleton is not None:
        vectorstore_module._client_singleton.close()
    vectorstore_module._client_singleton = None
    vectorstore_module._client_singleton_key = None

    import app.rag.ingestion as ingestion_module
    ingestion_module._REINDEX_STATE_PATH = TEST_REINDEX_STATE
    ingestion_module.ingest_policy_directory("data/policies")

    yield db_module

    if vectorstore_module._client_singleton is not None:
        vectorstore_module._client_singleton.close()
    vectorstore_module._client_singleton = None
    vectorstore_module._client_singleton_key = None
    if os.path.exists(TEST_QDRANT_PATH):
        shutil.rmtree(TEST_QDRANT_PATH)
    if os.path.exists(TEST_REINDEX_STATE):
        os.remove(TEST_REINDEX_STATE)
    if os.path.exists(tmp_db):
        os.remove(tmp_db)


def test_groundedness_after_temporal_correctness_case(isolated_env):
    """THE Phase 10 DoD test."""
    from app.rag.traced_retrieval import traced_hybrid_search
    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow
    from app.core.metrics import compute_rag_metrics
    from app.core.tracing import get_trace

    db = isolated_env.SessionLocal()
    case_id = "case-groundedness-1"

    results = traced_hybrid_search(
        db, trace_id=case_id, query="return window apparel",
        as_of_date="2025-06-15", doc_type="return_policy", top_k=5,
    )
    assert len(results) > 0
    retrieved_doc_ids = {r.metadata.get("doc_id") for r in results}
    assert "RET-POLICY-2025-A" in retrieved_doc_ids

    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: transaction status is 'declined'"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=30.0,
        fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A",
        retrieved_policy_version="1",
        db=db, case_id=case_id,
    )
    assert result.decision.cited_policy.doc_id == "RET-POLICY-2025-A"

    rag_metrics = compute_rag_metrics(db)
    assert rag_metrics["groundedness_score"] == 1.0, f"Expected fully grounded, got {rag_metrics}"
    assert rag_metrics["citing_decision_count"] == 1

    case_entry = rag_metrics["per_case_groundedness"][0]
    assert case_entry["cited_doc_id"] == "RET-POLICY-2025-A"
    assert case_entry["cited_version"] == "1"
    assert case_entry["grounded"] is True
    assert "RET-POLICY-2025-A" in case_entry["actually_retrieved_doc_ids"]

    trace = get_trace(db, case_id)
    span_names = [s["agent_or_tool_name"] for s in trace]
    assert "rag_retrieval" in span_names
    assert "resolution_decision" in span_names
    db.close()


def test_groundedness_flags_a_hallucinated_citation(isolated_env):
    from app.rag.traced_retrieval import traced_hybrid_search
    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow
    from app.core.metrics import compute_rag_metrics

    db = isolated_env.SessionLocal()
    case_id = "case-groundedness-2"

    traced_hybrid_search(db, trace_id=case_id, query="return window apparel",
                          as_of_date="2025-06-15", doc_type="return_policy", top_k=5)

    run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: transaction status is 'declined'"],
        inventory_result={"any_shortfall": False},
        order_amount_usd=30.0,
        fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2026-A",
        retrieved_policy_version="2",
        db=db, case_id=case_id,
    )

    rag_metrics = compute_rag_metrics(db)
    case_entry = rag_metrics["per_case_groundedness"][0]
    assert case_entry["grounded"] is False
    assert rag_metrics["groundedness_score"] == 0.0
    db.close()


def test_metrics_endpoints_via_api(isolated_env):
    from fastapi.testclient import TestClient
    from app.rag.traced_retrieval import traced_hybrid_search
    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow

    db = isolated_env.SessionLocal()
    case_id = "case-api-1"
    traced_hybrid_search(db, trace_id=case_id, query="return window apparel",
                          as_of_date="2025-06-15", doc_type="return_policy", top_k=5)
    run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: transaction status is 'declined'"],
        inventory_result={"any_shortfall": False}, order_amount_usd=30.0,
        fraud_flag_present=False, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0, retrieved_policy_doc_id="RET-POLICY-2025-A",
        retrieved_policy_version="1", db=db, case_id=case_id,
    )
    db.close()

    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/metrics/rag")
        assert resp.status_code == 200
        assert resp.json()["groundedness_score"] == 1.0

        resp2 = client.get(f"/api/v1/traces/{case_id}")
        assert resp2.status_code == 200
        assert len(resp2.json()["spans"]) == 2

        resp3 = client.get("/api/v1/metrics/not_a_real_scope")
        assert resp3.status_code == 404


def test_pii_redacted_before_persist(isolated_env):
    from app.core.tracing import record_span, get_trace

    db = isolated_env.SessionLocal()
    record_span(
        db, trace_id="case-pii-1", agent_or_tool_name="comms_workflow",
        input_data={"customer_email": "jane.doe@example.com"},
        output_data={"body": "Sent confirmation to jane.doe@example.com"},
    )
    trace = get_trace(db, "case-pii-1")
    assert "jane.doe@example.com" not in str(trace)
    assert "REDACTED_EMAIL" in str(trace)
    db.close()


def test_circuit_breaker_trip_creates_an_alert(isolated_env):
    from app.tools.payment import get_payment_gateway, reset_fake_gateway
    from app.core.circuit_breaker import reset_all_breakers
    from app.agents.execution_agent import execute_resolution
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy
    from app.core.alerting import get_recent_alerts

    reset_fake_gateway()
    reset_all_breakers()
    db = isolated_env.SessionLocal()
    gateway = get_payment_gateway()
    gateway.seed_transaction("pi_alert_1", amount_usd=100.0)
    gateway.inject_transient_failures(999)

    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=20.0, confidence=0.9,
        reasoning="Standard refund for alert test.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )
    execute_resolution(db, case_id="case-alert-1", decision=decision, payment_intent_id="pi_alert_1", max_retries=5)

    alerts = get_recent_alerts(db, event_type="circuit_breaker_trip")
    assert len(alerts) >= 1
    db.close()


def test_tier1_block_creates_an_alert(isolated_env):
    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy
    from app.core.alerting import get_recent_alerts

    db = isolated_env.SessionLocal()
    adversarial = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=5000.0, confidence=1.0,
        reasoning="Extremely confident about this large refund amount here.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )
    run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: x"], inventory_result={"any_shortfall": False},
        order_amount_usd=5000.0, fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90, auto_execute_value_ceiling_usd=50.0,
        override_decision=adversarial, db=db, case_id="case-tier1-alert-1",
    )
    alerts = get_recent_alerts(db, event_type="tier1_block")
    assert len(alerts) >= 1
    db.close()
