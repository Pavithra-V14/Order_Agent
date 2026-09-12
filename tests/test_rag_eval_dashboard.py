"""
Tests for the live RAG evaluation dashboard (POST /testing/rag-eval-dashboard/run,
GET /testing/rag-eval-dashboard/history) - found fully implemented in
app/api/v1/testing.py and app/core/db.py's RagEvalRunRecord during a
direct audit, but with ZERO test coverage of any kind. Verified working
correctly by hand first (a real run against a real, freshly-ingested
Qdrant genuinely returned precision/recall=1.0 and a persisted history
row), then given permanent regression coverage here.
"""
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    """Dedicated temp SQLite DB - found necessary directly: this test
    file failed with 'attempt to write a readonly database' when run
    as part of the full suite, since relying on the shared, module-
    level SessionLocal() left it vulnerable to whatever DB state an
    earlier test in the same run had left behind. Same pattern used
    elsewhere in this suite (e.g. tests/test_full_case_pipeline.py)."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_rag_eval_dashboard_db_{os.getpid()}_{id(object())}.db")
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
    """Real, dedicated, freshly-ingested Qdrant - this eval genuinely
    executes real retrieval queries, so it needs real ingested content
    to produce meaningful (or even non-crashing) results."""
    import shutil
    tmp_qdrant = tempfile.mkdtemp(prefix="test_rag_eval_dashboard_qdrant_")
    tmp_reindex_state = os.path.join(tempfile.gettempdir(), f"test_rag_eval_dashboard_reindex_{os.getpid()}_{id(object())}.json")
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


def test_run_live_rag_evaluation_returns_real_precision_and_recall():
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        resp = client.post("/api/v1/testing/rag-eval-dashboard/run")
        assert resp.status_code == 200
        data = resp.json()

        assert data["run_id"]
        assert data["run_at"]
        assert data["offline"]["avg_precision_at_k"] is not None
        assert data["offline"]["avg_recall_at_k"] is not None
        assert 0.0 <= data["offline"]["avg_precision_at_k"] <= 1.0
        assert 0.0 <= data["offline"]["avg_recall_at_k"] <= 1.0
        assert len(data["offline"]["per_case"]["precision_recall"]) > 0
        # live faithfulness data can legitimately be None (no real case
        # traffic has run yet in this isolated test), but the KEY must
        # exist and the run must not have crashed computing it
        assert "faithfulness_score" in data["live"]


def test_run_live_rag_evaluation_persists_a_real_history_row():
    """THE regression test proving this isn't just a stateless
    computation — a real row must be queryable afterward via the
    history endpoint, which is what makes this genuinely "live over
    time" rather than a one-off snapshot."""
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        run_resp = client.post("/api/v1/testing/rag-eval-dashboard/run")
        run_id = run_resp.json()["run_id"]

        history_resp = client.get("/api/v1/testing/rag-eval-dashboard/history")
        assert history_resp.status_code == 200
        history = history_resp.json()
        assert any(r["run_id"] == run_id for r in history)


def test_history_shows_multiple_runs_most_recent_first():
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        run1 = client.post("/api/v1/testing/rag-eval-dashboard/run").json()["run_id"]
        run2 = client.post("/api/v1/testing/rag-eval-dashboard/run").json()["run_id"]

        history = client.get("/api/v1/testing/rag-eval-dashboard/history").json()
        assert len(history) >= 2
        run_ids_in_order = [r["run_id"] for r in history]
        assert run_ids_in_order.index(run2) < run_ids_in_order.index(run1), (
            "most recent run must appear first"
        )


def test_history_respects_limit_parameter():
    from app.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as client:
        for _ in range(3):
            client.post("/api/v1/testing/rag-eval-dashboard/run")

        history = client.get("/api/v1/testing/rag-eval-dashboard/history?limit=2").json()
        assert len(history) == 2


def test_live_signal_reflects_real_case_traffic_when_present():
    """The 'live' half of this dashboard should genuinely change when
    real case traffic with a cited policy actually exists, proving it's
    not a hardcoded or always-null field."""
    from datetime import datetime, timezone
    from app.core.db import SessionLocal, TraceSpanRecord
    from app.main import app
    from fastapi.testclient import TestClient

    db = SessionLocal()
    db.add(TraceSpanRecord(
        trace_id="trace-rag-dashboard-test", agent_or_tool_name="rag_retrieval",
        input={}, output={"num_results": 1},
        span_metadata={"retrieval_doc_ids_used": ["RET-POLICY-2025-A"], "retrieval_doc_versions_used": ["1"]},
    ))
    db.add(TraceSpanRecord(
        trace_id="trace-rag-dashboard-test", agent_or_tool_name="resolution_decision",
        input={}, output={},
        span_metadata={"cited_doc_id": "RET-POLICY-2025-A", "cited_version": "1"},
    ))
    db.commit()
    db.close()

    with TestClient(app) as client:
        resp = client.post("/api/v1/testing/rag-eval-dashboard/run")
        data = resp.json()
        assert data["live"]["citing_decision_count"] >= 1
        assert data["live"]["faithfulness_score"] is not None
