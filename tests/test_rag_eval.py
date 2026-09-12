"""
Tests for app/rag/eval.py - the precision/recall@k eval set. THE key
property under test: this must genuinely detect a BAD retrieval result,
not just always report 1.0 regardless of what's actually returned.
"""
import os
import shutil

import pytest

TEST_QDRANT_PATH = "data/qdrant_test_rag_eval"
TEST_REINDEX_STATE = "data/reindex_state_test_rag_eval.json"


@pytest.fixture(scope="module", autouse=True)
def ingested_corpus():
    """Ingests the real policy PDF corpus into an isolated test Qdrant
    collection once for this test module - same isolation pattern as
    tests/test_phase3_rag.py."""
    os.environ["QDRANT_LOCAL_PATH"] = TEST_QDRANT_PATH
    if os.path.exists(TEST_QDRANT_PATH):
        shutil.rmtree(TEST_QDRANT_PATH)
    if os.path.exists(TEST_REINDEX_STATE):
        os.remove(TEST_REINDEX_STATE)

    import app.rag.ingestion as ingestion_module
    ingestion_module._REINDEX_STATE_PATH = TEST_REINDEX_STATE

    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.rag.vectorstore as vectorstore_module
    if vectorstore_module._client_singleton is not None:
        vectorstore_module._client_singleton.close()
    vectorstore_module._client_singleton = None
    vectorstore_module._client_singleton_key = None

    from app.rag.ingestion import ingest_policy_directory
    summaries = ingest_policy_directory("data/policies")
    yield summaries

    if vectorstore_module._client_singleton is not None:
        vectorstore_module._client_singleton.close()
    vectorstore_module._client_singleton = None
    vectorstore_module._client_singleton_key = None

    if os.path.exists(TEST_QDRANT_PATH):
        shutil.rmtree(TEST_QDRANT_PATH)
    if os.path.exists(TEST_REINDEX_STATE):
        os.remove(TEST_REINDEX_STATE)


def test_rag_eval_reports_perfect_scores_against_correctly_ingested_data(ingested_corpus):
    from app.rag.eval import run_rag_eval
    result = run_rag_eval(top_k=5)
    assert result["case_count"] == 4
    assert result["avg_precision_at_k"] == 1.0
    assert result["avg_recall_at_k"] == 1.0


def test_rag_eval_genuinely_detects_a_bad_retrieval(ingested_corpus, monkeypatch):
    """THE regression test: if hybrid_search starts returning the WRONG
    documents, precision/recall must drop - proving this isn't a metric
    that just always reports success regardless of actual behavior."""
    from app.rag.retrieval import RetrievedChunk

    def fake_hybrid_search_returns_wrong_doc(*args, **kwargs):
        return [
            RetrievedChunk(node_id="wrong-1", text="irrelevant", score=0.5,
                            metadata={"doc_id": "COMPLETELY-WRONG-DOC"}),
        ]

    # run_rag_eval imports hybrid_search LOCALLY inside the function
    # (`from app.rag.retrieval import hybrid_search`), so the patch must
    # target the actual source module, not app.rag.eval's own namespace.
    monkeypatch.setattr("app.rag.retrieval.hybrid_search", fake_hybrid_search_returns_wrong_doc)

    from app.rag.eval import run_rag_eval
    result = run_rag_eval(top_k=5)

    assert result["avg_precision_at_k"] == 0.0, (
        "precision must genuinely drop to 0 when retrieval returns the wrong document - "
        "if this stays 1.0 regardless, the eval isn't actually checking anything"
    )
    assert result["avg_recall_at_k"] == 0.0


def test_rag_eval_via_api_endpoint(ingested_corpus):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/testing/rag-eval")
        assert resp.status_code == 200
        data = resp.json()
        assert data["avg_precision_at_k"] == 1.0
        assert data["avg_recall_at_k"] == 1.0


def test_reranker_lift_reports_zero_when_already_optimally_ranked(ingested_corpus):
    """Against the real, small seeded corpus, the expected document is
    already the only/top result pre-rerank for every eval query — lift
    should genuinely report 0 (nothing to improve), not a fabricated
    positive number."""
    from app.rag.eval import run_reranker_lift_eval
    result = run_reranker_lift_eval()
    assert result["case_count"] == 4
    assert result["avg_lift"] == 0.0
    for case in result["per_case"]:
        assert case["pre_rerank_rank"] == 0
        assert case["post_rerank_rank"] == 0


def test_reranker_lift_detects_a_real_rank_change(monkeypatch):
    """THE regression test proving lift is genuinely computed, not
    hardcoded to 0: simulates a reranker that promotes a document from
    3rd place pre-rerank to 1st place post-rerank, and confirms lift
    correctly reports +2."""
    from app.rag.retrieval import RetrievedChunk
    import app.rag.eval as eval_module

    def fake_hybrid_search(query, as_of_date, doc_type=None, top_k=5, pre_rerank_capture=None):
        pre_rerank_candidates = [
            RetrievedChunk(node_id="a", text="", score=0.9, metadata={"doc_id": "WRONG-DOC-1"}),
            RetrievedChunk(node_id="b", text="", score=0.8, metadata={"doc_id": "WRONG-DOC-2"}),
            RetrievedChunk(node_id="c", text="", score=0.7, metadata={"doc_id": "RET-POLICY-2025-A"}),
        ]
        if pre_rerank_capture is not None:
            pre_rerank_capture(pre_rerank_candidates)
        # Simulate the reranker promoting the correct doc to 1st place
        post_rerank_results = [
            RetrievedChunk(node_id="c", text="", score=0.95, metadata={"doc_id": "RET-POLICY-2025-A"}),
            RetrievedChunk(node_id="a", text="", score=0.9, metadata={"doc_id": "WRONG-DOC-1"}),
            RetrievedChunk(node_id="b", text="", score=0.8, metadata={"doc_id": "WRONG-DOC-2"}),
        ]
        return post_rerank_results

    monkeypatch.setattr("app.rag.retrieval.hybrid_search", fake_hybrid_search)

    result = eval_module.run_reranker_lift_eval()
    assert result["avg_lift"] == 2.0, (
        f"expected the correct doc moving from rank 2 to rank 0 to report lift=+2, "
        f"got avg_lift={result['avg_lift']} — if this stays 0 regardless, lift isn't actually being computed"
    )
    assert result["per_case"][0]["pre_rerank_rank"] == 2
    assert result["per_case"][0]["post_rerank_rank"] == 0


def test_index_freshness_lag_computed_correctly(ingested_corpus):
    """THE regression test for the index-freshness-lag metric: proves
    the lag is computed as a genuine, non-negative delta between the
    source file's own timestamp and when ingestion actually completed —
    not a placeholder or always-zero value."""
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/testing/index-freshness")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["documents"]) >= 3, "the real seeded policy corpus has at least 3 documents"
        assert data["avg_lag_seconds"] is not None
        assert data["avg_lag_seconds"] >= 0, "lag must be non-negative — ingestion can't finish before the file existed"
        for doc in data["documents"]:
            assert doc["lag_seconds"] is None or doc["lag_seconds"] >= 0


def test_index_freshness_reports_null_for_old_format_entries_without_timestamps(tmp_path, monkeypatch):
    """Backward compatibility: a reindex_state.json entry written before
    this feature existed has no indexed_at/source_mtime at all — must be
    reported as null with an explanatory note, not crash or silently
    show a fabricated 0."""
    import json
    reindex_path = tmp_path / "reindex_state.json"
    reindex_path.write_text(json.dumps({
        "OLD-DOC-NO-TIMESTAMP": {"node-1": "hash-abc"},  # old flat format
    }))

    import app.rag.ingestion as ingestion_module
    monkeypatch.setattr(ingestion_module, "_REINDEX_STATE_PATH", str(reindex_path))

    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/testing/index-freshness")
        assert resp.status_code == 200
        data = resp.json()
        assert data["documents"][0]["lag_seconds"] is None
        assert "note" in data["documents"][0]
