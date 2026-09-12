"""
Phase 3 acceptance test -- per the build checklist's Phase 3 DoD:

"Given a test order bound to the OLD policy version, a query for 'what's
the return window for this order' retrieves and correctly cites the OLD
6-month policy, not the current 4-month one."

This is the single most important test in the whole RAG layer -- it proves
the metadata-filter-before-search design (8.2.4) actually works, not just
that retrieval "returns something plausible."
"""
import os
import shutil

import pytest

TEST_QDRANT_PATH = "data/qdrant_local_test_phase3"
TEST_REINDEX_STATE = "data/reindex_state_test_phase3.json"

@pytest.fixture(scope="module", autouse=True)
def ingested_corpus():
    """Ingests the real policy PDF corpus into an isolated test Qdrant
    collection once for this test module."""
    os.environ["QDRANT_LOCAL_PATH"] = TEST_QDRANT_PATH
    if os.path.exists(TEST_QDRANT_PATH):
        shutil.rmtree(TEST_QDRANT_PATH)
    if os.path.exists(TEST_REINDEX_STATE):
        os.remove(TEST_REINDEX_STATE)

    # patch the reindex-state path used by ingestion.py for isolation
    import app.rag.ingestion as ingestion_module
    ingestion_module._REINDEX_STATE_PATH = TEST_REINDEX_STATE

    from app.core.config import get_settings
    get_settings.cache_clear()

    # reset the Qdrant client singleton so it picks up the test path
    # rather than reusing a client opened against the dev path by an
    # earlier test module in this same pytest process
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

def test_ingestion_parsed_both_policy_versions_correctly(ingested_corpus):
    by_doc = {s["doc_id"]: s for s in ingested_corpus}
    assert by_doc["RET-POLICY-2025-A"]["effective_start"] == "2025-01-01"
    assert by_doc["RET-POLICY-2025-A"]["effective_end"] == "2026-01-31"
    assert by_doc["RET-POLICY-2026-A"]["effective_start"] == "2026-02-01"
    assert by_doc["RET-POLICY-2026-A"]["effective_end"] is None

def test_order_under_old_policy_retrieves_old_window_not_new(ingested_corpus):
    """THE core acceptance test. Order purchased 2025-06-15 -> falls inside
    RET-POLICY-2025-A's effective range (2025-01-01 to 2026-01-31) -> must
    retrieve the 180-day apparel window, and must NOT retrieve the 120-day
    window from RET-POLICY-2026-A even though that's the "current" policy."""
    from app.rag.retrieval import hybrid_search

    results = hybrid_search(
        query="return window apparel",
        as_of_date="2025-06-15",
        doc_type="return_policy",
        top_k=5,
    )

    assert len(results) > 0, "expected at least one retrieved chunk"
    all_text = " ".join(r.text for r in results)
    all_doc_ids = {r.metadata.get("doc_id") for r in results}

    assert "RET-POLICY-2026-A" not in all_doc_ids, (
        f"Order under the 2025-A policy incorrectly retrieved content from "
        f"the superseding 2026-A policy. Retrieved doc_ids: {all_doc_ids}"
    )
    assert "180" in all_text, (
        f"Expected the old 180-day apparel window in retrieved text, got: {all_text}"
    )

def test_order_under_new_policy_retrieves_new_window_not_old(ingested_corpus):
    """The mirror case: an order placed AFTER the 2026-02-01 cutover must
    retrieve the current 120-day window, not the superseded 180-day one."""
    from app.rag.retrieval import hybrid_search

    results = hybrid_search(
        query="return window apparel",
        as_of_date="2026-05-01",
        doc_type="return_policy",
        top_k=5,
    )

    assert len(results) > 0
    all_text = " ".join(r.text for r in results)
    all_doc_ids = {r.metadata.get("doc_id") for r in results}

    assert "RET-POLICY-2025-A" not in all_doc_ids, (
        f"Order under the 2026-A policy incorrectly retrieved content from "
        f"the superseded 2025-A policy. Retrieved doc_ids: {all_doc_ids}"
    )
    assert "120" in all_text, (
        f"Expected the new 120-day apparel window in retrieved text, got: {all_text}"
    )

def test_fraud_policy_not_retrieved_for_return_policy_query(ingested_corpus):
    """doc_type filtering: a return-window query should never surface the
    fraud policy document, even though both are "policy" documents."""
    from app.rag.retrieval import hybrid_search

    results = hybrid_search(
        query="return window apparel",
        as_of_date="2025-06-15",
        doc_type="return_policy",
        top_k=5,
    )
    doc_types = {r.metadata.get("doc_type") for r in results}
    assert doc_types <= {"return_policy"}, f"leaked non-return-policy doc: {doc_types}"

def test_chart_caption_is_retrievable(ingested_corpus):
    """Proves the chart-extraction path (8.2.1) actually produced
    searchable text, not just a saved image nobody can query against."""
    from app.rag.retrieval import hybrid_search

    results = hybrid_search(
        query="return window comparison chart category",
        as_of_date="2025-06-15",
        top_k=10,
    )
    chart_hits = [r for r in results if r.metadata.get("element_type") == "chart"]
    assert len(chart_hits) > 0, "expected the chart's caption text to be retrievable"
    assert chart_hits[0].metadata.get("image_path"), "chart node must carry its image_path for citation/display"
