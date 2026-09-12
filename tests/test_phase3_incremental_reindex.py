"""
Phase 3 DoD (efficiency claim, architecture doc 8.2.5): incremental
reindexing must re-embed only changed elements, never the full corpus,
and must never touch documents that weren't modified at all.
"""
import os
import shutil

import pytest

TEST_QDRANT_PATH = "data/qdrant_local_test_reindex"
TEST_REINDEX_STATE = "data/reindex_state_test_reindex.json"
TEST_POLICY_DIR = "data/policies_reindex_test"

@pytest.fixture
def isolated_env():
    os.environ["QDRANT_LOCAL_PATH"] = TEST_QDRANT_PATH
    for p in (TEST_QDRANT_PATH,):
        if os.path.exists(p):
            shutil.rmtree(p)
    if os.path.exists(TEST_REINDEX_STATE):
        os.remove(TEST_REINDEX_STATE)
    if os.path.exists(TEST_POLICY_DIR):
        shutil.rmtree(TEST_POLICY_DIR)
    os.makedirs(TEST_POLICY_DIR)

    import app.rag.ingestion as ingestion_module
    ingestion_module._REINDEX_STATE_PATH = TEST_REINDEX_STATE

    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.rag.vectorstore as vectorstore_module
    if vectorstore_module._client_singleton is not None:
        vectorstore_module._client_singleton.close()
    vectorstore_module._client_singleton = None
    vectorstore_module._client_singleton_key = None

    yield

    if vectorstore_module._client_singleton is not None:
        vectorstore_module._client_singleton.close()
    vectorstore_module._client_singleton = None
    vectorstore_module._client_singleton_key = None
    for p in (TEST_QDRANT_PATH, TEST_POLICY_DIR):
        if os.path.exists(p):
            shutil.rmtree(p)
    if os.path.exists(TEST_REINDEX_STATE):
        os.remove(TEST_REINDEX_STATE)

def _write_simple_policy_pdf(path: str, doc_id: str, body_sentence: str):
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(path, pagesize=letter)
    story = [
        Paragraph("Test Policy", styles["Title"]),
        Paragraph(f"Document ID: {doc_id} | Version: 1 | Effective: 2025-01-01 to present", styles["Normal"]),
        Spacer(1, 10),
        Paragraph(body_sentence, styles["Normal"]),
    ]
    doc.build(story)

def test_unchanged_corpus_reembeds_nothing_on_second_run(isolated_env):
    from app.rag.ingestion import ingest_policy_pdf

    path = os.path.join(TEST_POLICY_DIR, "TEST-DOC-A.pdf")
    _write_simple_policy_pdf(path, "TEST-DOC-A", "This is the original policy text.")

    first = ingest_policy_pdf(path)
    assert first["embedded_this_run"] > 0

    second = ingest_policy_pdf(path)
    assert second["embedded_this_run"] == 0, "unchanged document should re-embed nothing"
    assert second["skipped_unchanged"] == first["total_nodes"]

def test_only_modified_document_reembeds_others_stay_skipped(isolated_env):
    from app.rag.ingestion import ingest_policy_pdf

    path_a = os.path.join(TEST_POLICY_DIR, "TEST-DOC-A.pdf")
    path_b = os.path.join(TEST_POLICY_DIR, "TEST-DOC-B.pdf")
    _write_simple_policy_pdf(path_a, "TEST-DOC-A", "This is document A's original text.")
    _write_simple_policy_pdf(path_b, "TEST-DOC-B", "This is document B's original text.")

    ingest_policy_pdf(path_a)
    ingest_policy_pdf(path_b)

    # modify only document A
    _write_simple_policy_pdf(path_a, "TEST-DOC-A", "This is document A's UPDATED text with new content.")

    result_a = ingest_policy_pdf(path_a)
    result_b = ingest_policy_pdf(path_b)  # re-run on unchanged B

    assert result_a["embedded_this_run"] > 0, "modified document must re-embed its changed content"
    assert result_b["embedded_this_run"] == 0, "unmodified sibling document must not be touched"
