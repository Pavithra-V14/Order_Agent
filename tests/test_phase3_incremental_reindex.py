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
    original_qdrant_local_path = os.environ.get("QDRANT_LOCAL_PATH")
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
    original_reindex_state_path = ingestion_module._REINDEX_STATE_PATH
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

    # Restore both module-level and env-var state to what it was before
    # this fixture ran - found necessary directly from a real failure:
    # without this, ingestion_module._REINDEX_STATE_PATH and
    # QDRANT_LOCAL_PATH stayed permanently pointed at this fixture's
    # test-specific paths for the rest of the whole pytest session
    # (Python module state and os.environ both persist across test
    # files within the same process), silently breaking any LATER test
    # in ANY OTHER file that assumes ingestion writes to the real
    # data/reindex_state.json or the real local Qdrant path.
    ingestion_module._REINDEX_STATE_PATH = original_reindex_state_path
    if original_qdrant_local_path is not None:
        os.environ["QDRANT_LOCAL_PATH"] = original_qdrant_local_path
    else:
        os.environ.pop("QDRANT_LOCAL_PATH", None)
    get_settings.cache_clear()

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


def test_insertion_near_start_of_page_no_longer_forces_a_full_repage_reembed(isolated_env):
    """THE real, end-to-end proof of the content-defined chunking fix
    in app/rag/chunking.py: inserting one new sentence near the START
    of a multi-sentence page must NOT force every downstream chunk on
    that page to be re-embedded, the way the old position-based
    "every N sentences, counted from the top" splitter did (confirmed
    directly with a standalone simulation before this fix - a single
    inserted sentence there caused every subsequent chunk on the page
    to hash as "new," even though most of the underlying text never
    changed a word).

    This ingests a real multi-sentence page, inserts a new sentence
    near the start, re-ingests, and asserts that re-embedding touched
    only a small minority of the page's total chunks - not all of
    them."""
    from app.rag.ingestion import ingest_policy_pdf
    path = os.path.join(TEST_POLICY_DIR, "TEST-INSERTION-RIPPLE.pdf")
    original_text = (
        "Items may be returned within 30 days of purchase. "
        "Electronics have a 15-day return window instead. "
        "All returns require the original receipt. "
        "Refunds are issued to the original payment method. "
        "Store credit is also available upon request. "
        "Gift cards never expire under this policy."
    )
    _write_simple_policy_pdf(path, "TEST-INSERTION-RIPPLE", original_text)
    first = ingest_policy_pdf(path)
    total_chunks_first_run = first["total_nodes"]
    assert first["embedded_this_run"] == total_chunks_first_run, "first ingestion embeds everything, as expected"

    # Insert ONE new sentence near the START of the page - the exact
    # scenario that broke every downstream chunk under the old scheme.
    edited_text = (
        "Items may be returned within 30 days of purchase. "
        "Clearance items are final sale and cannot be returned. "
        "Electronics have a 15-day return window instead. "
        "All returns require the original receipt. "
        "Refunds are issued to the original payment method. "
        "Store credit is also available upon request. "
        "Gift cards never expire under this policy."
    )
    _write_simple_policy_pdf(path, "TEST-INSERTION-RIPPLE", edited_text)
    second = ingest_policy_pdf(path)

    assert second["embedded_this_run"] > 0, "the genuinely new sentence's chunk must still be re-embedded"
    assert second["embedded_this_run"] < total_chunks_first_run, (
        f"expected re-embedding to touch only a minority of the page's {total_chunks_first_run} chunks "
        f"after a single near-the-start insertion, but {second['embedded_this_run']} were re-embedded - "
        f"this is exactly the ripple the content-defined chunking fix exists to prevent"
    )


def test_reingestion_stale_cleanup_bumps_generation(isolated_env):
    """Confirms a real, previously-missing protection: re-ingestion's
    own stale-chunk deletion (removing content that no longer exists
    in an edited document) must bump the same mutation-generation
    counter the explicit delete endpoint bumps - otherwise a
    concurrent search could still cache genuinely stale data right
    after THIS kind of deletion, even though the same race was already
    closed for the delete button."""
    from app.rag.ingestion import ingest_policy_pdf
    from app.rag.mutation_lock import get_rag_generation, reset_rag_generation_for_tests

    reset_rag_generation_for_tests()
    path = os.path.join(TEST_POLICY_DIR, "TEST-STALE-BUMP.pdf")
    _write_simple_policy_pdf(path, "TEST-STALE-BUMP",
                              "First sentence here. Second sentence here. Third sentence here.")
    ingest_policy_pdf(path)

    generation_before = get_rag_generation()
    # Remove content - triggers the stale-chunk cleanup path, not just
    # an addition.
    _write_simple_policy_pdf(path, "TEST-STALE-BUMP", "First sentence here.")
    ingest_policy_pdf(path)

    assert get_rag_generation() > generation_before, (
        "re-ingestion's stale-chunk cleanup must bump the mutation generation, "
        "the same way the explicit delete endpoint does"
    )
    reset_rag_generation_for_tests()


def test_concurrent_ingestion_and_deletion_do_not_lose_each_others_reindex_state(isolated_env):
    """THE real concurrency proof for the shared, cross-module lock in
    app/rag/reindex_state_lock.py: ingesting one document while
    deleting an UNRELATED, already-existing one, at the same time via
    real threads, must not lose either operation's effect on the
    shared reindex_state.json file."""
    import threading
    from app.rag.ingestion import ingest_policy_pdf

    path_to_delete = os.path.join(TEST_POLICY_DIR, "TEST-CONCURRENT-DELETE.pdf")
    path_to_add = os.path.join(TEST_POLICY_DIR, "TEST-CONCURRENT-ADD.pdf")
    _write_simple_policy_pdf(path_to_delete, "TEST-CONCURRENT-DELETE", "Some existing content.")
    ingest_policy_pdf(path_to_delete)
    _write_simple_policy_pdf(path_to_add, "TEST-CONCURRENT-ADD", "Some new content.")

    import app.rag.ingestion as ingestion_module
    from app.rag.reindex_state_lock import reindex_state_lock

    def _delete_doc_id(doc_id):
        with reindex_state_lock():
            state = ingestion_module._load_reindex_state()
            if doc_id in state:
                del state[doc_id]
            ingestion_module._save_reindex_state(state)

    def _run_ingest():
        ingest_policy_pdf(path_to_add)

    def _run_delete():
        _delete_doc_id("TEST-CONCURRENT-DELETE")

    t1 = threading.Thread(target=_run_ingest)
    t2 = threading.Thread(target=_run_delete)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    final_state = ingestion_module._load_reindex_state()
    assert "TEST-CONCURRENT-ADD" in final_state, (
        "the concurrent ingestion's entry must survive - it must not be lost to a race with the deletion"
    )
    assert "TEST-CONCURRENT-DELETE" not in final_state, (
        "the concurrent deletion's effect must survive - it must not be lost to a race with the ingestion"
    )
