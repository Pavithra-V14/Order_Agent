"""
Tests for refresh_stale_metadata (app/rag/vectorstore.py) and its wiring
into ingest_policy_pdf - generalizes what were two separate, narrower
fixes (refresh_node_metadata for document-level fields only,
refresh_stale_page_numbers for page only) into one mechanism: for a
chunk whose TEXT is unchanged (so it's correctly skipped for
re-embedding), detect and correct drift in ANY metadata field by
comparing what's actually stored against what this run just computed
fresh - without needing to know in advance which field might drift.
"""
import os
import shutil
import uuid

import pytest

from tests.test_phase3_incremental_reindex import _write_simple_policy_pdf

TEST_QDRANT_PATH = "data/qdrant_local_test_page_staleness"
TEST_REINDEX_STATE = "data/reindex_state_test_page_staleness.json"
TEST_POLICY_DIR = "data/policies_page_staleness_test"


@pytest.fixture
def isolated_env():
    """Same isolation shape as test_phase3_incremental_reindex.py's own
    fixture, with its own dedicated paths so the two test files can
    never collide with each other's state."""
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
    ingestion_module._REINDEX_STATE_PATH = original_reindex_state_path
    if original_qdrant_local_path is not None:
        os.environ["QDRANT_LOCAL_PATH"] = original_qdrant_local_path
    else:
        os.environ.pop("QDRANT_LOCAL_PATH", None)
    get_settings.cache_clear()


def test_refresh_stale_metadata_corrects_a_single_drifted_field(isolated_env):
    """Direct, unit-level proof of the primitive: page=3 stored,
    page=4 expected -> corrected via set_payload, other fields untouched."""
    from app.rag.vectorstore import get_qdrant_client, ensure_collection, refresh_stale_metadata
    from qdrant_client.models import PointStruct

    client = get_qdrant_client()
    ensure_collection(client, "test_meta_staleness_1", dim=4)
    node_id = str(uuid.UUID(int=1))
    client.upsert(collection_name="test_meta_staleness_1", points=[
        PointStruct(id=node_id, vector=[0.1, 0.2, 0.3, 0.4], payload={"page": 3, "text": "unchanged text"}),
    ])

    corrected = refresh_stale_metadata(client, "test_meta_staleness_1", {node_id: {"page": 4}})

    assert corrected == {node_id: ["page"]}
    stored = client.retrieve(collection_name="test_meta_staleness_1", ids=[node_id], with_payload=True)
    assert stored[0].payload["page"] == 4
    assert stored[0].payload["text"] == "unchanged text", "correcting one field must not touch any other"


def test_refresh_stale_metadata_corrects_a_non_page_field_too(isolated_env):
    """Proves this genuinely generalizes beyond page - a document-level
    field (e.g. a return-window value added or changed after the chunk
    was first embedded) must be caught exactly the same way."""
    from app.rag.vectorstore import get_qdrant_client, ensure_collection, refresh_stale_metadata
    from qdrant_client.models import PointStruct

    client = get_qdrant_client()
    ensure_collection(client, "test_meta_staleness_2", dim=4)
    node_id = str(uuid.UUID(int=2))
    client.upsert(collection_name="test_meta_staleness_2", points=[
        PointStruct(id=node_id, vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"page": 1, "return_window_days_by_category": None}),
    ])

    corrected = refresh_stale_metadata(client, "test_meta_staleness_2", {
        node_id: {"page": 1, "return_window_days_by_category": {"apparel": 30}},
    })

    assert corrected == {node_id: ["return_window_days_by_category"]}, (
        "only the field that actually differs should be reported/patched - 'page' matched already"
    )
    stored = client.retrieve(collection_name="test_meta_staleness_2", ids=[node_id], with_payload=True)
    assert stored[0].payload["return_window_days_by_category"] == {"apparel": 30}


def test_refresh_stale_metadata_corrects_multiple_fields_in_one_pass(isolated_env):
    """Two fields drift at once - both must be caught and corrected in
    a single call, not just whichever one a narrower, hardcoded check
    happened to look for."""
    from app.rag.vectorstore import get_qdrant_client, ensure_collection, refresh_stale_metadata
    from qdrant_client.models import PointStruct

    client = get_qdrant_client()
    ensure_collection(client, "test_meta_staleness_3", dim=4)
    node_id = str(uuid.UUID(int=3))
    client.upsert(collection_name="test_meta_staleness_3", points=[
        PointStruct(id=node_id, vector=[0.1, 0.2, 0.3, 0.4],
                    payload={"page": 2, "element_type": "text_child", "version": "1"}),
    ])

    corrected = refresh_stale_metadata(client, "test_meta_staleness_3", {
        node_id: {"page": 3, "element_type": "text_child", "version": "2"},
    })

    assert set(corrected[node_id]) == {"page", "version"}, "element_type matched, must not be reported as corrected"
    stored = client.retrieve(collection_name="test_meta_staleness_3", ids=[node_id], with_payload=True)
    assert stored[0].payload["page"] == 3
    assert stored[0].payload["version"] == "2"


def test_refresh_stale_metadata_writes_nothing_when_everything_already_matches(isolated_env):
    """Control case: zero drift must mean zero writes and an empty
    result dict - proves the comparison actually gates every write,
    not just some of them."""
    from app.rag.vectorstore import get_qdrant_client, ensure_collection, refresh_stale_metadata
    from qdrant_client.models import PointStruct

    client = get_qdrant_client()
    ensure_collection(client, "test_meta_staleness_4", dim=4)
    node_id = str(uuid.UUID(int=4))
    client.upsert(collection_name="test_meta_staleness_4", points=[
        PointStruct(id=node_id, vector=[0.1, 0.2, 0.3, 0.4], payload={"page": 5, "version": "1"}),
    ])

    corrected = refresh_stale_metadata(client, "test_meta_staleness_4", {
        node_id: {"page": 5, "version": "1"},
    })

    assert corrected == {}, "a node with no drift at all must be entirely absent from the result"


def test_ingestion_corrects_real_page_drift_on_reingestion(isolated_env):
    """THE end-to-end regression test: simulates the real scenario found
    in production - a chunk's page number in Qdrant is wrong (here,
    deliberately corrupted to stand in for a real reflow), its TEXT is
    completely unchanged, and re-running ingestion on the same,
    unmodified file must correct the stored page back to reality -
    without re-embedding anything, since the skip-logic must still
    correctly treat this as unchanged content."""
    from app.rag.ingestion import ingest_policy_pdf
    from app.rag.vectorstore import get_qdrant_client
    from app.core.config import get_settings

    path = os.path.join(TEST_POLICY_DIR, "TEST-PAGE-DRIFT.pdf")
    _write_simple_policy_pdf(path, "TEST-PAGE-DRIFT", "This sentence never changes across runs.")

    first = ingest_policy_pdf(path)
    assert first["embedded_this_run"] > 0, "first ingestion must genuinely embed something"

    client = get_qdrant_client()
    collection = get_settings().qdrant_collection
    all_points, _ = client.scroll(collection_name=collection, scroll_filter=None, limit=100,
                                    with_payload=True)
    this_doc_points = [p for p in all_points if p.payload.get("doc_id") == "TEST-PAGE-DRIFT"]
    assert this_doc_points, "the freshly-ingested document's points must be findable"

    # Deliberately corrupt the stored page number - standing in for a
    # real reflow (something inserted earlier in a real multi-page PDF)
    # without needing to engineer an actual multi-page document to get
    # the same observable effect: "text unchanged, stored page wrong."
    corrupted_id = this_doc_points[0].id
    real_page = this_doc_points[0].payload["page"]
    wrong_page = real_page + 99
    client.set_payload(collection_name=collection, payload={"page": wrong_page}, points=[corrupted_id])

    verify_corrupted = client.retrieve(collection_name=collection, ids=[corrupted_id], with_payload=["page"])
    assert verify_corrupted[0].payload["page"] == wrong_page, "corruption setup must have actually taken effect"

    second = ingest_policy_pdf(path)
    assert second["embedded_this_run"] == 0, (
        "text is genuinely unchanged - this run must not re-embed anything, "
        "proving the page fix works WITHOUT relying on a full re-embed"
    )

    fixed = client.retrieve(collection_name=collection, ids=[corrupted_id], with_payload=["page"])
    assert fixed[0].payload["page"] == real_page, (
        "re-ingestion must have detected and corrected the drifted page number, "
        "even though the underlying text was never re-embedded"
    )
