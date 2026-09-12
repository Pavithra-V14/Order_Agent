"""
Phase 11 DoD: "simulate a stock-update webhook arriving mid-diagnosis,
confirm the next inventory read reflects the update, not the cached
value."
"""
import os
import tempfile

import pytest

@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_phase11_{os.getpid()}_{id(object())}.db")
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}"

    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)
    db_module.init_db()

    yield db_module

    try:

        if os.path.exists(tmp_path):

            os.remove(tmp_path)

    except PermissionError:

        pass  # Windows may still hold a brief lock from engine cleanup; harmless to leave a stray temp file

@pytest.fixture(autouse=True)
def reset_caches():
    from app.cache.ttl_cache import reset_all_caches
    from app.tools.carrier import reset_fake_carrier
    reset_all_caches()
    reset_fake_carrier()
    yield

def test_webhook_invalidation_reflects_new_stock_immediately(isolated_db):
    """The core scenario: diagnosis reads stock (gets cached), a webhook
    updates the actual stock level mid-diagnosis, and the VERY NEXT read
    (well within the cache's 45s TTL) sees the new number."""
    from app.tools.wms import seed_stock, handle_inventory_update_webhook
    from app.cache.tool_cache import get_stock_cached

    db = isolated_db.SessionLocal()
    seed_stock(db, sku="SKU-CACHE-1", warehouse="WH-A", on_hand_qty=10, sellable_qty=10)

    first_read = get_stock_cached(db, "SKU-CACHE-1", "WH-A")
    assert first_read[0]["sellable_qty"] == 10

    handle_inventory_update_webhook(db, sku="SKU-CACHE-1", warehouse="WH-A",
                                     new_on_hand_qty=2, new_sellable_qty=2)

    second_read = get_stock_cached(db, "SKU-CACHE-1", "WH-A")
    assert second_read[0]["sellable_qty"] == 2, (
        f"Expected the post-webhook value (2) via cache invalidation, "
        f"got {second_read[0]['sellable_qty']} - stale cache was served instead"
    )
    db.close()

def test_negative_control_cache_genuinely_serves_stale_data_without_invalidation(isolated_db):
    """Necessary negative control: without going through the webhook
    handler, a direct DB update is NOT reflected on the next cached read,
    proving the cache genuinely caches (otherwise the DoD test above
    would be meaningless)."""
    from app.tools.wms import seed_stock
    from app.cache.tool_cache import get_stock_cached
    from app.core.db import MockInventoryRecord

    db = isolated_db.SessionLocal()
    seed_stock(db, sku="SKU-CACHE-2", warehouse="WH-A", on_hand_qty=10, sellable_qty=10)

    first_read = get_stock_cached(db, "SKU-CACHE-2", "WH-A")
    assert first_read[0]["sellable_qty"] == 10

    record = db.query(MockInventoryRecord).filter(
        MockInventoryRecord.sku == "SKU-CACHE-2", MockInventoryRecord.warehouse == "WH-A"
    ).first()
    record.sellable_qty = 2
    db.commit()

    stale_read = get_stock_cached(db, "SKU-CACHE-2", "WH-A")
    assert stale_read[0]["sellable_qty"] == 10, (
        "Expected the STALE cached value (10) since nothing invalidated the cache"
    )
    db.close()

def test_carrier_webhook_invalidation(isolated_db):
    from app.tools.carrier import get_carrier_gateway, handle_carrier_status_webhook
    from app.cache.tool_cache import get_tracking_cached

    gateway = get_carrier_gateway()
    gateway.seed_tracking("TRK123", "label_created")

    first = get_tracking_cached("TRK123")
    assert first["status"] == "label_created"

    handle_carrier_status_webhook("TRK123", "delivered")

    second = get_tracking_cached("TRK123")
    assert second["status"] == "delivered"

def test_ttl_cache_expires_after_ttl():
    import time
    from app.cache.ttl_cache import TTLCache

    cache = TTLCache(name="test-ttl")
    cache.set("k1", "v1", ttl_seconds=0.05)
    assert cache.get("k1") == "v1"
    time.sleep(0.1)
    assert cache.get("k1") is None

def test_ttl_cache_permanent_entry_never_expires():
    import time
    from app.cache.ttl_cache import TTLCache

    cache = TTLCache(name="test-permanent")
    cache.set("k1", "v1", ttl_seconds=None)
    time.sleep(0.1)
    assert cache.get("k1") == "v1"

def test_embedding_cache_hit_skips_recompute():
    from app.cache.embedding_cache import get_or_compute_embedding

    call_count = {"n": 0}

    def fake_compute(text):
        call_count["n"] += 1
        return [1.0, 2.0, 3.0]

    v1, hit1 = get_or_compute_embedding("some policy text", fake_compute)
    assert hit1 is False
    assert call_count["n"] == 1

    v2, hit2 = get_or_compute_embedding("some policy text", fake_compute)
    assert hit2 is True
    assert call_count["n"] == 1
    assert v1 == v2

def test_embedding_cache_different_text_is_a_miss():
    from app.cache.embedding_cache import get_or_compute_embedding

    call_count = {"n": 0}

    def fake_compute(text):
        call_count["n"] += 1
        return [len(text)]

    get_or_compute_embedding("text A", fake_compute)
    get_or_compute_embedding("text B", fake_compute)
    assert call_count["n"] == 2

def test_retrieval_cache_hit_recorded_in_trace(isolated_db):
    """Confirms the second identical query is served from cache, not
    recomputed."""
    import shutil
    qdrant_path = "data/qdrant_local_test_phase11"
    if os.path.exists(qdrant_path):
        shutil.rmtree(qdrant_path)
    os.environ["QDRANT_LOCAL_PATH"] = qdrant_path

    from app.core.config import get_settings
    get_settings.cache_clear()
    import app.rag.vectorstore as vectorstore_module
    if vectorstore_module._client_singleton is not None:
        vectorstore_module._client_singleton.close()
    vectorstore_module._client_singleton = None
    vectorstore_module._client_singleton_key = None

    reindex_state = "data/reindex_state_test_phase11.json"
    if os.path.exists(reindex_state):
        os.remove(reindex_state)
    import app.rag.ingestion as ingestion_module
    ingestion_module._REINDEX_STATE_PATH = reindex_state
    ingestion_module.ingest_policy_directory("data/policies")

    from app.rag.traced_retrieval import traced_hybrid_search
    from app.core.tracing import get_trace

    db = isolated_db.SessionLocal()
    traced_hybrid_search(db, trace_id="case-cache-1", query="return window apparel",
                          as_of_date="2025-06-15", doc_type="return_policy", top_k=5)
    traced_hybrid_search(db, trace_id="case-cache-1", query="return window apparel",
                          as_of_date="2025-06-15", doc_type="return_policy", top_k=5)

    trace = get_trace(db, "case-cache-1")
    cache_hits = [s["metadata"]["cache_hit"] for s in trace if s["agent_or_tool_name"] == "rag_retrieval"]
    assert cache_hits == [False, True], f"Expected [miss, hit], got {cache_hits}"
    db.close()

    if vectorstore_module._client_singleton is not None:
        vectorstore_module._client_singleton.close()
    vectorstore_module._client_singleton = None
    vectorstore_module._client_singleton_key = None
    if os.path.exists(qdrant_path):
        shutil.rmtree(qdrant_path)
    if os.path.exists(reindex_state):
        os.remove(reindex_state)
