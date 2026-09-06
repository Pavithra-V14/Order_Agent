"""
Tests for the Redis-backed cache path - real Redis, not mocked. Requires
a local redis-server running on localhost:6379. Skipped automatically if
no Redis is reachable, so this doesn't break `pytest tests/` in an
environment without one.
"""
import time

import pytest


def _redis_available() -> bool:
    try:
        import redis
        client = redis.from_url("redis://localhost:6379/0")
        client.ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _redis_available(), reason="No local Redis reachable on localhost:6379")


@pytest.fixture(autouse=True)
def redis_settings():
    import os
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.cache.ttl_cache import reset_all_caches
    reset_all_caches()

    yield

    reset_all_caches()
    os.environ.pop("REDIS_URL", None)
    get_settings.cache_clear()


def test_get_cache_returns_redis_backed_instance_when_configured(redis_settings):
    from app.cache.ttl_cache import get_cache, RedisTTLCache
    cache = get_cache("test_redis_selection")
    assert isinstance(cache, RedisTTLCache), (
        "get_cache() must return RedisTTLCache when settings.redis_url is set"
    )


def test_redis_cache_set_get_roundtrip(redis_settings):
    from app.cache.ttl_cache import get_cache
    cache = get_cache("test_redis_roundtrip")
    cache.set("k1", {"a": 1, "b": [1, 2, 3]}, ttl_seconds=60)
    assert cache.get("k1") == {"a": 1, "b": [1, 2, 3]}


def test_redis_cache_ttl_actually_expires(redis_settings):
    from app.cache.ttl_cache import get_cache
    cache = get_cache("test_redis_ttl")
    cache.set("k1", "value", ttl_seconds=1)
    assert cache.get("k1") == "value"
    time.sleep(1.5)
    assert cache.get("k1") is None


def test_redis_cache_permanent_entry_has_no_ttl(redis_settings):
    """Matches embedding_cache.py's usage: ttl_seconds=None must mean
    genuinely permanent, not expire-immediately."""
    from app.cache.ttl_cache import get_cache
    cache = get_cache("test_redis_permanent")
    cache.set("k1", "permanent value", ttl_seconds=None)
    time.sleep(1.2)
    assert cache.get("k1") == "permanent value"


def test_redis_cache_invalidate_removes_key(redis_settings):
    from app.cache.ttl_cache import get_cache
    cache = get_cache("test_redis_invalidate")
    cache.set("k1", "value", ttl_seconds=60)
    assert cache.invalidate("k1") is True
    assert cache.get("k1") is None
    assert cache.invalidate("k1") is False, "invalidating an already-gone key returns False, not an error"


def test_redis_cache_numpy_array_roundtrip(redis_settings):
    """The embedding cache stores numpy vectors - confirms pickle-based
    serialization actually handles this."""
    import numpy as np
    from app.cache.ttl_cache import get_cache
    cache = get_cache("test_redis_numpy")
    vec = np.array([1.5, 2.5, 3.5])
    cache.set("vec", vec, ttl_seconds=None)
    retrieved = cache.get("vec")
    assert np.array_equal(retrieved, vec)


def test_redis_cache_namespacing_prevents_key_collisions(redis_settings):
    """Two different cache names must not collide, even with the same key."""
    from app.cache.ttl_cache import get_cache
    cache_a = get_cache("namespace_a")
    cache_b = get_cache("namespace_b")
    cache_a.set("shared_key", "value_a", ttl_seconds=60)
    cache_b.set("shared_key", "value_b", ttl_seconds=60)
    assert cache_a.get("shared_key") == "value_a"
    assert cache_b.get("shared_key") == "value_b"


def test_full_webhook_invalidation_scenario_against_real_redis(redis_settings):
    """Re-runs Phase 11's core DoD test, but now against REAL Redis
    instead of the in-process cache."""
    import os, tempfile
    tmp_db = os.path.join(tempfile.gettempdir(), "test_redis_webhook.db")
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_db}"
    from app.core.config import get_settings
    get_settings.cache_clear()
    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)
    db_module.init_db()

    from app.tools.wms import seed_stock, handle_inventory_update_webhook
    from app.cache.tool_cache import get_stock_cached

    db = db_module.SessionLocal()
    seed_stock(db, sku="SKU-REDIS-1", warehouse="WH-A", on_hand_qty=10, sellable_qty=10)

    first_read = get_stock_cached(db, "SKU-REDIS-1", "WH-A")
    assert first_read[0]["sellable_qty"] == 10

    handle_inventory_update_webhook(db, sku="SKU-REDIS-1", warehouse="WH-A",
                                     new_on_hand_qty=2, new_sellable_qty=2)

    second_read = get_stock_cached(db, "SKU-REDIS-1", "WH-A")
    assert second_read[0]["sellable_qty"] == 2, "webhook invalidation must work against real Redis too"

    db.close()
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
