"""
Tests for a real gap found during a direct audit: app/cache/tool_cache.py's
get_stock_cached() and get_tracking_cached() were fully implemented,
and their invalidate_* counterparts were correctly wired into webhook
handlers - but nothing ever actually CALLED get_stock_cached/
get_tracking_cached in the first place. Every real diagnosis run hit
the database/gateway directly regardless, making the cache-aside layer
exist in name only. Fixed by wiring app/agents/diagnosis_agent.py's
_fetch_inventory/_fetch_carrier to use the cached versions.
"""
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_tool_cache_{os.getpid()}_{id(object())}.db")
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
def reset_caches():
    from app.cache.ttl_cache import reset_all_caches
    reset_all_caches()
    yield
    reset_all_caches()


def test_diagnosis_inventory_lookup_actually_uses_the_cache(isolated_db):
    """THE regression test: calling _fetch_inventory twice for the same
    SKU within the TTL window must hit the real wms.get_stock only
    ONCE, not twice — proving the cache is genuinely being read from
    and written to, not bypassed."""
    from app.tools.wms import seed_stock
    import app.tools.wms as wms_module
    from app.agents.diagnosis_agent import _fetch_inventory

    db = isolated_db.SessionLocal()
    seed_stock(db, sku="SKU-CACHE-TEST", warehouse="WH-A", on_hand_qty=10, sellable_qty=8)

    call_count = {"n": 0}
    real_get_stock = wms_module.get_stock

    def counting_get_stock(*args, **kwargs):
        call_count["n"] += 1
        return real_get_stock(*args, **kwargs)

    wms_module.get_stock = counting_get_stock
    try:
        _fetch_inventory(db, [{"sku": "SKU-CACHE-TEST"}])
        _fetch_inventory(db, [{"sku": "SKU-CACHE-TEST"}])
    finally:
        wms_module.get_stock = real_get_stock

    assert call_count["n"] == 1, (
        f"the real wms.get_stock must only be called ONCE across two lookups within the cache TTL — "
        f"got {call_count['n']} real calls, meaning the cache-aside layer isn't actually being used"
    )
    db.close()


def test_diagnosis_carrier_lookup_actually_uses_the_cache(isolated_db):
    """Same regression, for carrier tracking lookups."""
    from app.tools.carrier import get_carrier_gateway
    from app.agents.diagnosis_agent import _fetch_carrier

    db = isolated_db.SessionLocal()
    gateway = get_carrier_gateway()
    gateway.seed_tracking("TRACK-CACHE-TEST", "delivered")

    call_count = {"n": 0}
    real_get_tracking_status = gateway.get_tracking_status

    def counting_get_tracking_status(*args, **kwargs):
        call_count["n"] += 1
        return real_get_tracking_status(*args, **kwargs)

    gateway.get_tracking_status = counting_get_tracking_status
    try:
        _fetch_carrier("TRACK-CACHE-TEST", db=db, case_id="case-cache-test-1")
        _fetch_carrier("TRACK-CACHE-TEST", db=db, case_id="case-cache-test-2")
    finally:
        gateway.get_tracking_status = real_get_tracking_status

    assert call_count["n"] == 1, (
        f"the real gateway.get_tracking_status must only be called ONCE across two lookups within "
        f"the cache TTL — got {call_count['n']} real calls"
    )
    db.close()


def test_carrier_cache_key_distinguishes_by_carrier_parameter(isolated_db):
    """THE regression test proving the carrier parameter (needed for
    Shippo's test-mode magic tokens like SHIPPO_DELIVERED, which only
    mean something under carrier="shippo") is genuinely part of the
    cache key — the same tracking_number under a different carrier
    must NOT return a stale, wrongly-cached result."""
    from app.cache.tool_cache import get_tracking_cached
    from app.tools.carrier import get_carrier_gateway

    gateway = get_carrier_gateway()
    gateway.seed_tracking("SAME-NUMBER-DIFFERENT-CARRIER", "delivered")

    calls = []
    real_get_tracking_status = gateway.get_tracking_status

    def recording_get_tracking_status(tracking_number, carrier=None):
        calls.append(carrier)
        return real_get_tracking_status(tracking_number, carrier=carrier)

    gateway.get_tracking_status = recording_get_tracking_status
    try:
        get_tracking_cached("SAME-NUMBER-DIFFERENT-CARRIER", carrier=None)
        get_tracking_cached("SAME-NUMBER-DIFFERENT-CARRIER", carrier="shippo")
    finally:
        gateway.get_tracking_status = real_get_tracking_status

    assert calls == [None, "shippo"], (
        "a different carrier parameter for the same tracking_number must be treated as a genuinely "
        "different cache entry, not silently served from a stale, wrongly-carrier'd cache hit"
    )
