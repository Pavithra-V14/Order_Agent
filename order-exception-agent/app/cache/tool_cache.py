"""
Tool/API response cache - architecture doc 8.9: very short TTL (30-60s)
PLUS event-driven invalidation, specifically to prevent the phantom-stock
(edge case 3.1) and marketplace-lag (edge case 3.4) failure modes from
serving a stale read during a live diagnosis.

Wraps app.tools.wms.get_stock and app.tools.carrier.get_tracking_status
with a cache-aside layer, and exposes invalidate_* functions that webhook
handlers call the moment new state arrives.
"""
from __future__ import annotations

from app.cache.ttl_cache import get_cache
from app.tools import wms as wms_tool
from app.tools import carrier as carrier_tool

INVENTORY_TTL_SECONDS = 45.0
TRACKING_TTL_SECONDS = 45.0


def _stock_cache_key(sku: str, warehouse: str = None) -> str:
    return f"stock:{sku}:{warehouse or 'all'}"


def get_stock_cached(db, sku: str, warehouse: str = None) -> list:
    """Cache-aside read. On a cache hit, does NOT touch the DB at all."""
    cache = get_cache("inventory")
    key = _stock_cache_key(sku, warehouse)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = wms_tool.get_stock(db, sku, warehouse)
    cache.set(key, result, ttl_seconds=INVENTORY_TTL_SECONDS)
    return result


def invalidate_stock_cache(sku: str, warehouse: str = None) -> int:
    """Called by the inventory-update webhook handler the moment new
    stock data arrives. Invalidates both the specific-warehouse key and
    the all-warehouses aggregate key."""
    cache = get_cache("inventory")
    count = 0
    if cache.invalidate(_stock_cache_key(sku, warehouse)):
        count += 1
    if cache.invalidate(_stock_cache_key(sku, None)):
        count += 1
    return count


def _tracking_cache_key(tracking_number: str) -> str:
    return f"tracking:{tracking_number}"


def get_tracking_cached(tracking_number: str) -> dict:
    cache = get_cache("carrier_tracking")
    key = _tracking_cache_key(tracking_number)
    cached = cache.get(key)
    if cached is not None:
        return cached
    gateway = carrier_tool.get_carrier_gateway()
    result = gateway.get_tracking_status(tracking_number)
    cache.set(key, result, ttl_seconds=TRACKING_TTL_SECONDS)
    return result


def invalidate_tracking_cache(tracking_number: str) -> bool:
    """Called by the carrier-status webhook handler."""
    cache = get_cache("carrier_tracking")
    return cache.invalidate(_tracking_cache_key(tracking_number))
