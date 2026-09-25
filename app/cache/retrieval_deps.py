"""
Atomic cache-dependency tracking for the retrieval cache's targeted
invalidation (app/cache/retrieval_cache.py) - replaces an earlier
get/set-based version that had a real, documented race: two searches
registering a dependency on the SAME doc_id at the same instant could
read the same starting list, and whichever wrote second would silently
overwrite the first's addition, dropping one cache key from the
dependency list until its own TTL expired it naturally.

Fixed by using Redis's native SADD/SREM - "add this one item to this
set" is a single, atomic operation with no read-modify-write step at
all. Two SADDs landing at the exact same instant both succeed; there
is no window where one can overwrite the other, because there is
never a "read the whole set, then write the whole set back" step for
two concurrent calls to race inside of.

Falls back to a thread-safe in-process set (protected by a real lock,
not a bare Python set - a bare set's method calls are NOT guaranteed
atomic under all conditions) when Redis isn't configured, matching
every other dual-backend primitive in this project.
"""
from __future__ import annotations

import threading

_in_process_deps: dict[str, set[str]] = {}
_in_process_lock = threading.Lock()

_KEY_PREFIX = "retrieval_deps:"


def _get_redis_client():
    from app.core.config import get_settings
    settings = get_settings()
    if not settings.redis_url:
        return None
    import redis as redis_lib
    return redis_lib.from_url(settings.redis_url, decode_responses=True)


def add_dependency(doc_id: str, cache_key: str, ttl_seconds: float) -> None:
    """Registers that the cache entry at cache_key used doc_id - called
    once per doc_id per cache write, from set_cached_retrieval.

    Redis path: SADD is atomic on its own; the TTL is then applied with
    EXPIRE as a second call. This leaves a real but harmless gap: if
    the process died in the instant between SADD and EXPIRE, the set
    would persist without an expiry. Not a correctness issue for this
    use case - a dependency entry that outlives its cache entry just
    means a future invalidation call includes one extra, already-
    expired cache key, and invalidating an already-gone key is a safe
    no-op (see invalidate_dependents below). Doing both atomically
    would need a Lua script or Redis's newer SADD+EXPIRE pipelining;
    not worth the complexity for a gap this benign.
    """
    client = _get_redis_client()
    if client is not None:
        redis_key = _KEY_PREFIX + doc_id
        client.sadd(redis_key, cache_key)
        client.expire(redis_key, int(ttl_seconds))
        return
    with _in_process_lock:
        _in_process_deps.setdefault(doc_id, set()).add(cache_key)


def get_dependents(doc_id: str) -> set[str]:
    """Returns every cache key currently registered as depending on
    doc_id."""
    client = _get_redis_client()
    if client is not None:
        return client.smembers(_KEY_PREFIX + doc_id)
    with _in_process_lock:
        return set(_in_process_deps.get(doc_id, set()))


def clear_dependents(doc_id: str) -> None:
    """Removes the whole dependency record for doc_id - called after
    its dependent cache entries have been invalidated, so a stale
    record doesn't linger and get checked again unnecessarily."""
    client = _get_redis_client()
    if client is not None:
        client.delete(_KEY_PREFIX + doc_id)
        return
    with _in_process_lock:
        _in_process_deps.pop(doc_id, None)


def reset_dependencies_for_tests() -> None:
    """Test helper only."""
    client = _get_redis_client()
    if client is not None:
        for key in client.keys(_KEY_PREFIX + "*"):
            client.delete(key)
    with _in_process_lock:
        _in_process_deps.clear()
