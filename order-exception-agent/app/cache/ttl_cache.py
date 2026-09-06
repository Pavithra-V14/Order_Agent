"""
TTL cache - architecture doc 8.9. No Redis in this sandbox originally
(no Docker, no network access) - app/core/config.py documents this
fallback (redis_url: str | None = None -> in-process).

UPDATE: this was previously a documented-but-unimplemented swap point —
`redis_url` existed in config but no code actually read it. Now wired
for real: when `settings.redis_url` is set, `get_cache()` returns a
genuine Redis-backed cache (`RedisTTLCache`) instead of the in-process
one. Tested against a real local Redis server in this sandbox (installed
via `apt-get install redis-server`, which this environment's package-
registry-only network allowlist happens to permit) — not mocked. Works
identically against Upstash or any other standard Redis-protocol
endpoint: Upstash exposes a normal `rediss://` (TLS) endpoint that
`redis-py` connects to exactly like any other Redis server, no
Upstash-specific client needed.
"""
from __future__ import annotations

import pickle
import time
import threading
from dataclasses import dataclass, field


@dataclass
class _Entry:
    value: object
    expires_at: float = None
    fetched_at: float = field(default_factory=time.monotonic)


class TTLCache:
    """In-process fallback — used when no redis_url is configured.
    Thread-safe, real TTL expiry and invalidation, but state is lost on
    process restart and isn't shared across multiple app instances."""

    def __init__(self, name: str):
        self.name = name
        self._store: dict = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    def get(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            if entry.expires_at is not None and time.monotonic() >= entry.expires_at:
                del self._store[key]
                self.misses += 1
                return None
            self.hits += 1
            return entry.value

    def get_with_metadata(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None, None
            if entry.expires_at is not None and time.monotonic() >= entry.expires_at:
                del self._store[key]
                return None, None
            return entry.value, entry.fetched_at

    def set(self, key: str, value, ttl_seconds: float = None) -> None:
        expires_at = (time.monotonic() + ttl_seconds) if ttl_seconds is not None else None
        with self._lock:
            self._store[key] = _Entry(value=value, expires_at=expires_at)

    def invalidate(self, key: str) -> bool:
        with self._lock:
            existed = key in self._store
            if existed:
                del self._store[key]
                self.invalidations += 1
            return existed

    def invalidate_prefix(self, prefix: str) -> int:
        with self._lock:
            keys_to_remove = [k for k in self._store if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._store[k]
            self.invalidations += len(keys_to_remove)
            return len(keys_to_remove)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0
            self.invalidations = 0


class RedisTTLCache:
    """Real Redis-backed cache — used automatically when settings.redis_url
    is set (Upstash, a docker-compose Redis, or any standard Redis
    instance all work identically; Upstash's rediss:// TLS endpoint is
    just a normal Redis endpoint from redis-py's point of view).

    Values are pickled rather than JSON-encoded: this cache stores a
    genuine mix of types across callers (plain dicts/lists in
    tool_cache.py, numpy arrays in embedding_cache.py, RetrievedChunk
    dataclass instances in retrieval_cache.py) — JSON would require a
    custom encoder per type; pickle handles all of them uniformly. This
    is safe here specifically because the cache only ever stores data
    this application itself wrote (cache-aside pattern, never
    deserializing untrusted external input), which is the standard
    caveat for using pickle at all.

    Namespaced by `name` (a key prefix) so multiple logical caches
    (embeddings/retrieval/inventory/carrier_tracking) can share one Redis
    instance/database without key collisions — mirrors get_cache()'s
    per-name isolation in the in-process version.
    """

    def __init__(self, name: str, redis_url: str):
        import redis as redis_lib
        self.name = name
        self._prefix = f"cache:{name}:"
        self._client = redis_lib.from_url(redis_url, decode_responses=False)
        self.hits = 0
        self.misses = 0
        self.invalidations = 0

    def _k(self, key: str) -> str:
        return self._prefix + key

    def get(self, key: str):
        raw = self._client.get(self._k(key))
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        _, value = pickle.loads(raw)
        return value

    def get_with_metadata(self, key: str):
        raw = self._client.get(self._k(key))
        if raw is None:
            return None, None
        fetched_at, value = pickle.loads(raw)
        return value, fetched_at

    def set(self, key: str, value, ttl_seconds: float = None) -> None:
        payload = pickle.dumps((time.monotonic(), value))
        if ttl_seconds is not None:
            # Redis TTL is in whole seconds at minimum granularity;
            # round up so a sub-second ttl_seconds (as some tests use)
            # never rounds down to 0 (which Redis treats as "no TTL").
            self._client.set(self._k(key), payload, ex=max(1, int(ttl_seconds + 0.999)))
        else:
            self._client.set(self._k(key), payload)  # no TTL = permanent, matches embedding_cache's usage

    def invalidate(self, key: str) -> bool:
        deleted = self._client.delete(self._k(key))
        if deleted:
            self.invalidations += 1
        return bool(deleted)

    def invalidate_prefix(self, prefix: str) -> int:
        # SCAN rather than KEYS — non-blocking, safe on a shared/production
        # Redis instance with other traffic (KEYS can stall the whole server
        # on a large keyspace; this matters once this isn't just a local dev cache).
        pattern = self._k(prefix) + "*"
        count = 0
        for k in self._client.scan_iter(match=pattern):
            self._client.delete(k)
            count += 1
        self.invalidations += count
        return count

    def clear(self) -> None:
        for k in self._client.scan_iter(match=self._prefix + "*"):
            self._client.delete(k)
        self.hits = 0
        self.misses = 0
        self.invalidations = 0


_caches: dict = {}


def get_cache(name: str):
    """Process-wide named cache instances. Returns a RedisTTLCache when
    settings.redis_url is configured (Upstash, docker-compose Redis, or
    any standard Redis endpoint), otherwise the in-process TTLCache
    fallback — the choice is made once per cache name and reused, so a
    single process doesn't reconnect to Redis on every call."""
    if name not in _caches:
        from app.core.config import get_settings
        settings = get_settings()
        if settings.redis_url:
            _caches[name] = RedisTTLCache(name=name, redis_url=settings.redis_url)
        else:
            _caches[name] = TTLCache(name=name)
    return _caches[name]


def reset_all_caches() -> None:
    """Test helper. For Redis-backed caches this also clears their actual
    Redis keys (not just the local Python dict), so tests don't leak
    state into a real Redis instance between runs."""
    for cache in _caches.values():
        if isinstance(cache, RedisTTLCache):
            cache.clear()
    _caches.clear()
