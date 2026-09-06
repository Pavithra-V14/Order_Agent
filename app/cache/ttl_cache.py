"""
TTL cache - architecture doc 8.9. No Redis in this sandbox (no Docker, no
network access) - app/core/config.py already documents this fallback
(redis_url: str | None = None -> in-process). This module IS that
fallback: a real, working in-process TTL cache behind the same
get/set/delete interface a Redis-backed cache would expose, so swapping
to Redis later is a class-swap in get_cache(), not a rewrite of every
call site.
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field


@dataclass
class _Entry:
    value: object
    expires_at: float = None
    fetched_at: float = field(default_factory=time.monotonic)


class TTLCache:
    """Thread-safe in-process cache with per-entry TTL. Real behavior:
    entries genuinely expire, get() genuinely returns None past expiry,
    and invalidate() genuinely removes an entry before its TTL would
    otherwise have done so - the whole point for event-driven invalidation."""

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
        """Returns (value, fetched_at) or (None, None) - fetched_at lets a
        caller decide "this is stale for MY purposes" per 8.9's
        stale-data-detection note."""
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
        """Event-driven invalidation - actively removes a key rather than
        waiting out its TTL."""
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


_caches: dict = {}


def get_cache(name: str) -> TTLCache:
    """Process-wide named cache instances so different subsystems don't
    collide on keys and each has independent hit/miss stats."""
    if name not in _caches:
        _caches[name] = TTLCache(name=name)
    return _caches[name]


def reset_all_caches() -> None:
    _caches.clear()
