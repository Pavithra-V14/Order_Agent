"""
Embedding cache - architecture doc 8.9: "Permanent - never expires, only
invalidated when the source text itself changes (tracked via
content_hash)." Thin, deliberately permanent-TTL wrapper around TTLCache.
"""
from __future__ import annotations

import hashlib

from app.cache.ttl_cache import get_cache


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_cached_embedding(text: str):
    cache = get_cache("embeddings")
    return cache.get(_content_hash(text))


def set_cached_embedding(text: str, vector) -> None:
    cache = get_cache("embeddings")
    cache.set(_content_hash(text), vector, ttl_seconds=None)


def get_or_compute_embedding(text: str, compute_fn):
    """Cache-aside pattern: returns (vector, was_cache_hit). compute_fn()
    is only called on a genuine miss."""
    cached = get_cached_embedding(text)
    if cached is not None:
        return cached, True
    vector = compute_fn(text)
    set_cached_embedding(text, vector)
    return vector, False
