"""
Retrieval cache - architecture doc 8.9: 5-15 min TTL. Policy documents
change infrequently, but this stays short enough that a same-day policy
correction propagates fast.
"""
from __future__ import annotations

import hashlib
import json

from app.cache.ttl_cache import get_cache

RETRIEVAL_TTL_SECONDS = 600.0


def _query_cache_key(query, as_of_date, doc_type, channel, product_category, top_k) -> str:
    basis = json.dumps({
        "query": query, "as_of_date": as_of_date, "doc_type": doc_type,
        "channel": channel, "product_category": product_category, "top_k": top_k,
    }, sort_keys=True)
    return hashlib.sha256(basis.encode()).hexdigest()


def get_cached_retrieval(query, as_of_date, doc_type=None, channel=None, product_category=None, top_k=5):
    cache = get_cache("retrieval")
    key = _query_cache_key(query, as_of_date, doc_type, channel, product_category, top_k)
    return cache.get(key)


def set_cached_retrieval(query, as_of_date, results, doc_type=None, channel=None, product_category=None, top_k=5):
    cache = get_cache("retrieval")
    key = _query_cache_key(query, as_of_date, doc_type, channel, product_category, top_k)
    cache.set(key, results, ttl_seconds=RETRIEVAL_TTL_SECONDS)


def invalidate_retrieval_cache_for_doc_type(doc_type: str) -> None:
    """Called after a new policy version is ingested. Cache keys are
    content-hashed, not prefixed by doc_type alone, so this does a full
    clear rather than a targeted invalidation - acceptable given entries
    are cheap to recompute and policy updates are rare, but noted as a
    real limitation versus a prefix-addressable key scheme."""
    cache = get_cache("retrieval")
    cache.clear()
