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


def query_cache_key(query, as_of_date, doc_type, channel, product_category, top_k) -> str:
    basis = json.dumps({
        "query": query, "as_of_date": as_of_date, "doc_type": doc_type,
        "channel": channel, "product_category": product_category, "top_k": top_k,
    }, sort_keys=True)
    return hashlib.sha256(basis.encode()).hexdigest()


def get_cached_retrieval(query, as_of_date, doc_type=None, channel=None, product_category=None, top_k=5):
    cache = get_cache("retrieval")
    key = query_cache_key(query, as_of_date, doc_type, channel, product_category, top_k)
    return cache.get(key)


def set_cached_retrieval(query, as_of_date, results, doc_type=None, channel=None, product_category=None, top_k=5):
    cache = get_cache("retrieval")
    key = query_cache_key(query, as_of_date, doc_type, channel, product_category, top_k)
    cache.set(key, results, ttl_seconds=RETRIEVAL_TTL_SECONDS)

    # Records which doc_ids this specific cached result actually used -
    # a real, targeted alternative to the full cache.clear() this used
    # to require on every deletion: tag each entry with what it
    # depends on (the same idea as Fastly/Varnish's "surrogate keys" in
    # real production HTTP caches), so invalidation can later remove
    # ONLY the entries that actually used a given document. Uses
    # add_dependency's atomic Redis SADD - not a plain read-modify-
    # write - specifically because an earlier, simpler version of this
    # had a real, confirmed race: two concurrent writes to the same
    # doc_id's dependency list could overwrite each other, silently
    # dropping one cache key from the list.
    from app.cache.retrieval_deps import add_dependency
    doc_ids = {r.metadata.get("doc_id") for r in results if r.metadata.get("doc_id")}
    for doc_id in doc_ids:
        add_dependency(doc_id, key, RETRIEVAL_TTL_SECONDS)


def invalidate_retrieval_cache_for_doc_type(doc_id: str) -> None:
    """Called after a policy document is deleted or superseded.

    Targeted, not a full clear - a real fix requested and built
    directly from a conversation about how real production caches
    (Fastly's Surrogate-Key headers, Varnish's bans) solve exactly
    this: tag each cached entry with what it depends on, then
    invalidate by that tag instead of wiping everything. Only cache
    entries that actually used THIS doc_id are removed; every other
    cached search - for every other, unrelated document - is left
    completely untouched, unlike the full cache.clear() this replaced.
    """
    from app.cache.retrieval_deps import get_dependents, clear_dependents
    cache = get_cache("retrieval")
    for key in get_dependents(doc_id):
        cache.invalidate(key)
    clear_dependents(doc_id)
