"""
RAG mutation generation counter — closes a real, documented race
condition: retrieval reads Qdrant, then writes its result to the
retrieval cache, in two separate steps (see traced_retrieval.py's
traced_hybrid_search). If a document deletion runs between those two
steps, the retrieval's own cache-write can reintroduce exactly the
stale data the deletion's own cache-clear step just removed — seconds
after deletion completed, invisible to anyone testing the deletion in
isolation rather than under real concurrent traffic.

Fix: every RAG-mutating operation (currently: policy document deletion
— app/api/v1/policies.py's delete_policy) bumps a generation counter
FIRST, before touching anything else. Retrieval reads the generation
before AND after its own (potentially slow) Qdrant read; if the
generation changed in between, a mutation happened concurrently and
the result is not trustworthy enough to persist into a shared cache —
it's still returned to this one caller, just never written where a
LATER, unrelated caller could read it back.

This is optimistic concurrency control, not a lock: nothing blocks or
waits, cheap enough to check on every single retrieval, and correct
regardless of how the two operations happen to interleave in time —
unlike a naive "check if a lock is currently held" approach, which has
its own race (a mutation could acquire a lock after the check but
before the read completes, and still slip through).

Backed by Redis (via INCR) when configured, matching every other cache
primitive in this project (app/cache/ttl_cache.py's dual-backend
pattern) — a real production deployment with multiple worker processes
needs a genuinely shared counter, not a per-process one; an in-process
int would let each worker believe nothing changed while a DIFFERENT
worker was mid-deletion. Falls back to a plain, thread-safe in-process
counter otherwise, which is correct within a single process (the same
honest scope every other in-process fallback in this project carries)
but does not protect across multiple separate worker processes without
Redis configured — noted here rather than silently assumed away.
"""
from __future__ import annotations

import threading

_in_process_generation = 0
_in_process_lock = threading.Lock()

_REDIS_KEY = "rag:mutation_generation"


def _get_redis_client():
    from app.core.config import get_settings
    settings = get_settings()
    if not settings.redis_url:
        return None
    import redis as redis_lib
    return redis_lib.from_url(settings.redis_url, decode_responses=True)


def bump_rag_generation() -> int:
    """Called by any operation that mutates what's in the RAG index —
    as the FIRST thing it does, before touching Qdrant, disk, or cache
    — so any retrieval already in flight is guaranteed to observe a
    changed generation by the time it finishes, regardless of exactly
    when in its own execution the mutation happened to land."""
    client = _get_redis_client()
    if client is not None:
        return client.incr(_REDIS_KEY)
    global _in_process_generation
    with _in_process_lock:
        _in_process_generation += 1
        return _in_process_generation


def get_rag_generation() -> int:
    client = _get_redis_client()
    if client is not None:
        val = client.get(_REDIS_KEY)
        return int(val) if val is not None else 0
    with _in_process_lock:
        return _in_process_generation


def reset_rag_generation_for_tests() -> None:
    """Test helper only — real code never needs to reset this."""
    client = _get_redis_client()
    if client is not None:
        client.delete(_REDIS_KEY)
    global _in_process_generation
    with _in_process_lock:
        _in_process_generation = 0
