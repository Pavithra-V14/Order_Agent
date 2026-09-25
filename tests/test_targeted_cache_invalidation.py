"""
Tests for the targeted, dependency-tagged retrieval cache invalidation
in app/cache/retrieval_cache.py - replaces what used to be a full
cache.clear() on every policy deletion with the same "surrogate key"
idea real production caches use (Fastly's Surrogate-Key headers,
Varnish's bans): tag each cached entry with which doc_ids it actually
used, then invalidate only entries that used a given doc_id - leaving
every other cached search, for every unrelated document, untouched.
"""
import pytest

from app.cache.retrieval_cache import (
    get_cached_retrieval, set_cached_retrieval, invalidate_retrieval_cache_for_doc_type,
)
from app.cache.ttl_cache import reset_all_caches
from app.rag.retrieval import RetrievedChunk


@pytest.fixture(autouse=True)
def _reset_caches():
    reset_all_caches()
    yield
    reset_all_caches()


def _chunk(doc_id, text="some text"):
    return RetrievedChunk(node_id=f"node-{doc_id}", text=text, score=0.9, metadata={"doc_id": doc_id})


def test_invalidating_one_doc_id_leaves_unrelated_cached_queries_untouched():
    """THE core proof this is targeted, not a full wipe: two different
    cached queries depend on two different documents. Invalidating
    doc A must remove ONLY the query that used doc A - the query that
    used doc B must still be a cache hit afterward."""
    set_cached_retrieval("query about returns", "2025-06-15", [_chunk("DOC-A")])
    set_cached_retrieval("query about fraud", "2025-06-15", [_chunk("DOC-B")])

    assert get_cached_retrieval("query about returns", "2025-06-15") is not None
    assert get_cached_retrieval("query about fraud", "2025-06-15") is not None

    invalidate_retrieval_cache_for_doc_type("DOC-A")

    assert get_cached_retrieval("query about returns", "2025-06-15") is None, \
        "the query that used DOC-A must be invalidated"
    assert get_cached_retrieval("query about fraud", "2025-06-15") is not None, \
        "the UNRELATED query about DOC-B must still be a cache hit - this is the whole point of targeted invalidation"


def test_a_query_depending_on_multiple_docs_is_invalidated_by_any_one_of_them():
    """A single cached result can legitimately span multiple documents
    (a broad, unfiltered search). Invalidating EITHER of the documents
    it used must invalidate that cache entry - it's still correctly
    stale if any one of its underlying sources changed."""
    set_cached_retrieval("broad cross-doc query", "2025-06-15", [_chunk("DOC-X"), _chunk("DOC-Y")])
    assert get_cached_retrieval("broad cross-doc query", "2025-06-15") is not None

    invalidate_retrieval_cache_for_doc_type("DOC-Y")

    assert get_cached_retrieval("broad cross-doc query", "2025-06-15") is None, \
        "a result depending on DOC-Y must be invalidated when DOC-Y changes, even if DOC-X was also involved"


def test_invalidating_a_doc_id_nothing_ever_cached_it_is_a_safe_no_op():
    """Deleting a document that was never actually part of any cached
    result must not error and must not touch anything else."""
    set_cached_retrieval("some other query", "2025-06-15", [_chunk("DOC-UNRELATED")])

    invalidate_retrieval_cache_for_doc_type("DOC-NEVER-CACHED")

    assert get_cached_retrieval("some other query", "2025-06-15") is not None, \
        "invalidating a doc_id with no cached dependents must not affect anything"


def test_concurrent_dependency_registration_loses_nothing():
    """THE actual proof the atomic fix works: many threads registering
    a dependency on the SAME doc_id at the same time, via real
    threading (not a mock) - this is the exact scenario an earlier,
    read-modify-write version of this could lose entries under, since
    two concurrent 'read the list, append, write it back' sequences
    can overwrite each other. Using Redis's SADD (or a lock-protected
    in-process set) has no such window - every single registration
    must survive, regardless of how many land at once."""
    import threading
    from app.cache.retrieval_deps import add_dependency, get_dependents, reset_dependencies_for_tests

    reset_dependencies_for_tests()
    doc_id = "CONCURRENT-TEST-DOC"
    thread_count = 50
    expected_keys = {f"cache-key-{i}" for i in range(thread_count)}

    def _register(i):
        add_dependency(doc_id, f"cache-key-{i}", ttl_seconds=600)

    threads = [threading.Thread(target=_register, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    actual_keys = get_dependents(doc_id)
    assert actual_keys == expected_keys, (
        f"expected all {thread_count} concurrently-registered keys to survive, "
        f"but got {len(actual_keys)} - some were lost to a race"
    )
    reset_dependencies_for_tests()
