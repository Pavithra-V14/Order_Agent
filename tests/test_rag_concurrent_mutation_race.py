"""
Tests for the real race condition fixed in app/rag/mutation_lock.py:
retrieval reads Qdrant, then writes its result to the retrieval cache,
in two separate steps. If a policy document deletion runs BETWEEN
those two steps, the retrieval's own cache-write can reintroduce
exactly the stale data the deletion's own cache-clear step just
removed - a real bug that would be completely invisible testing either
operation in isolation, only showing up under genuine concurrent
traffic.

These tests simulate the race DETERMINISTICALLY (mocking hybrid_search
to bump the generation counter as part of its own execution, standing
in for "a deletion runs while this read is in flight") rather than
relying on real thread timing, which would be flaky by nature.
"""
import os
import tempfile
from unittest.mock import patch

import pytest

from app.rag.mutation_lock import (
    bump_rag_generation, get_rag_generation, reset_rag_generation_for_tests,
)
from tests.test_phase12_api import client_and_db, _write_test_policy_pdf
from app.rag.retrieval import RetrievedChunk


@pytest.fixture(autouse=True)
def _reset_generation():
    reset_rag_generation_for_tests()
    yield
    reset_rag_generation_for_tests()


def test_bump_and_get_rag_generation():
    assert get_rag_generation() == 0
    assert bump_rag_generation() == 1
    assert get_rag_generation() == 1
    assert bump_rag_generation() == 2
    assert get_rag_generation() == 2


def test_retrieval_caches_normally_with_no_concurrent_mutation(client_and_db):
    """Control case - proves the fix doesn't break the ordinary,
    no-race path: a normal retrieval with nothing mutating concurrently
    must still populate the cache exactly as before."""
    _, db_module = client_and_db
    from app.rag.traced_retrieval import traced_hybrid_search
    from app.cache.retrieval_cache import get_cached_retrieval
    from app.cache.ttl_cache import reset_all_caches
    reset_all_caches()

    fake_results = [RetrievedChunk(node_id="fake-node-1", text="fake result", score=0.9, metadata={"doc_id": "TEST-DOC"})]
    db = db_module.SessionLocal()
    with patch("app.rag.traced_retrieval.hybrid_search", return_value=fake_results):
        results = traced_hybrid_search(
            db, trace_id="trace-no-race", query="test query", as_of_date="2025-06-15",
        )
    db.close()

    assert results == fake_results
    cached = get_cached_retrieval("test query", "2025-06-15")
    assert cached is not None, "a retrieval with no concurrent mutation must be cached normally"


def test_retrieval_skips_caching_when_mutation_happens_during_the_read(client_and_db):
    """THE core regression test: if bump_rag_generation() is called
    WHILE hybrid_search is executing (standing in for a real concurrent
    policy deletion), the result must NOT be written to the shared
    cache - even though it's still correctly returned to this one
    caller. Proves the fix actually closes the race rather than just
    checking the generation and forgetting to act on it."""
    _, db_module = client_and_db
    from app.rag.traced_retrieval import traced_hybrid_search
    from app.cache.retrieval_cache import get_cached_retrieval
    from app.cache.ttl_cache import reset_all_caches
    reset_all_caches()

    fake_results = [RetrievedChunk(node_id="fake-node-1", text="fake result", score=0.9, metadata={"doc_id": "TEST-DOC"})]

    def _slow_search_with_concurrent_mutation(**kwargs):
        # Simulates a policy deletion completing WHILE this retrieval's
        # own Qdrant read was in flight.
        bump_rag_generation()
        return fake_results

    db = db_module.SessionLocal()
    with patch("app.rag.traced_retrieval.hybrid_search", side_effect=_slow_search_with_concurrent_mutation):
        results = traced_hybrid_search(
            db, trace_id="trace-with-race", query="racy query", as_of_date="2025-06-15",
        )
    db.close()

    assert results == fake_results, "the in-flight caller must still get its own correct result"
    cached = get_cached_retrieval("racy query", "2025-06-15")
    assert cached is None, (
        "a result computed while a concurrent mutation was in progress must NOT be cached - "
        "caching it here would silently reintroduce exactly the stale data a deletion just removed"
    )


def test_delete_policy_bumps_generation_before_touching_anything_else(client_and_db):
    """Integration-level proof that the actual delete endpoint wires
    this in for real, not just that the primitive itself works in
    isolation."""
    client, _ = client_and_db

    doc_id = "TEST-RACE-GENERATION"
    fname = f"{doc_id}.pdf"
    path = os.path.join("data", "policies", fname)
    tmp_source = os.path.join(tempfile.gettempdir(), fname)
    _write_test_policy_pdf(tmp_source, doc_id)
    try:
        with open(tmp_source, "rb") as f:
            resp = client.post("/api/v1/policies/upload", files={"file": (fname, f, "application/pdf")})
        assert resp.status_code == 200

        generation_before = get_rag_generation()
        resp = client.delete(f"/api/v1/policies/{fname}?confirm=true")
        assert resp.status_code == 200
        assert get_rag_generation() > generation_before, (
            "deleting a policy document must bump the RAG mutation generation"
        )
    finally:
        if os.path.exists(path):
            os.remove(path)
        if os.path.exists(tmp_source):
            os.remove(tmp_source)
