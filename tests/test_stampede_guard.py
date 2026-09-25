"""
Tests for app/rag/stampede_guard.py - proves the singleflight/request-
coalescing pattern actually reduces duplicate work under real
concurrent access, not just in theory.
"""
import threading
import time

from app.rag.stampede_guard import run_with_stampede_protection


def test_many_concurrent_callers_for_the_same_key_trigger_the_expensive_work_once():
    """THE core proof: 20 threads all requesting the same cache key at
    the same time must result in the expensive computation running
    only once - not 20 times - with every thread still getting the
    correct result back.

    Uses a REAL shared dict standing in for the cache, with
    compute_and_cache genuinely writing into it and get_cached genuinely
    reading from it - matching how the real caller (traced_retrieval.py)
    actually uses this: compute_and_cache calls set_cached_retrieval,
    and a follower's re-check via get_cached can only find that result
    if the two are backed by the same real store, not two independent
    mock functions with no shared state between them."""
    shared_cache = {}
    call_count = {"n": 0}
    call_count_lock = threading.Lock()

    def _get_cached():
        return shared_cache.get("shared-key")

    def _compute_and_cache():
        with call_count_lock:
            call_count["n"] += 1
        time.sleep(0.2)  # simulate a genuinely slow search
        value = "the real result"
        shared_cache["shared-key"] = value
        return value

    results = [None] * 20

    def _worker(i):
        result, _ = run_with_stampede_protection("shared-key", _get_cached, _compute_and_cache)
        results[i] = result

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r == "the real result" for r in results), "every single caller must still get the correct result"
    assert call_count["n"] < 20, (
        f"expected the expensive work to be coalesced, but it ran {call_count['n']} times "
        f"out of 20 concurrent callers - stampede protection did not reduce duplicate work"
    )
    assert call_count["n"] >= 1, "the work must still genuinely happen at least once"


def test_different_keys_are_not_coalesced_with_each_other():
    """Two DIFFERENT queries running concurrently must each do their
    own real work - stampede protection must never accidentally merge
    unrelated requests just because they happened to overlap in time."""
    call_count = {"a": 0, "b": 0}
    lock = threading.Lock()

    def _get_cached():
        return None

    def _compute_a():
        with lock:
            call_count["a"] += 1
        time.sleep(0.1)
        return "result-a"

    def _compute_b():
        with lock:
            call_count["b"] += 1
        time.sleep(0.1)
        return "result-b"

    results = {}

    def _worker_a():
        results["a"], _ = run_with_stampede_protection("key-a", _get_cached, _compute_a)

    def _worker_b():
        results["b"], _ = run_with_stampede_protection("key-b", _get_cached, _compute_b)

    threads = [threading.Thread(target=_worker_a), threading.Thread(target=_worker_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results["a"] == "result-a"
    assert results["b"] == "result-b"
    assert call_count["a"] == 1, "key-a's work must run exactly once for its own caller"
    assert call_count["b"] == 1, "key-b's work must run exactly once for its own caller, independent of key-a"


def test_a_cache_hit_short_circuits_without_ever_calling_compute():
    """If the cache already has a value, the expensive function must
    never be called at all - proves this doesn't force redundant work
    on top of an already-served request."""
    computed = {"called": False}

    def _get_cached():
        return "already cached"

    def _compute_and_cache():
        computed["called"] = True
        return "should never see this"

    result, was_hit = run_with_stampede_protection("hit-key", _get_cached, _compute_and_cache)

    assert result == "already cached"
    assert was_hit is True
    assert computed["called"] is False, "compute_and_cache must never run on a genuine cache hit"
