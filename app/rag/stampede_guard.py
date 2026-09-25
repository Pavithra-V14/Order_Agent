"""
Cache stampede protection for retrieval, a.k.a. "single-flight" or
"request coalescing" - the same pattern Go's singleflight package and
many production caching layers use. Closes a real, un-addressed gap:
if a popular query's cache entry expires (or gets invalidated) right
as real traffic hits it, every one of those concurrent requests misses
the cache at the same instant and independently hits Qdrant - the
exact same expensive work, repeated N times for N simultaneous
callers, when one real answer would have served all of them.

The fix: the FIRST caller for a given cache key becomes the "leader"
and does the real work. Every other caller for that SAME key, arriving
while the leader is still working, waits for the leader to finish and
reuses its result - instead of also hitting Qdrant.
"""
from __future__ import annotations

import threading

_inflight: dict[str, threading.Event] = {}
_inflight_lock = threading.Lock()

# How long a follower waits for the leader before giving up and doing
# its own search anyway - a safety valve. If the leader hangs or is
# unusually slow, followers shouldn't wait forever; falling back to
# doing the work themselves is always correct, just not optimally fast.
_MAX_FOLLOWER_WAIT_SECONDS = 10.0


def run_with_stampede_protection(key: str, get_cached, compute_and_cache):
    """
    key: the cache key this call is for.
    get_cached: zero-arg callable, returns the cached value or None.
    compute_and_cache: zero-arg callable that does the real work AND
        writes it to the cache - called only by the leader.

    Returns: (result, was_cache_hit: bool)
    """
    cached = get_cached()
    if cached is not None:
        return cached, True

    with _inflight_lock:
        existing_event = _inflight.get(key)
        if existing_event is None:
            my_event = threading.Event()
            _inflight[key] = my_event
            is_leader = True
        else:
            is_leader = False

    if not is_leader:
        existing_event.wait(timeout=_MAX_FOLLOWER_WAIT_SECONDS)
        cached = get_cached()
        if cached is not None:
            return cached, True
        # Leader's result isn't visible yet (timed out, or the leader's
        # write raced with this check) - falling back to doing the
        # work ourselves is always correct, just means we didn't get
        # the coalescing benefit this one time.

    try:
        result = compute_and_cache()
        return result, False
    finally:
        if is_leader:
            with _inflight_lock:
                _inflight.pop(key, None)
            my_event.set()
