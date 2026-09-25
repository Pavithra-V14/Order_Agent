"""
Shared lock for all reads-then-writes of data/reindex_state.json.

Found necessary directly from a follow-up conversation: ingestion.py
already protects concurrent ingestions of DIFFERENT documents from
overwriting each other's entries in this shared file, using a
threading.Lock private to that module. But app/api/v1/policies.py's
delete_policy ALSO reads, modifies, and writes this same file (removing
a doc_id's entry, or updating its source_filename) - using no lock at
all, and even if it did, a plain threading.Lock() in a DIFFERENT module
is a different lock object entirely, so it wouldn't serialize against
ingestion's own lock anyway. A deletion and a concurrent ingestion of a
DIFFERENT document could still race and silently drop one of their
entries from the file.

Uses filelock.FileLock rather than a plain threading.Lock specifically
so this protects across PROCESSES too, not just threads within one
running server - a real gap a bare threading.Lock can't close, since
multiple worker processes each have their own separate Python memory
and would each have their own separate, disconnected Lock object.
"""
from __future__ import annotations

import os
import filelock

_REINDEX_STATE_PATH = os.path.join("data", "reindex_state.json")
_LOCK_PATH = _REINDEX_STATE_PATH + ".lock"

# A generous but bounded timeout - long enough that a normal, brief
# read-merge-write never hits it, short enough that a genuinely stuck
# lock (e.g. a crashed process that died mid-lock, though filelock's
# own lock files are released automatically when their holding process
# exits, even on a crash) fails loudly with a clear error rather than
# hanging a request forever.
_LOCK_TIMEOUT_SECONDS = 30


def reindex_state_lock() -> filelock.FileLock:
    """Returns a FileLock for the critical section that reads, merges,
    and writes data/reindex_state.json. Use as:

        with reindex_state_lock():
            state = _load_reindex_state()
            state[doc_id] = ...   # or: del state[doc_id]
            _save_reindex_state(state)

    Keep the locked section to just this read-merge-write - not the
    slow embedding/deletion work around it - so concurrent operations
    on genuinely different documents don't need to fully serialize
    against each other, only this last, fast step does.
    """
    os.makedirs(os.path.dirname(_REINDEX_STATE_PATH), exist_ok=True)
    return filelock.FileLock(_LOCK_PATH, timeout=_LOCK_TIMEOUT_SECONDS)
