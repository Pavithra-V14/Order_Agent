"""
Idempotency enforcement — architecture doc Layer 5's core reliability
mechanism, implemented for real (not just described) at the tool layer.

Every write-capable tool (refund, inventory transfer, label generation)
wraps its execution with `with_idempotency()`. Given the same
idempotency_key twice, the second call returns the first call's stored
result WITHOUT re-invoking the underlying gateway — this is what makes a
retried webhook, a duplicate case escalation, or a network-retry-after-
timeout safe rather than a duplicate-refund incident (the exact failure
mode named in Part 0's "≥3 failure modes" and tested in
tests/test_phase4_tools.py).

If the same key is reused with DIFFERENT request arguments, this raises —
silently returning a cached result for a different request would be a
worse bug than no idempotency at all.
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable, TypeVar

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.db import IdempotencyRecord

T = TypeVar("T")


def _fingerprint(args: dict) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True, default=str).encode()).hexdigest()


class IdempotencyKeyReusedWithDifferentArgs(Exception):
    pass


def with_idempotency(
    db: Session,
    idempotency_key: str,
    tool_name: str,
    request_args: dict,
    execute_fn: Callable[[], dict],
) -> tuple[dict, bool]:
    """Returns (result, was_replayed). `was_replayed=True` means
    execute_fn() was NOT called — the stored result from a prior call with
    this exact key was returned instead.

    CONTRACT: execute_fn() must NOT call db.commit() itself. It should
    stage its changes on the session (db.add(...), attribute mutations)
    and return a result dict; this function commits the staged changes
    AND the idempotency record together in one transaction. An earlier
    version of the WMS transfer tool had execute_fn() commit its own stock
    mutation before this function's own commit — under a race, both
    concurrent callers could commit their stock mutation before either's
    idempotency-record insert resolved, causing a real double-decrement
    even though the idempotency table correctly recorded only one winner.
    Single-transaction commit closes that gap: if the idempotency-record
    insert loses the race and rolls back, the staged stock mutation rolls
    back with it, atomically.

    Important honesty note: this local check-then-execute still has a
    race window (two concurrent calls can both pass the SELECT before
    either commits — see the IntegrityError handling below, which now
    correctly discards the losing transaction's staged changes too). The
    real safety net for external-system side effects (e.g. an actual
    Stripe refund) is passing this SAME idempotency_key through to the
    downstream gateway's own idempotency support (Stripe's API natively
    accepts one) — this function is a local cost-saver, atomicity
    guarantee for OUR OWN state, and audit record, not a distributed lock.
    """
    fingerprint = _fingerprint(request_args)

    existing = db.get(IdempotencyRecord, idempotency_key)
    if existing is not None:
        if existing.request_fingerprint != fingerprint:
            raise IdempotencyKeyReusedWithDifferentArgs(
                f"idempotency_key {idempotency_key!r} was already used for tool "
                f"{existing.tool_name!r} with different arguments. Refusing to "
                f"either re-execute or silently return a mismatched cached result."
            )
        return existing.result, True

    result = execute_fn()

    record = IdempotencyRecord(
        idempotency_key=idempotency_key,
        tool_name=tool_name,
        request_fingerprint=fingerprint,
        result=result,
    )
    db.add(record)
    try:
        db.commit()
    except IntegrityError:
        # Race: another concurrent call inserted a record for this exact
        # key between our SELECT and our INSERT (the multi-warehouse-style
        # race condition from the edge-case inventory, applied here to
        # writes generally, not just inventory). We already executed
        # execute_fn() once — that's an accepted cost of losing the race,
        # NOT a duplicate action from the caller's perspective, since we
        # now return the WINNING record's result, not our own execution's
        # result, keeping the "same key -> same returned result" guarantee.
        db.rollback()
        winning = db.get(IdempotencyRecord, idempotency_key)
        if winning is None:
            raise  # shouldn't happen, but don't swallow silently if it does
        from app.core.alerting import send_alert
        send_alert(db, "idempotency_collision", {
            "idempotency_key": idempotency_key, "tool_name": tool_name,
            "note": "concurrent calls raced on the same key — the losing transaction's "
                    "staged changes were rolled back and the winner's result is returned",
        })
        return winning.result, True

    return result, False
