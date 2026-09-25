"""
Execution Agent - architecture doc Layer 5 / Part 3 topology. A
deterministic WORKFLOW (not an agent loop): given an AUTO_EXECUTE-routed
ResolutionDecision, calls the correct write tool with a derived
idempotency key, wrapped in a circuit breaker + bounded retry.

Critical design point: a transient failure (timeout, circuit open) is NOT
the same outcome as "execution failed" - it means the case goes to
PENDING_RETRY, to be retried later (by a background worker in a full
deployment), not silently dropped or treated as a permanent failure. This
is what the chaos test proves.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from sqlalchemy.orm import Session

from app.core.circuit_breaker import get_circuit_breaker, CircuitOpenError
from app.core.tracing import record_tool_call
from app.guardrails.schema import ResolutionDecision, ResolutionAction
from app.tools import payment as payment_tool, wms as wms_tool, carrier as carrier_tool


class ExecutionStatus(str, Enum):
    EXECUTED = "executed"
    PENDING_RETRY = "pending_retry"      # transient: dependency down / timed out - safe to retry later
    FAILED = "failed"                    # permanent: retrying the same request can never succeed
    NO_ACTION_NEEDED = "no_action_needed"


# Exception class NAMES (so the fake gateway path needs no stripe import)
# that describe the REQUEST, not the dependency's health: retrying them is
# waste, and counting them against the circuit breaker lets one bad
# request block every other customer (observed against real Stripe: an
# "already been refunded" error, retried 3x, opened the payment circuit).
_PERMANENT_ERROR_NAMES = frozenset({
    "InvalidRequestError", "CardError", "AuthenticationError", "PermissionError",
    "IdempotencyError", "IdempotencyKeyReusedWithDifferentArgs", "ValueError", "KeyError",
})


def _is_transient(e: Exception) -> bool:
    return type(e).__name__ not in _PERMANENT_ERROR_NAMES


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    result: dict = field(default_factory=dict)
    idempotency_key: str = ""
    attempts_made: int = 0
    error: str = ""


def _derive_idempotency_key(case_id: str, action: ResolutionAction, payment_intent_id: str = None,
                             order_id: str = None, amount_usd: float = None) -> str:
    """Deterministic per case+action+underlying-request-parameters — a
    genuine RETRY of the SAME operation (same case, same action, same
    payment_intent_id/order_id/amount) always reuses this key, so it
    never double-executes, per Layer 5. Retrying with the exact same
    parameters is the only thing this key represents "the same
    request" for.

    Found and fixed a real gap here: this previously derived the key
    from case_id+action ALONE, with zero awareness of the underlying
    payment_intent_id or amount. A genuinely NEW operation for the same
    case (e.g. re-processing a case against a freshly created payment,
    which is exactly what happens when a demo script creates a new
    Stripe test payment each run but reuses the same case_id) collided
    with the OLD key's already-used Stripe idempotency state — Stripe
    correctly rejected it with "Keys for idempotent requests can only
    be used with the same parameters they were first used with,"
    since from Stripe's perspective, the same key WAS reused for a
    genuinely different request. Incorporating the actual resource
    identifiers into the key means a genuinely different underlying
    request naturally gets a different key, while a true retry of the
    identical request still correctly reuses the same one.
    """
    # case_id is deliberately NOT part of the key when the underlying
    # resource is known: two cases opened for the same order (a duplicate
    # webhook delivery) must collapse to ONE refund, not two. Confirmed
    # against real Stripe test mode: two 50% partial credits on one
    # payment both succeeded under case-scoped keys.
    parts = [action.value]
    if payment_intent_id:
        parts.append(payment_intent_id)
    if order_id:
        parts.append(order_id)
    if not payment_intent_id and not order_id:
        parts.insert(0, case_id)
    if amount_usd is not None:
        parts.append(f"{amount_usd:.2f}")
    return ":".join(parts)


def execute_resolution(
    db: Session,
    case_id: str,
    decision: ResolutionDecision,
    payment_intent_id: str = None,
    order_id: str = None,
    max_retries: int = 3,
    retry_backoff_seconds: float = 0.01,
) -> ExecutionResult:
    """Executes the approved decision. Never assumes success from a single
    attempt succeeding at the HTTP-call level alone - the underlying tool
    already enforces idempotency, so this function's retry loop is safe
    to retry blindly on transient failures."""
    idempotency_key = _derive_idempotency_key(
        case_id, decision.action, payment_intent_id=payment_intent_id,
        order_id=order_id, amount_usd=decision.amount_usd,
    )

    if decision.action == ResolutionAction.DENY:
        return ExecutionResult(status=ExecutionStatus.NO_ACTION_NEEDED, idempotency_key=idempotency_key)

    if decision.action in (ResolutionAction.REFUND, ResolutionAction.PARTIAL_CREDIT):
        if not payment_intent_id:
            # Fail immediately and clearly, rather than attempting a
            # call that can NEVER succeed. Found from a real production
            # run: a case with no payment_intent_id at all (a real Groq
            # LLM diagnosed "payment_issue: missing payment information"
            # from the ABSENCE of a payment record, unlike
            # FakeLLMClient's deterministic planner, which only ever
            # reaches this root cause when an actual declined
            # transaction exists) still routed to REFUND — and
            # execute_resolution wasted 3 real retries against Stripe,
            # each failing identically with "One of the following
            # params should be provided for this request: payment_intent
            # or charge," since there was never a payment_intent_id to
            # refund in the first place. A missing precondition is a
            # deterministic failure, not a transient one — retrying it
            # is pure waste, and the real error is buried under three
            # copies of the same unhelpful Stripe message instead of one
            # clear one.
            return ExecutionResult(
                status=ExecutionStatus.FAILED, idempotency_key=idempotency_key, attempts_made=0,
                error=f"Cannot execute {decision.action.value} - no payment_intent_id is associated "
                      f"with this case. A refund requires a real payment to refund against; this "
                      f"case has none on record, which is itself worth investigating rather than "
                      f"retried blindly.",
            )
        breaker = get_circuit_breaker("payment", failure_threshold=3, reset_timeout_seconds=30.0)
        gateway = payment_tool.get_payment_gateway()

        attempts = 0
        last_error = ""
        for attempt in range(1, max_retries + 1):
            attempts = attempt
            try:
                result = breaker.call(lambda: record_tool_call(
                    db, case_id, "payment.issue_refund", True, gateway.issue_refund,
                    db, payment_intent_id, decision.amount_usd, idempotency_key,
                ), is_failure=_is_transient)
                return ExecutionResult(status=ExecutionStatus.EXECUTED, result=result,
                                        idempotency_key=idempotency_key, attempts_made=attempts)
            except CircuitOpenError as e:
                from app.core.alerting import send_alert
                send_alert(db, "circuit_breaker_trip", {
                    "case_id": case_id, "dependency": "payment", "attempt": attempts, "error": str(e),
                })
                return ExecutionResult(status=ExecutionStatus.PENDING_RETRY, idempotency_key=idempotency_key,
                                        attempts_made=attempts, error=str(e))
            except Exception as e:
                last_error = str(e)
                if not _is_transient(e):
                    return ExecutionResult(status=ExecutionStatus.FAILED, idempotency_key=idempotency_key,
                                            attempts_made=attempts, error=last_error)
                if attempt < max_retries:
                    time.sleep(retry_backoff_seconds * (2 ** (attempt - 1)))
                continue

        return ExecutionResult(status=ExecutionStatus.PENDING_RETRY, idempotency_key=idempotency_key,
                                attempts_made=attempts, error=last_error)

    if decision.action == ResolutionAction.RESHIP:
        breaker = get_circuit_breaker("carrier", failure_threshold=3, reset_timeout_seconds=30.0)
        gateway = carrier_tool.get_carrier_gateway()
        try:
            result = breaker.call(lambda: record_tool_call(
                db, case_id, "carrier.generate_return_label", True,
                gateway.generate_return_label, db, order_id, idempotency_key,
            ), is_failure=_is_transient)
            return ExecutionResult(status=ExecutionStatus.EXECUTED, result=result,
                                    idempotency_key=idempotency_key, attempts_made=1)
        except CircuitOpenError as e:
            from app.core.alerting import send_alert
            send_alert(db, "circuit_breaker_trip", {"case_id": case_id, "dependency": "carrier", "error": str(e)})
            return ExecutionResult(status=ExecutionStatus.PENDING_RETRY, idempotency_key=idempotency_key,
                                    attempts_made=1, error=str(e))
        except Exception as e:
            status = ExecutionStatus.PENDING_RETRY if _is_transient(e) else ExecutionStatus.FAILED
            return ExecutionResult(status=status, idempotency_key=idempotency_key,
                                    attempts_made=1, error=str(e))

    return ExecutionResult(status=ExecutionStatus.NO_ACTION_NEEDED, idempotency_key=idempotency_key)
