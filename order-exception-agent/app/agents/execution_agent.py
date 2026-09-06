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
from app.guardrails.schema import ResolutionDecision, ResolutionAction
from app.tools import payment as payment_tool, wms as wms_tool, carrier as carrier_tool


class ExecutionStatus(str, Enum):
    EXECUTED = "executed"
    PENDING_RETRY = "pending_retry"
    NO_ACTION_NEEDED = "no_action_needed"


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    result: dict = field(default_factory=dict)
    idempotency_key: str = ""
    attempts_made: int = 0
    error: str = ""


def _derive_idempotency_key(case_id: str, action: ResolutionAction) -> str:
    """Deterministic per case+action-type - a retry of the SAME case's
    SAME action always reuses this key, so a retried execution attempt
    never double-executes, per Layer 5."""
    return f"{case_id}:{action.value}"


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
    idempotency_key = _derive_idempotency_key(case_id, decision.action)

    if decision.action == ResolutionAction.DENY:
        return ExecutionResult(status=ExecutionStatus.NO_ACTION_NEEDED, idempotency_key=idempotency_key)

    if decision.action in (ResolutionAction.REFUND, ResolutionAction.PARTIAL_CREDIT):
        breaker = get_circuit_breaker("payment", failure_threshold=3, reset_timeout_seconds=30.0)
        gateway = payment_tool.get_payment_gateway()

        attempts = 0
        last_error = ""
        for attempt in range(1, max_retries + 1):
            attempts = attempt
            try:
                result = breaker.call(lambda: gateway.issue_refund(
                    db, payment_intent_id, decision.amount_usd, idempotency_key
                ))
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
                if attempt < max_retries:
                    time.sleep(retry_backoff_seconds)
                continue

        return ExecutionResult(status=ExecutionStatus.PENDING_RETRY, idempotency_key=idempotency_key,
                                attempts_made=attempts, error=last_error)

    if decision.action == ResolutionAction.RESHIP:
        breaker = get_circuit_breaker("carrier", failure_threshold=3, reset_timeout_seconds=30.0)
        gateway = carrier_tool.get_carrier_gateway()
        try:
            result = breaker.call(lambda: gateway.generate_return_label(db, order_id, idempotency_key))
            return ExecutionResult(status=ExecutionStatus.EXECUTED, result=result,
                                    idempotency_key=idempotency_key, attempts_made=1)
        except CircuitOpenError as e:
            from app.core.alerting import send_alert
            send_alert(db, "circuit_breaker_trip", {"case_id": case_id, "dependency": "carrier", "error": str(e)})
            return ExecutionResult(status=ExecutionStatus.PENDING_RETRY, idempotency_key=idempotency_key,
                                    attempts_made=1, error=str(e))
        except Exception as e:
            return ExecutionResult(status=ExecutionStatus.PENDING_RETRY, idempotency_key=idempotency_key,
                                    attempts_made=1, error=str(e))

    return ExecutionResult(status=ExecutionStatus.NO_ACTION_NEEDED, idempotency_key=idempotency_key)
