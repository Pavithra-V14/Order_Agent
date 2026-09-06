"""
Phase 8 DoD: "The chaos test passes - a simulated payment-gateway timeout
results in exactly one refund attempt being recorded as 'pending retry,'
not a duplicate charge/refund, and the circuit breaker trips after the
configured threshold."
"""
import os
import tempfile

import pytest

@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_phase8_{os.getpid()}_{id(object())}.db")
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}"

    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)
    db_module.init_db()

    yield db_module

    try:

        if os.path.exists(tmp_path):

            os.remove(tmp_path)

    except PermissionError:

        pass  # Windows may still hold a brief lock from engine cleanup; harmless to leave a stray temp file

@pytest.fixture(autouse=True)
def reset_all():
    from app.tools.payment import reset_fake_gateway
    from app.tools.carrier import reset_fake_carrier
    from app.core.circuit_breaker import reset_all_breakers
    reset_fake_gateway()
    reset_fake_carrier()
    reset_all_breakers()
    yield

def test_chaos_permanent_gateway_failure_trips_circuit_and_stays_pending_retry(isolated_db):
    """Permanent (not transient) failure: the gateway never recovers.
    Must end in PENDING_RETRY, the circuit must trip (OPEN) after exactly
    failure_threshold real attempts, and no further calls should even
    reach the gateway once it's open (fail-fast, not endless retrying)."""
    from app.tools.payment import get_payment_gateway
    from app.agents.execution_agent import execute_resolution, ExecutionStatus
    from app.core.circuit_breaker import get_circuit_breaker, CircuitState
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy

    db = isolated_db.SessionLocal()
    gateway = get_payment_gateway()
    gateway.seed_transaction("pi_chaos_1", amount_usd=100.0)
    gateway.inject_transient_failures(999)

    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=42.0, confidence=0.95,
        reasoning="Standard refund for a chaos test scenario.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )

    result = execute_resolution(
        db, case_id="case-chaos-1", decision=decision, payment_intent_id="pi_chaos_1",
        max_retries=5,
    )

    assert result.status == ExecutionStatus.PENDING_RETRY, f"Expected pending_retry, got {result.status}"

    breaker = get_circuit_breaker("payment", failure_threshold=3, reset_timeout_seconds=30.0)
    assert breaker.state == CircuitState.OPEN, "Circuit must have tripped OPEN after 3 consecutive failures"
    assert breaker.call_attempts == 3, (
        f"Expected exactly 3 real attempts to reach the breaker before it opened and started "
        f"failing fast, got {breaker.call_attempts}"
    )
    assert gateway.refund_call_count == 3, (
        f"Expected exactly 3 real gateway calls (no wasted calls after the circuit opened), "
        f"got {gateway.refund_call_count}"
    )
    from app.core.db import IdempotencyRecord
    record = db.get(IdempotencyRecord, "case-chaos-1:refund")
    assert record is None, "No idempotency record should exist - every attempt failed, nothing succeeded"
    db.close()

def test_chaos_transient_failure_recovers_and_retry_does_not_duplicate(isolated_db):
    """The recovery half: gateway fails twice (transient) then succeeds on
    the 3rd attempt. Must end EXECUTED, with exactly 3 real gateway calls.
    A SECOND call to execute_resolution with the same case/action
    afterward must NOT issue a second refund."""
    from app.tools.payment import get_payment_gateway
    from app.agents.execution_agent import execute_resolution, ExecutionStatus
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy

    db = isolated_db.SessionLocal()
    gateway = get_payment_gateway()
    gateway.seed_transaction("pi_chaos_2", amount_usd=100.0)
    gateway.inject_transient_failures(2)

    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=42.0, confidence=0.95,
        reasoning="Standard refund for a transient-failure recovery test.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )

    result1 = execute_resolution(db, case_id="case-chaos-2", decision=decision, payment_intent_id="pi_chaos_2")
    assert result1.status == ExecutionStatus.EXECUTED
    assert result1.attempts_made == 3
    assert gateway.refund_call_count == 3

    result2 = execute_resolution(db, case_id="case-chaos-2", decision=decision, payment_intent_id="pi_chaos_2")
    assert result2.status == ExecutionStatus.EXECUTED
    assert result2.result["id"] == result1.result["id"], "replay must return the SAME refund id"
    assert gateway.refund_call_count == 3, (
        f"A second execute_resolution call with the same idempotency key must NOT issue "
        f"another real refund; expected call_count to stay at 3, got {gateway.refund_call_count}"
    )
    db.close()

def test_circuit_resets_to_half_open_after_timeout(isolated_db):
    """The other half of the circuit breaker's contract: OPEN isn't
    permanent - after reset_timeout_seconds, the next call is allowed
    through again (HALF_OPEN), not failed fast forever."""
    import time
    from app.core.circuit_breaker import CircuitBreaker, CircuitState

    breaker = CircuitBreaker(name="test-quick-reset", failure_threshold=2, reset_timeout_seconds=0.05)

    def always_fails():
        raise RuntimeError("boom")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(always_fails)
    assert breaker.state == CircuitState.OPEN

    time.sleep(0.1)
    assert breaker.state == CircuitState.HALF_OPEN, "circuit should allow a trial call again after the reset timeout"

def test_verification_confirms_a_real_completed_refund(isolated_db):
    from app.tools.payment import get_payment_gateway
    from app.agents.execution_agent import execute_resolution
    from app.agents.verification_agent import verify_refund, VerificationStatus
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy

    db = isolated_db.SessionLocal()
    gateway = get_payment_gateway()
    gateway.seed_transaction("pi_verify_1", amount_usd=50.0)
    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=25.0, confidence=0.95,
        reasoning="Standard refund for verification test.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )
    exec_result = execute_resolution(db, case_id="case-verify-1", decision=decision, payment_intent_id="pi_verify_1")

    verification = verify_refund(db, exec_result.idempotency_key)
    assert verification.status == VerificationStatus.VERIFIED
    db.close()

def test_verification_fails_when_no_record_exists(isolated_db):
    from app.agents.verification_agent import verify_refund, VerificationStatus

    db = isolated_db.SessionLocal()
    verification = verify_refund(db, "case-nonexistent:refund")
    assert verification.status == VerificationStatus.VERIFICATION_FAILED
    db.close()

def test_verification_reship_not_yet_verifiable_before_carrier_scan(isolated_db):
    from app.agents.execution_agent import execute_resolution
    from app.agents.verification_agent import verify_reship, VerificationStatus
    from app.guardrails.schema import ResolutionDecision, ResolutionAction

    db = isolated_db.SessionLocal()
    decision = ResolutionDecision(
        action=ResolutionAction.RESHIP, amount_usd=0.0, confidence=0.9,
        reasoning="Reshipping the unfulfillable item.",
    )
    exec_result = execute_resolution(db, case_id="case-reship-1", decision=decision, order_id="ORD-RESHIP-1")

    verification = verify_reship(db, exec_result.idempotency_key)
    assert verification.status == VerificationStatus.NOT_YET_VERIFIABLE
    db.close()

def test_comms_sends_refund_notification_with_correct_amount():
    from app.tools.notification import reset_sent_log, get_sent_notifications
    from app.agents.comms_workflow import send_case_notification

    reset_sent_log()
    send_case_notification("CUST-1", "resolved_refund", amount=42.50)
    sent = get_sent_notifications("CUST-1")
    assert len(sent) == 1
    assert "42.50" in sent[0]["body"]

def test_comms_unknown_event_raises():
    from app.agents.comms_workflow import send_case_notification

    with pytest.raises(ValueError):
        send_case_notification("CUST-1", "not_a_real_event")
