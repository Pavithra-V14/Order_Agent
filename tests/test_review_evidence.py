"""
Regression tests for the live-path review findings (R1-R5, R10 and the
shared-circuit-breaker cascade). Each of these reproduced a real defect
before its fix - R2, R4 and the breaker cascade were also confirmed
against real Stripe test mode - and now asserts the corrected behaviour.
Runs on the project's fake gateways/LLM (tests/conftest.py strips real
credentials), so no real Stripe/LLM call is made.
"""
import importlib
import os
import tempfile
from datetime import datetime, timezone, timedelta

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.agents.llm_client import BaseLLMClient, DiagnosisStepPlan


@pytest.fixture
def env():
    tmp_db = os.path.join(tempfile.gettempdir(), f"test_review_evidence_{os.getpid()}_{id(object())}.db")
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_db}"
    os.environ["AUTH_ENABLED"] = "false"
    from app.core.config import get_settings
    get_settings.cache_clear()
    import app.core.db as db_module
    importlib.reload(db_module)
    db_module.init_db()
    from app.tools.payment import reset_fake_gateway
    from app.tools.carrier import reset_fake_carrier
    from app.cache.ttl_cache import reset_all_caches
    from app.workers.job_queue import reset_job_queue
    from app.core.circuit_breaker import reset_all_breakers
    reset_fake_gateway(); reset_fake_carrier(); reset_all_caches(); reset_job_queue(); reset_all_breakers()
    yield db_module
    os.environ.pop("AUTH_ENABLED", None)
    get_settings.cache_clear()


class NeverConcludesLLM(BaseLLMClient):
    """A real LLM that never reaches 'conclude' (the README documents a
    real Groq model calling check_carrier 7x)."""
    def plan_next_diagnosis_step(self, case_context, findings_so_far):
        return DiagnosisStepPlan(action="check_order", reasoning="checking again")

    def assess_fraud_risk(self, case_context, customer_risk_profile):
        return {"risk_score": 0.0, "flag": False, "reasons": []}


class BrokenLLM(BaseLLMClient):
    """An LLM that is down, or returns output missing required keys."""
    def plan_next_diagnosis_step(self, case_context, findings_so_far):
        raise RuntimeError("Circuit 'llm_groq' is OPEN")

    def assess_fraud_risk(self, case_context, customer_risk_profile):
        return {"flag": "false"}      # no risk_score, string flag


def _seed(db_module, order_id, amount, pi, stock=5):
    from app.tools.oms import create_order
    from app.tools.payment import get_payment_gateway
    db = db_module.SessionLocal()
    get_payment_gateway().seed_transaction(pi, amount_usd=amount)
    create_order(db, order_id=order_id, customer_id=f"CUST-{order_id}", channel="direct", status="paid",
                 total_amount_usd=amount, purchase_date=datetime.now(timezone.utc) - timedelta(days=3),
                 line_items=[{"sku": f"SKU-{order_id}", "category": "apparel", "qty": 1, "price": amount}],
                 payment_intent_id=pi)
    db.add(db_module.MockInventoryRecord(sku=f"SKU-{order_id}", warehouse="WH-1",
                                         on_hand_qty=stock, sellable_qty=stock))
    db.commit()
    return db


def _new_case(db_module, db, case_id, order_id):
    db.add(db_module.ExceptionCase(id=case_id, order_id=order_id, customer_id=f"CUST-{order_id}",
                                   channel="direct", exception_type="return",
                                   state=db_module.CaseState.DETECTED))
    db.commit()


def _pipeline(db, case_id, order_id, amount, pi, llm):
    from app.agents.orchestrator import run_full_case_pipeline
    return run_full_case_pipeline(
        db, case_id=case_id, order_id=order_id, customer_id=f"CUST-{order_id}", order_amount_usd=amount,
        auto_execute_confidence_threshold=0.90, auto_execute_value_ceiling_usd=50.0,
        payment_intent_id=pi, retrieved_policy_doc_id="RET-POLICY-2025-A", retrieved_policy_version="1",
        llm=llm,
    )


# ---------------------------------------------------------------- R1
def test_R1_inconclusive_diagnosis_escalates_and_moves_no_money(env):
    from app.tools.payment import get_payment_gateway
    db = _seed(env, "ORD-R1", 30.0, "pi_r1")
    _new_case(env, db, "case-r1", "ORD-R1")

    result = _pipeline(db, "case-r1", "ORD-R1", 30.0, "pi_r1", NeverConcludesLLM())

    assert result["diagnosis"]["diagnosis_root_causes"][0].startswith("diagnosis_incomplete")
    assert result["routing"] == "escalate"
    assert get_payment_gateway().refund_call_count == 0
    db.expire_all()
    case = db.get(env.ExceptionCase, "case-r1")
    assert case.state == env.CaseState.ESCALATED
    assert case.resolution_decision["requires_human_review"] is True


def test_R1_empty_or_timed_out_diagnosis_never_proposes_a_refund():
    from app.agents.resolution_policy_workflow import propose_resolution_decision
    for causes in ([], ["diagnosis_timeout: wall-clock limit reached"],
                   ["diagnosis_incomplete: step ceiling reached before concluding"]):
        d = propose_resolution_decision(diagnosis_root_causes=causes, inventory_result={}, order_amount_usd=25.0,
                                        retrieved_policy_doc_id="RET-POLICY-2025-A")
        assert d.action.value == "deny" and d.amount_usd == 0.0 and d.requires_human_review, causes


# ---------------------------------------------------------------- R2
def test_R2_duplicate_webhook_opens_one_case_and_one_refund(env):
    from app.workers import handlers
    from app.agents.execution_agent import execute_resolution
    from app.guardrails.schema import ResolutionDecision
    from app.tools.payment import get_payment_gateway
    _seed(env, "ORD-R2", 20.0, "pi_r2").close()

    payload = {"order_id": "ORD-R2", "new_status": "return_requested"}
    first = handlers.handle_oms_webhook(payload)
    second = handlers.handle_oms_webhook(payload)      # provider retry, no event id

    db = env.SessionLocal()
    cases = db.query(env.ExceptionCase).filter_by(order_id="ORD-R2").all()
    assert len(cases) == 1
    assert second["case_created"] is None and second["existing_open_case"] == first["case_created"]

    # Even if two cases DID exist for one order, money keys are order-scoped.
    decision = ResolutionDecision(action="partial_credit", amount_usd=10.0, confidence=0.93,
                                  reasoning="Duplicate-delivery regression test.")
    before = get_payment_gateway().refund_call_count
    r1 = execute_resolution(db, "case-a", decision, payment_intent_id="pi_r2", order_id="ORD-R2")
    r2 = execute_resolution(db, "case-b", decision, payment_intent_id="pi_r2", order_id="ORD-R2")
    assert r1.idempotency_key == r2.idempotency_key
    assert get_payment_gateway().refund_call_count - before == 1


def test_R2_same_event_id_is_processed_once(env):
    from app.workers import handlers
    _seed(env, "ORD-R2E", 20.0, "pi_r2e").close()
    payload = {"order_id": "ORD-R2E", "new_status": "payment_failed", "event_id": "evt_123"}
    handlers.handle_oms_webhook(payload)
    again = handlers.handle_oms_webhook(payload)
    assert again["duplicate_event"] is True


def test_R2_delivery_exception_is_filed_as_carrier_not_return(env):
    from app.workers import handlers
    _seed(env, "ORD-R2C", 20.0, "pi_r2c").close()
    out = handlers.handle_oms_webhook({"order_id": "ORD-R2C", "new_status": "delivery_exception"})
    db = env.SessionLocal()
    assert db.get(env.ExceptionCase, out["case_created"]).exception_type == "carrier"


# ---------------------------------------------------------------- R3
@pytest.mark.parametrize("llm_root_cause, payment_status", [
    ("payment_issue: none found, payment looks fine", "succeeded"),
    ("inventory_issue: none, all SKUs in stock", None),
    ("customer_changed_mind: no system fault", None),
])
def test_R3_unsupported_or_unrecognized_causes_escalate(llm_root_cause, payment_status):
    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow
    r = run_resolution_policy_workflow(
        diagnosis_root_causes=[llm_root_cause], inventory_result={"any_shortfall": False},
        order_amount_usd=40.0, fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90, auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A", retrieved_policy_version="1",
        payment_status=payment_status,
    )
    assert r.routing.value == "escalate" and r.decision.amount_usd == 0.0


def test_R3_carrier_claim_without_carrier_evidence_is_not_trusted():
    """The real Groq model turned 'no tracking number' into
    'carrier_issue: carrier_unavailable'."""
    from app.agents.resolution_policy_workflow import propose_resolution_decision
    findings = {"carrier": {"unavailable": True, "note": "no tracking number"}}
    d = propose_resolution_decision(diagnosis_root_causes=["carrier_issue: carrier_unavailable"],
                                    inventory_result={"any_shortfall": False}, order_amount_usd=30.0,
                                    retrieved_policy_doc_id="RET-POLICY-2025-A", diagnosis_findings=findings)
    assert d.requires_human_review and d.action.value == "deny"

    findings = {"carrier": {"status": "lost"}}
    d = propose_resolution_decision(diagnosis_root_causes=["carrier_issue: lost"],
                                    inventory_result={"any_shortfall": False}, order_amount_usd=30.0,
                                    retrieved_policy_doc_id="RET-POLICY-2025-A", diagnosis_findings=findings)
    assert d.action.value == "reship" and not d.requires_human_review


def test_R3_invalid_planner_actions_and_outage_end_as_incomplete(env):
    from app.agents.diagnosis_agent import run_diagnosis
    db = _seed(env, "ORD-R3", 30.0, "pi_r3")
    result = run_diagnosis(db, BrokenLLM(), order_id="ORD-R3", case_id="case-r3")
    assert result.terminated_reason == "planner_error"
    assert result.root_causes[0].startswith("diagnosis_incomplete")


# ---------------------------------------------------------------- R4
def _escalated_case(env, amount):
    db = _seed(env, "ORD-R4", amount, "pi_r4")
    db.add(env.ExceptionCase(
        id="case-r4", order_id="ORD-R4", customer_id="CUST-ORD-R4", channel="direct", exception_type="return",
        state=env.CaseState.ESCALATED,
        resolution_decision={"action": "refund", "amount_usd": 40.0, "confidence": 0.7,
                             "reasoning": "Escalated for review.",
                             "cited_policy": {"doc_id": "RET-POLICY-2025-A", "version": "1",
                                              "clause_summary": "return window"}}))
    db.commit(); db.close()


def test_R4_human_edit_over_the_hard_ceiling_is_rejected(env):
    import app.main as main_module
    importlib.reload(main_module)
    from app.tools.payment import get_payment_gateway
    _escalated_case(env, 1500.0)
    with TestClient(main_module.app) as client:
        resp = client.post("/api/v1/escalations/case-r4/decision", json={
            "action": "edit",
            "final_resolution": {"action": "refund", "amount_usd": 1500.0, "confidence": 1.0,
                                 "reasoning": "Edited by reviewer, no policy cited."}})
    assert resp.status_code == 422
    assert get_payment_gateway().refund_call_count == 0
    db = env.SessionLocal()
    assert db.get(env.ExceptionCase, "case-r4").state == env.CaseState.ESCALATED


def test_R4_cs_agent_approval_limit():
    from app.api.v1.escalations import _enforce_human_decision_limits
    from app.guardrails.schema import ResolutionDecision, CitedPolicy
    cited = CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window")
    big = ResolutionDecision(action="refund", amount_usd=400.0, confidence=1.0,
                             reasoning="Reviewer approved refund.", cited_policy=cited)
    with pytest.raises(HTTPException) as e:
        _enforce_human_decision_limits(big, "cs_agent")
    assert e.value.status_code == 403
    _enforce_human_decision_limits(big, "admin")     # within the admin limit


def test_R4_edit_without_citation_inherits_the_proposal_citation_and_second_decision_conflicts(env):
    import app.main as main_module
    importlib.reload(main_module)
    _escalated_case(env, 40.0)
    with TestClient(main_module.app) as client:
        resp = client.post("/api/v1/escalations/case-r4/decision", json={
            "action": "edit",
            "final_resolution": {"action": "refund", "amount_usd": 30.0, "confidence": 1.0,
                                 "reasoning": "Reviewer reduced the refund."}})
        assert resp.status_code == 200 and resp.json()["outcome"] == "resolved"
        again = client.post("/api/v1/escalations/case-r4/decision", json={"action": "approve"})
    assert again.status_code in (400, 409)


# ---------------------------------------------------------------- R5
def test_R5_blocked_case_gets_blocked_state(env):
    db = _seed(env, "ORD-R5", 1200.0, "pi_r5", stock=0)
    _new_case(env, db, "case-r5", "ORD-R5")
    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow
    from app.guardrails.schema import ResolutionDecision, CitedPolicy
    over = ResolutionDecision(action="refund", amount_usd=1200.0, confidence=1.0, reasoning="Over the ceiling.",
                              cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="x"))
    r = run_resolution_policy_workflow(diagnosis_root_causes=["payment_issue: x"], inventory_result={},
                                       order_amount_usd=1200.0, fraud_flag_present=False,
                                       auto_execute_confidence_threshold=0.9, auto_execute_value_ceiling_usd=50.0,
                                       override_decision=over)
    assert r.routing.value == "blocked"


# ------------------------------------------------ breaker cascade + R10
def test_permanent_errors_fail_fast_and_do_not_open_the_shared_breaker(env):
    """Against real Stripe, an 'already refunded' error retried 3x opened
    the payment circuit and blocked the next customers' refunds."""
    from app.agents.execution_agent import execute_resolution, ExecutionStatus
    from app.core.circuit_breaker import get_circuit_breaker, CircuitState
    from app.guardrails.schema import ResolutionDecision
    db = env.SessionLocal()
    decision = ResolutionDecision(action="refund", amount_usd=10.0, confidence=0.93,
                                  reasoning="Refund against an unknown payment.")
    for i in range(4):
        r = execute_resolution(db, f"case-{i}", decision, payment_intent_id="pi_does_not_exist", order_id=f"O-{i}")
        assert r.status == ExecutionStatus.FAILED and r.attempts_made == 1
    assert get_circuit_breaker("payment").state == CircuitState.CLOSED


def test_R10_fraud_llm_failure_degrades_to_rules_and_forces_review(env):
    from app.agents.workflow_agents import run_fraud_risk_agent
    db = env.SessionLocal()
    out = run_fraud_risk_agent(db, BrokenLLM(), customer_id="CUST-X")
    assert out["degraded"] is True and 0.0 <= out["risk_score"] <= 1.0
    assert any("deterministic rules" in r for r in out["reasons"])


def test_R6_auto_executed_refund_is_verified_before_resolved(env):
    from app.agents.llm_client import FakeLLMClient
    db = _seed(env, "ORD-R6", 30.0, "pi_r6")
    _new_case(env, db, "case-r6", "ORD-R6")
    from app.agents.resolution_completion import complete_resolution
    from app.guardrails.schema import ResolutionDecision, CitedPolicy
    d = ResolutionDecision(action="refund", amount_usd=30.0, confidence=0.93, reasoning="Within return window.",
                           cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="x"))
    out = complete_resolution(db, db.get(env.ExceptionCase, "case-r6"), d, d, "system:auto_execute", "auto_execute",
                              payment_intent_id="pi_r6")
    assert out["outcome"] == "resolved" and out["verification"] == "verified"


# ---------------------------------------------------------------- R5 reconciler
def _cited(action, amount):
    from app.guardrails.schema import ResolutionDecision, CitedPolicy
    return ResolutionDecision(action=action, amount_usd=amount, confidence=0.93, reasoning="Reconciler test case.",
                              cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="x"))


def test_R5_reconciler_retries_a_pending_retry_case_to_resolved(env):
    from app.agents.resolution_completion import complete_resolution
    from app.tools.payment import get_payment_gateway
    from app.workers.reconciler import run_reconciliation
    db = _seed(env, "ORD-RC1", 30.0, "pi_rc1")
    _new_case(env, db, "case-rc1", "ORD-RC1")
    get_payment_gateway().inject_transient_failures(3)          # gateway down for this attempt
    d = _cited("refund", 30.0)
    out = complete_resolution(db, db.get(env.ExceptionCase, "case-rc1"), d, d, "system:auto_execute",
                              "auto_execute", payment_intent_id="pi_rc1")
    assert out["outcome"] == "execution_pending_retry"
    assert db.get(env.ExceptionCase, "case-rc1").state == env.CaseState.PENDING_RETRY

    from app.core.circuit_breaker import reset_all_breakers
    reset_all_breakers()                                         # dependency recovered
    summary = run_reconciliation(db)
    db.expire_all()
    assert "case-rc1" in summary["case_ids"]["retried"]
    assert db.get(env.ExceptionCase, "case-rc1").state == env.CaseState.RESOLVED
    assert get_payment_gateway().refund_call_count == 4          # 3 failed attempts + 1 success, never 2 refunds


def test_R5_reconciler_resolves_a_reship_once_the_carrier_scans_it(env):
    from app.agents.resolution_completion import complete_resolution
    from app.tools.carrier import get_carrier_gateway
    from app.workers.reconciler import run_reconciliation
    db = _seed(env, "ORD-RC2", 30.0, "pi_rc2")
    _new_case(env, db, "case-rc2", "ORD-RC2")
    d = _cited("reship", 0.0)
    out = complete_resolution(db, db.get(env.ExceptionCase, "case-rc2"), d, d, "system:auto_execute", "auto_execute")
    assert out["outcome"] == "awaiting_verification"
    case = db.get(env.ExceptionCase, "case-rc2")
    assert case.state == env.CaseState.VERIFYING

    get_carrier_gateway().seed_tracking(out["execution"]["tracking_number"], "in_transit")
    run_reconciliation(db)
    db.expire_all()
    assert db.get(env.ExceptionCase, "case-rc2").state == env.CaseState.RESOLVED


def test_R5_reconciler_escalates_a_case_stuck_in_diagnosing(env):
    from app.workers.reconciler import run_reconciliation
    db = _seed(env, "ORD-RC3", 30.0, "pi_rc3")
    old = datetime.now(timezone.utc) - timedelta(hours=2)
    db.add(env.ExceptionCase(id="case-rc3", order_id="ORD-RC3", customer_id="CUST-ORD-RC3", channel="direct",
                             exception_type="return", state=env.CaseState.DIAGNOSING,
                             created_at=old, updated_at=old))
    for i in range(2):   # already re-run twice
        db.add(env.AuditLogEntry(case_id="case-rc3", actor="system:reconciler", action="reconciler_rerun",
                                 detail={"attempt": i + 1}))
    db.commit()
    summary = run_reconciliation(db)
    db.expire_all()
    case = db.get(env.ExceptionCase, "case-rc3")
    assert "case-rc3" in summary["case_ids"]["escalated"]
    assert case.state == env.CaseState.ESCALATED
    assert case.resolution_decision["requires_human_review"] is True


# ---------------------------------------------------------------- R12 lanes
def test_R12_fast_lane_is_not_blocked_by_a_running_pipeline():
    import threading
    from app.workers.job_queue import InProcessJobQueue, JobStatus
    q = InProcessJobQueue(slow_workers=1)
    gate = threading.Event()
    q.register_handler("process_oms_webhook", lambda p: (gate.wait(5), {"slow": True})[1])
    q.register_handler("process_inventory_webhook", lambda p: {"fast": True})
    q.start_worker()
    try:
        slow = q.enqueue("process_oms_webhook", {})
        fast = q.enqueue("process_inventory_webhook", {})
        assert q.wait_for_job(fast, timeout=2).status == JobStatus.SUCCEEDED
        assert q.get_job(slow).status == JobStatus.RUNNING      # still inside the "pipeline"
    finally:
        gate.set()
        q.stop_worker()


def test_R12_concurrent_duplicate_webhooks_open_one_case(env, monkeypatch):
    import threading
    from app.workers import handlers
    _seed(env, "ORD-R12", 20.0, "pi_r12").close()
    monkeypatch.setattr(handlers, "_run_pipeline_for_new_case", lambda *a, **k: None)
    start = threading.Barrier(4)

    def deliver():
        start.wait()
        handlers.handle_oms_webhook({"order_id": "ORD-R12", "new_status": "return_requested"})

    threads = [threading.Thread(target=deliver) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    db = env.SessionLocal()
    assert db.query(env.ExceptionCase).filter_by(order_id="ORD-R12").count() == 1


# ---------------------------------------------------------------- R13 data minimisation
def test_R13_planner_gets_case_type_but_no_payment_or_customer_identifiers(env):
    from app.agents.diagnosis_agent import run_diagnosis
    seen = []

    class RecordingLLM(BaseLLMClient):
        def plan_next_diagnosis_step(self, case_context, findings_so_far):
            seen.append((dict(case_context), findings_so_far))
            if "order" not in findings_so_far:
                return DiagnosisStepPlan(action="check_order", reasoning="need the order")
            if "payment" not in findings_so_far:
                return DiagnosisStepPlan(action="check_payment", reasoning="verify payment")
            return DiagnosisStepPlan(action="conclude", reasoning="done",
                                     root_causes=["no_anomaly_detected: fine"])

        def assess_fraud_risk(self, case_context, customer_risk_profile):
            return {"risk_score": 0.0, "flag": False, "reasons": []}

    db = _seed(env, "ORD-R13", 30.0, "pi_r13")
    result = run_diagnosis(db, RecordingLLM(), order_id="ORD-R13", payment_intent_id="pi_r13",
                           case_id="case-r13", exception_type="return")
    assert result.terminated_reason == "concluded"
    assert all(ctx.get("exception_type") == "return" for ctx, _ in seen)
    sent = repr([f for _, f in seen])
    for secret in ("pi_r13", "CUST-ORD-R13", "payment_fingerprint", "payment_method_breakdown"):
        assert secret not in sent, secret
    assert result.findings["order"]["payment_intent_id"] == "pi_r13"   # rules/audit still see everything


def test_R5_reconciler_scheduler_runs_passes_on_its_own(env):
    import time
    from app.workers.reconciler import ReconcilerScheduler
    s = ReconcilerScheduler(interval_seconds=0.05)
    s.start()
    try:
        deadline = time.monotonic() + 5
        while s.passes < 2 and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        s.stop()
    assert s.passes >= 2


def test_no_anomaly_contradicted_by_tool_data_goes_to_a_human():
    """Found by the live-LLM eval: the real model said 'no anomaly' on a
    stock shortfall; the rules then took the return-window refund path."""
    from datetime import date, timedelta
    from app.agents.resolution_policy_workflow import propose_resolution_decision
    common = dict(diagnosis_root_causes=["no_anomaly_detected: all checks passed"], order_amount_usd=30.0,
                  retrieved_policy_doc_id="RET-POLICY-2025-A", product_category="apparel",
                  purchase_date=(date.today() - timedelta(days=5)).isoformat(),
                  return_window_days_by_category={"apparel": 180})
    d = propose_resolution_decision(inventory_result={"any_shortfall": True}, **common)
    assert d.requires_human_review and "shortfall" in d.reasoning
    d = propose_resolution_decision(inventory_result={"any_shortfall": False}, payment_status="requires_payment_method", **common)
    assert d.requires_human_review
    d = propose_resolution_decision(inventory_result={"any_shortfall": False}, payment_status="succeeded", **common)
    assert d.action.value == "refund" and not d.requires_human_review
