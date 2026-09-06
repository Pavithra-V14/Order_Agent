"""
Phase 15: golden evaluation set, built from the edge-case inventory
(architecture doc Part 8.11). 8 scenarios, including the 3 the checklist
requires as a minimum: temporal-policy correctness, duplicate-refund
idempotency, and fraud-vs-high-LTV-customer weighting.

Architecture doc specifies DeepEval, whose default metrics (faithfulness,
G-Eval, etc.) require a real LLM judge - network access + an API key
this sandbox doesn't have. This is also the wrong tool for THIS system's
current state: every agent decision here is rule-based (FakeLLMClient),
so there's no semantic ambiguity for an LLM judge to resolve - the right
eval is a structural assertion ("did the system retrieve doc X and route
to outcome Y"), which this harness provides directly. DeepEval becomes
the right tool once a real LLM is wired in and its outputs need semantic
judging, not just structural checking.

Each scenario returns (name, passed, detail) rather than raising, so the
harness can run to completion and report a full scorecard even if one
scenario fails.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    detail: str


def _fresh_db(suffix):
    tmp_path = os.path.join(tempfile.gettempdir(), f"golden_set_{suffix}.db")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}"
    from app.core.config import get_settings
    get_settings.cache_clear()
    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)
    db_module.init_db()
    return db_module, tmp_path


def scenario_temporal_policy_correctness() -> ScenarioResult:
    """Edge case: policy changed mid-order-lifecycle. An order under the
    OLD policy version must retrieve the OLD window, never the
    superseding version."""
    qdrant_path = "data/qdrant_local_golden_temporal"
    reindex_state = "data/reindex_state_golden_temporal.json"
    if os.path.exists(qdrant_path):
        shutil.rmtree(qdrant_path)
    if os.path.exists(reindex_state):
        os.remove(reindex_state)
    os.environ["QDRANT_LOCAL_PATH"] = qdrant_path

    db_module, tmp_path = _fresh_db("temporal")
    import app.rag.vectorstore as vs
    if vs._client_singleton is not None:
        vs._client_singleton.close()
    vs._client_singleton = None
    vs._client_singleton_key = None

    import app.rag.ingestion as ing
    ing._REINDEX_STATE_PATH = reindex_state
    ing.ingest_policy_directory("data/policies")

    from app.rag.retrieval import hybrid_search
    results = hybrid_search(query="return window apparel", as_of_date="2025-06-15",
                             doc_type="return_policy", top_k=5)
    doc_ids = {r.metadata.get("doc_id") for r in results}
    all_text = " ".join(r.text for r in results)

    if vs._client_singleton is not None:
        vs._client_singleton.close()
    vs._client_singleton = None
    vs._client_singleton_key = None
    if os.path.exists(qdrant_path):
        shutil.rmtree(qdrant_path)
    if os.path.exists(reindex_state):
        os.remove(reindex_state)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    passed = "RET-POLICY-2026-A" not in doc_ids and "180" in all_text
    return ScenarioResult("temporal_policy_correctness", passed,
                           f"retrieved doc_ids={doc_ids}, expected 180-day window present, 2026-A absent")


def scenario_duplicate_refund_idempotency() -> ScenarioResult:
    """Edge case: duplicate refund on retry. Two calls with the same
    idempotency_key must produce exactly one real execution."""
    db_module, tmp_path = _fresh_db("idempotency")
    from app.tools.payment import get_payment_gateway, reset_fake_gateway
    reset_fake_gateway()
    db = db_module.SessionLocal()
    gateway = get_payment_gateway()
    gateway.seed_transaction("pi_golden_1", amount_usd=100.0)

    r1 = gateway.issue_refund(db, "pi_golden_1", 42.0, idempotency_key="golden-key-1")
    r2 = gateway.issue_refund(db, "pi_golden_1", 42.0, idempotency_key="golden-key-1")
    db.close()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    passed = gateway.refund_call_count == 1 and r1["id"] == r2["id"]
    return ScenarioResult("duplicate_refund_idempotency", passed,
                           f"call_count={gateway.refund_call_count}, same_id={r1['id'] == r2['id']}")


def scenario_fraud_vs_high_ltv_customer() -> ScenarioResult:
    """Edge case: serial returner who is also high-value/low-fraud-risk.
    Risk scoring must weight return-reason pattern and prior fraud
    flags, NOT raw return count alone."""
    db_module, tmp_path = _fresh_db("fraud_ltv")
    db = db_module.SessionLocal()
    from app.memory.episodic import log_episode
    from app.agents.workflow_agents import run_fraud_risk_agent
    from app.agents.llm_client import FakeLLMClient

    customer_id = "CUST-GOLDEN-HIGH-LTV"
    for i in range(15):
        log_episode(db, customer_id, "case_resolved", {"exception_type": "return", "outcome": "approved"},
                    occurred_at=datetime(2025, 1, i % 28 + 1, tzinfo=timezone.utc))

    result = run_fraud_risk_agent(db, FakeLLMClient(), customer_id=customer_id,
                                   address_changed_same_day=False)
    db.close()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    passed = result["flag"] is False and result["risk_score"] < 0.6
    return ScenarioResult("fraud_vs_high_ltv_customer", passed,
                           f"risk_score={result['risk_score']}, flag={result['flag']}, reasons={result['reasons']}")


def scenario_multi_cause_diagnosis() -> ScenarioResult:
    """Edge case: compound root causes (payment decline + concurrent OOS)
    must both surface."""
    db_module, tmp_path = _fresh_db("multicause")
    db = db_module.SessionLocal()
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.tools.payment import get_payment_gateway, reset_fake_gateway
    from app.agents.llm_client import FakeLLMClient
    from app.agents.diagnosis_agent import run_diagnosis

    reset_fake_gateway()
    create_order(db, order_id="ORD-GOLDEN-MC", customer_id="CUST-GOLDEN-MC", channel="direct",
                 status="payment_failed", total_amount_usd=100.0,
                 purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
                 line_items=[{"sku": "SKU-GOLDEN-MC", "category": "electronics", "qty": 2, "price": 50.0}])
    get_payment_gateway().seed_transaction("pi_golden_mc", amount_usd=100.0, status="declined")
    seed_stock(db, sku="SKU-GOLDEN-MC", warehouse="WH-A", on_hand_qty=1, sellable_qty=0)
    seed_stock(db, sku="SKU-GOLDEN-MC", warehouse="WH-B", on_hand_qty=1, sellable_qty=1)

    result = run_diagnosis(db, FakeLLMClient(), order_id="ORD-GOLDEN-MC", payment_intent_id="pi_golden_mc")
    db.close()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    causes_text = " | ".join(result.root_causes)
    passed = "payment_issue" in causes_text and ("inventory_issue" in causes_text or "SKU-GOLDEN-MC" in causes_text)
    return ScenarioResult("multi_cause_diagnosis", passed, f"root_causes={result.root_causes}")


def scenario_runaway_loop_terminates() -> ScenarioResult:
    """Edge case: a non-converging diagnosis loop must stop at the step
    ceiling, never run indefinitely."""
    db_module, tmp_path = _fresh_db("runaway")
    db = db_module.SessionLocal()
    from app.tools.oms import create_order
    from app.agents.llm_client import BaseLLMClient, DiagnosisStepPlan
    from app.agents.diagnosis_agent import run_diagnosis

    create_order(db, order_id="ORD-GOLDEN-RUNAWAY", customer_id="CUST-GOLDEN-RUNAWAY", channel="direct",
                 status="paid", total_amount_usd=10.0, purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
                 line_items=[])

    class NeverConcludesLLM(BaseLLMClient):
        def plan_next_diagnosis_step(self, case_context, findings_so_far):
            return DiagnosisStepPlan(action="check_nonexistent", reasoning="golden set non-converging test")
        def assess_fraud_risk(self, case_context, customer_risk_profile):
            return {"risk_score": 0.0, "flag": False, "reasons": []}

    result = run_diagnosis(db, NeverConcludesLLM(), order_id="ORD-GOLDEN-RUNAWAY", max_steps=5)
    db.close()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    passed = result.terminated_reason == "max_steps_reached" and len(result.steps_taken) == 5
    return ScenarioResult("runaway_loop_terminates", passed,
                           f"terminated_reason={result.terminated_reason}, steps={len(result.steps_taken)}")


def scenario_tier1_hard_block() -> ScenarioResult:
    """Edge case: Tier 1 guardrail must block regardless of model
    confidence."""
    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy

    adversarial = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=5000.0, confidence=1.0,
        reasoning="Maximally confident reasoning that should still be blocked by hard numeric ceilings.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )
    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: x"], inventory_result={"any_shortfall": False},
        order_amount_usd=5000.0, fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90, auto_execute_value_ceiling_usd=50.0,
        override_decision=adversarial,
    )
    passed = result.routing.value == "blocked" and result.tier1_passed is False
    return ScenarioResult("tier1_hard_block", passed, f"routing={result.routing.value}")


def scenario_circuit_breaker_fails_fast() -> ScenarioResult:
    """Edge case: a permanently-failing dependency trips the circuit
    breaker and stops wasting calls."""
    db_module, tmp_path = _fresh_db("circuit")
    db = db_module.SessionLocal()
    from app.tools.payment import get_payment_gateway, reset_fake_gateway
    from app.core.circuit_breaker import reset_all_breakers, get_circuit_breaker, CircuitState
    from app.agents.execution_agent import execute_resolution
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy

    reset_fake_gateway()
    reset_all_breakers()
    gateway = get_payment_gateway()
    gateway.seed_transaction("pi_golden_circuit", amount_usd=100.0)
    gateway.inject_transient_failures(999)

    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=42.0, confidence=0.95,
        reasoning="Golden set circuit breaker scenario.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )
    execute_resolution(db, case_id="case-golden-circuit", decision=decision,
                        payment_intent_id="pi_golden_circuit", max_retries=5)
    breaker = get_circuit_breaker("payment", failure_threshold=3, reset_timeout_seconds=30.0)
    db.close()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    passed = breaker.state == CircuitState.OPEN and breaker.call_attempts == 3
    return ScenarioResult("circuit_breaker_fails_fast", passed,
                           f"state={breaker.state.value}, call_attempts={breaker.call_attempts}")


def scenario_webhook_cache_invalidation() -> ScenarioResult:
    """Edge case: phantom stock - a webhook must invalidate the cache so
    the next read reflects new state, not a stale one."""
    from app.tools.wms import seed_stock, handle_inventory_update_webhook
    from app.cache.tool_cache import get_stock_cached
    from app.cache.ttl_cache import reset_all_caches

    db_module, tmp_path = _fresh_db("cache")
    db = db_module.SessionLocal()
    reset_all_caches()
    seed_stock(db, sku="SKU-GOLDEN-CACHE", warehouse="WH-A", on_hand_qty=10, sellable_qty=10)
    get_stock_cached(db, "SKU-GOLDEN-CACHE", "WH-A")
    handle_inventory_update_webhook(db, sku="SKU-GOLDEN-CACHE", warehouse="WH-A",
                                     new_on_hand_qty=2, new_sellable_qty=2)
    second_read = get_stock_cached(db, "SKU-GOLDEN-CACHE", "WH-A")
    db.close()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    passed = second_read[0]["sellable_qty"] == 2
    return ScenarioResult("webhook_cache_invalidation", passed,
                           f"sellable_qty_after_webhook={second_read[0]['sellable_qty']}")


ALL_SCENARIOS = [
    scenario_temporal_policy_correctness,
    scenario_duplicate_refund_idempotency,
    scenario_fraud_vs_high_ltv_customer,
    scenario_multi_cause_diagnosis,
    scenario_runaway_loop_terminates,
    scenario_tier1_hard_block,
    scenario_circuit_breaker_fails_fast,
    scenario_webhook_cache_invalidation,
]


def run_golden_set():
    return [scenario() for scenario in ALL_SCENARIOS]


if __name__ == "__main__":
    results = run_golden_set()
    passed_count = sum(1 for r in results if r.passed)
    print(f"\nGolden Set Results: {passed_count}/{len(results)} passed\n")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.detail}")
    if passed_count < len(results):
        raise SystemExit(1)
