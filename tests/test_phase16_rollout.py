"""
Phase 16: rollback verification (must take effect in well under 5
minutes) and a staged rollout simulation (Internal -> Beta -> wider
synthetic load), documenting what would be monitored at each stage.
"""
import os
import time
import tempfile

import pytest

@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_phase16_{os.getpid()}_{id(object())}.db")
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
    os.environ.pop("AUTO_EXECUTION_ENABLED", None)
    from app.core.config import get_settings as gs
    gs.cache_clear()

@pytest.fixture(autouse=True)
def reset_all():
    from app.tools.payment import reset_fake_gateway
    from app.core.circuit_breaker import reset_all_breakers
    reset_fake_gateway()
    reset_all_breakers()
    yield

def test_rollback_disables_auto_execution_for_all_new_cases_in_under_5_minutes(isolated_db):
    """THE Phase 16 rollback test."""
    from app.core.config import get_settings
    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow

    get_settings.cache_clear()

    result_before = run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: transaction status is 'declined'"],
        inventory_result={"any_shortfall": False}, order_amount_usd=20.0,
        fraud_flag_present=False, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A", retrieved_policy_version="1",
        # A genuinely succeeded payment — the only scenario where a
        # refund actually makes sense, per a real bug found and fixed
        # separately (refunding a never-charged payment is nonsensical,
        # and real Stripe correctly rejects it). Needed here so this
        # test's ONLY variable is the rollback switch itself, not also
        # accidentally exercising the payment-status guardrail.
        payment_status="succeeded",
    )
    assert result_before.routing.value == "auto_execute"

    rollback_start = time.monotonic()
    os.environ["AUTO_EXECUTION_ENABLED"] = "false"
    get_settings.cache_clear()
    rollback_elapsed = time.monotonic() - rollback_start

    assert rollback_elapsed < 300.0, f"Rollback took {rollback_elapsed}s - must be well under 5 minutes (300s)"
    assert rollback_elapsed < 1.0, (
        f"In practice this is a config reload, not a redeploy - expected well under 1s, got {rollback_elapsed}s"
    )

    result_after = run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: transaction status is 'declined'"],
        inventory_result={"any_shortfall": False}, order_amount_usd=20.0,
        fraud_flag_present=False, auto_execute_confidence_threshold=0.90,
        auto_execute_value_ceiling_usd=50.0,
        retrieved_policy_doc_id="RET-POLICY-2025-A", retrieved_policy_version="1",
        payment_status="succeeded",
    )
    assert result_after.routing.value == "escalate"
    assert any("globally disabled" in r for r in result_after.routing_reasons)

def test_rollback_does_not_affect_tier1_blocking(isolated_db):
    """The rollback switch controls AUTO_EXECUTE-vs-ESCALATE only - it
    must NOT weaken Tier 1's hard ceilings."""
    from app.core.config import get_settings
    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy

    os.environ["AUTO_EXECUTION_ENABLED"] = "false"
    get_settings.cache_clear()

    adversarial = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=5000.0, confidence=1.0,
        reasoning="Confident reasoning, but this must still be BLOCKED not just escalated.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )
    result = run_resolution_policy_workflow(
        diagnosis_root_causes=["payment_issue: x"], inventory_result={"any_shortfall": False},
        order_amount_usd=5000.0, fraud_flag_present=False,
        auto_execute_confidence_threshold=0.90, auto_execute_value_ceiling_usd=50.0,
        override_decision=adversarial,
    )
    assert result.routing.value == "blocked", "Tier 1 blocking must remain active regardless of the rollback switch"

def test_staged_rollout_internal_stage(isolated_db):
    """Stage 1 (Internal): a handful of synthetic requests, confirming
    basic health before widening exposure."""
    from fastapi.testclient import TestClient
    import app.main as main_module
    import importlib
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        for i in range(5):
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200

    from tests.golden_set import run_golden_set
    results = run_golden_set()
    assert all(r.passed for r in results), "Internal stage requires 8/8 golden set before proceeding to Beta"

def test_staged_rollout_beta_stage(isolated_db):
    """Stage 2 (Beta): a small opt-in cohort's traffic, simulated as a
    batch of case creations + reads."""
    from fastapi.testclient import TestClient
    import app.main as main_module
    import importlib
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        created_ids = []
        for i in range(20):
            resp = client.post("/api/v1/cases", json={
                "order_id": f"ORD-BETA-{i}", "customer_id": f"CUST-BETA-{i}",
                "channel": "direct", "exception_type": "return",
            })
            assert resp.status_code == 201
            created_ids.append(resp.json()["id"])

        for case_id in created_ids:
            resp = client.get(f"/api/v1/cases/{case_id}")
            assert resp.status_code == 200
            assert resp.json()["state"] == "detected"

def test_staged_rollout_wider_synthetic_load_stage(isolated_db):
    """Stage 3 (wider synthetic load): reuses Phase 14's load-test
    pattern at higher volume than Beta."""
    from concurrent.futures import ThreadPoolExecutor
    from fastapi.testclient import TestClient
    import app.main as main_module
    import importlib
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        errors = []

        def make_request(i):
            resp = client.post("/api/v1/cases", json={
                "order_id": f"ORD-STAGE3-{i}", "customer_id": f"CUST-STAGE3-{i}",
                "channel": "direct", "exception_type": "return",
            })
            if resp.status_code != 201:
                errors.append((i, resp.status_code))

        with ThreadPoolExecutor(max_workers=15) as executor:
            list(executor.map(make_request, range(100)))

        assert len(errors) == 0, f"Stage 3 must show zero errors before proceeding to 50%/100%: {errors[:5]}"

    from app.core.circuit_breaker import get_circuit_breaker, CircuitState
    breaker = get_circuit_breaker("payment")
    assert breaker.state == CircuitState.CLOSED
