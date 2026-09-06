"""
Tests for app.workers.drift_watch - the weekly drift-comparison job.
"""
import os

import pytest

@pytest.fixture(autouse=True)
def isolated_baseline():
    test_baseline_path = "data/golden_set_baseline_test.json"
    if os.path.exists(test_baseline_path):
        os.remove(test_baseline_path)
    import app.workers.drift_watch as dw
    original_path = dw.BASELINE_PATH
    dw.BASELINE_PATH = test_baseline_path
    yield dw
    dw.BASELINE_PATH = original_path
    if os.path.exists(test_baseline_path):
        os.remove(test_baseline_path)

def test_drift_check_with_no_baseline_reports_that_clearly(isolated_baseline):
    result = isolated_baseline.run_drift_check()
    assert result["status"] == "no_baseline"

def test_establish_baseline_then_stable_check(isolated_baseline):
    baseline = isolated_baseline.establish_baseline()
    assert baseline["pass_count"] == 8
    assert os.path.exists(isolated_baseline.BASELINE_PATH)

    result = isolated_baseline.run_drift_check()
    assert result["status"] == "stable"
    assert result["regressions"] == []

def test_drift_check_detects_a_real_regression(isolated_baseline, monkeypatch):
    """Proves the mechanism actually catches drift, not just that it can
    report 'stable' - the only way to trust the 'stable' result above is
    to also see this test correctly report 'drift_detected' when a
    scenario that WAS passing stops passing."""
    isolated_baseline.establish_baseline()

    from tests.golden_set import ScenarioResult
    import app.workers.drift_watch as dw

    def fake_run_golden_set_with_regression():
        return [
            ScenarioResult("temporal_policy_correctness", False, "simulated regression"),
            ScenarioResult("duplicate_refund_idempotency", True, "ok"),
            ScenarioResult("fraud_vs_high_ltv_customer", True, "ok"),
            ScenarioResult("multi_cause_diagnosis", True, "ok"),
            ScenarioResult("runaway_loop_terminates", True, "ok"),
            ScenarioResult("tier1_hard_block", True, "ok"),
            ScenarioResult("circuit_breaker_fails_fast", True, "ok"),
            ScenarioResult("webhook_cache_invalidation", True, "ok"),
        ]

    monkeypatch.setattr(dw, "run_golden_set", fake_run_golden_set_with_regression)

    result = dw.run_drift_check()
    assert result["status"] == "drift_detected"
    assert "temporal_policy_correctness" in result["regressions"]
    assert len(result["regressions"]) == 1
