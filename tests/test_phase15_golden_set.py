"""
Phase 15 pytest wrapper around tests/golden_set.py. The golden set itself
lives in a separate module (not test_*.py-prefixed) so it can also be run
standalone via `python3 tests/golden_set.py` as a pre-deployment gate, or
fed to the drift-watch job, without pytest collection overhead.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.golden_set import run_golden_set, ALL_SCENARIOS


def test_golden_set_all_scenarios_pass():
    """The pre-deployment gate itself: every golden-set scenario must
    pass before anything ships."""
    results = run_golden_set()
    failed = [r for r in results if not r.passed]
    assert not failed, f"Golden set failures: {[(r.name, r.detail) for r in failed]}"
    assert len(results) == len(ALL_SCENARIOS) == 8


def test_golden_set_includes_the_three_mandatory_scenarios():
    """Per the build checklist's Phase 15 DoD: the golden set must include
    at minimum the temporal-policy case, the duplicate-refund idempotency
    case, and the fraud-vs-high-LTV-customer case."""
    scenario_names = {s.__name__ for s in ALL_SCENARIOS}
    required = {
        "scenario_temporal_policy_correctness",
        "scenario_duplicate_refund_idempotency",
        "scenario_fraud_vs_high_ltv_customer",
    }
    missing = required - scenario_names
    assert not missing, f"Golden set is missing required scenarios: {missing}"


def test_golden_set_is_deterministic_across_three_runs():
    """Per the checklist: 'Run the golden set 3x unchanged to establish
    your variance band.' This system's decision logic is entirely
    rule-based (FakeLLMClient), so the expected variance band is exactly
    zero. A real LLM-backed deployment would run this same check and
    expect a NON-zero but bounded variance band instead; zero variance
    here is a property of the current substitution, not a general claim
    about agentic systems."""
    pass_counts = []
    for _ in range(3):
        results = run_golden_set()
        pass_counts.append(sum(1 for r in results if r.passed))

    assert pass_counts == [8, 8, 8], (
        f"Expected identical 8/8 pass counts across all 3 runs, got {pass_counts}"
    )
