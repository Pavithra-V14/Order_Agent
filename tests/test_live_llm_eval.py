"""
Proves the live-LLM evaluation harness (app/eval/live_llm_eval.py) itself:
the deterministic FakeLLMClient must pass every scenario, and a model that
makes the known real-world mistakes must be caught by the checks. The real
model is evaluated with scripts/run_live_llm_eval.py, not here (no network
in the test suite).
"""
from app.agents.llm_client import BaseLLMClient, DiagnosisStepPlan, FakeLLMClient


def test_harness_passes_the_rule_based_client():
    from app.eval.live_llm_eval import run_live_eval
    report = run_live_eval(FakeLLMClient(), runs=1)
    failing = [s for s in report["scenarios"] if s["passed"] != s["runs"]]
    assert report["pass_rate"] == 1.0, failing
    assert report["planner_failure_rate"] == 0.0
    assert report["errors"] == []


class HallucinatingLLM(FakeLLMClient):
    """Reproduces what the real Groq model did before the prompt/evidence
    fixes: turn missing data into a fault."""
    def plan_next_diagnosis_step(self, case_context, findings_so_far):
        plan = super().plan_next_diagnosis_step(case_context, findings_so_far)
        if plan.action == "conclude":
            return DiagnosisStepPlan(action="conclude", reasoning=plan.reasoning,
                                     root_causes=["carrier_issue: carrier_unavailable",
                                                  "payment_issue: missing payment information"])
        return plan


def test_harness_catches_a_model_that_invents_faults():
    from app.eval.live_llm_eval import run_live_eval, SCENARIOS
    scenarios = [s for s in SCENARIOS if s.name in ("clean_return_invents_no_fault",
                                                    "missing_payment_is_not_a_payment_fault")]
    report = run_live_eval(HallucinatingLLM(), runs=2, scenarios=scenarios)
    assert report["pass_rate"] == 0.0
    assert report["inconsistent_scenarios"] == []


class DownLLM(BaseLLMClient):
    def plan_next_diagnosis_step(self, case_context, findings_so_far):
        raise RuntimeError("provider unavailable")

    def assess_fraud_risk(self, case_context, customer_risk_profile):
        raise RuntimeError("provider unavailable")


def test_harness_counts_planner_failures_and_degraded_fraud():
    from app.eval.live_llm_eval import run_live_eval
    report = run_live_eval(DownLLM(), runs=1)
    assert report["planner_failure_rate"] == 1.0
    assert report["pass_rate"] < 0.5
