"""
Evaluates the REAL configured LLM (Groq -> Gemini -> Cohere chain from .env)
on the live-LLM scenario set (app/eval/live_llm_eval.py).

Everything except the LLM is forced local: payment/carrier use the fake
gateways (fixed tool outputs, so only the model's reasoning varies), a
throwaway SQLite database per run, no Graphiti/Neo4j/Redis/Qdrant/Langfuse.
No money moves and no customer data leaves the machine except the
synthetic scenario data sent to the LLM.

Run it on every prompt or model change (bump PROMPT_VERSIONS in
app/agents/llm_client.py first) and before deploying:

    uv run python scripts/run_live_llm_eval.py              # 3 runs per scenario
    uv run python scripts/run_live_llm_eval.py --runs 5 --min-pass-rate 0.95

Writes data/eval/live_llm_eval_<timestamp>.json and exits 1 when the pass
rate is below --min-pass-rate, so it can gate a deploy.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

# Keep only the LLM provider keys; force every other backend local.
for var in ["MISTRAL_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "NEO4J_URI", "NEO4J_PASSWORD", "REDIS_URL",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "EASYPOST_API_KEY", "SHIPPO_API_KEY", "STRIPE_API_KEY"]:
    os.environ[var] = ""
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--min-pass-rate", type=float, default=0.9)
    parser.add_argument("--only", nargs="*", help="run only these scenario names")
    args = parser.parse_args()

    from app.core.config import get_settings
    import app.memory.graphiti_adapter as graphiti_adapter
    graphiti_adapter._is_graphiti_available = lambda: False   # episodic memory -> plain SQL

    from app.agents.llm_client import get_llm_client, LiteLLMClient, PROMPT_VERSIONS
    llm = get_llm_client()
    if not isinstance(llm, LiteLLMClient):
        sys.exit("No LLM key configured (GROQ_API_KEY / GOOGLE_API_KEY / COHERE_API_KEY) - nothing live to evaluate.")
    settings = get_settings()
    print(f"Evaluating {settings.router_model} | prompts {PROMPT_VERSIONS} | {args.runs} runs per scenario\n")

    from app.eval.live_llm_eval import run_live_eval, SCENARIOS
    scenarios = [s for s in SCENARIOS if not args.only or s.name in args.only]
    report = run_live_eval(llm, runs=args.runs, scenarios=scenarios)
    report.update({"model": settings.router_model, "prompt_versions": PROMPT_VERSIONS,
                   "evaluated_at": datetime.now(timezone.utc).isoformat()})

    width = max(len(s["scenario"]) for s in report["scenarios"])
    for s in report["scenarios"]:
        mark = "PASS" if s["passed"] == s["runs"] else ("FLAKY" if s["passed"] else "FAIL")
        print(f"  {mark:5}  {s['scenario']:<{width}}  {s['passed']}/{s['runs']}  {s['details'][0]}")
    print(f"\npass rate {report['pass_rate']:.0%} | planner failure rate {report['planner_failure_rate']:.0%} | "
          f"mean steps {report['mean_diagnosis_steps']} | {report['llm_calls']} LLM calls | "
          f"tokens {report['prompt_tokens']}/{report['completion_tokens']} | "
          f"p50 {report['p50_latency_ms']} ms, p95 {report['p95_latency_ms']} ms")
    if report["inconsistent_scenarios"]:
        print("inconsistent across runs:", report["inconsistent_scenarios"])
    for e in report["errors"]:
        print("error:", e)

    os.makedirs(os.path.join("data", "eval"), exist_ok=True)
    out = os.path.join("data", "eval", f"live_llm_eval_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("report:", out)
    sys.exit(0 if report["pass_rate"] >= args.min_pass_rate else 1)


if __name__ == "__main__":
    main()
