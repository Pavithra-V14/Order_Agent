"""
Phase 15 / architecture doc Layer 10 & Part 4.3: weekly drift-watch job.
Compares a new golden-set run against a stored baseline and alerts if
results have drifted outside the established variance band.

This runs the golden set itself as the drift signal, since this
deployment has no live production traffic to sample from - a real
deployment would sample resolved cases' Tier 3 judge scores instead of
re-running the golden set, but the comparison-against-baseline mechanism
is identical either way.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from tests.golden_set import run_golden_set

BASELINE_PATH = "data/golden_set_baseline.json"


def establish_baseline() -> dict:
    """Run the golden set once and store the result as the baseline -
    done once, at initial deployment, never automatically overwritten by
    the drift-watch job itself."""
    results = run_golden_set()
    baseline = {
        "established_at": datetime.now(timezone.utc).isoformat(),
        "pass_count": sum(1 for r in results if r.passed),
        "total": len(results),
        "scenario_results": {r.name: r.passed for r in results},
    }
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    return baseline


def load_baseline():
    if not os.path.exists(BASELINE_PATH):
        return None
    with open(BASELINE_PATH) as f:
        return json.load(f)


def run_drift_check() -> dict:
    """The weekly job itself. Runs the golden set fresh and compares
    against the stored baseline. Does NOT auto-update the baseline on
    drift - a human reviews and explicitly re-establishes it if the
    drift is an intentional, accepted change (same human-gate discipline
    as Phase 9's threshold recalibration)."""
    baseline = load_baseline()
    if baseline is None:
        return {"status": "no_baseline", "message": "No baseline established yet - call establish_baseline() first."}

    current_results = run_golden_set()
    current_pass_count = sum(1 for r in current_results if r.passed)
    current_by_name = {r.name: r.passed for r in current_results}

    regressions = [
        name for name, was_passing in baseline["scenario_results"].items()
        if was_passing and not current_by_name.get(name, False)
    ]
    new_scenarios = set(current_by_name) - set(baseline["scenario_results"])

    drifted = len(regressions) > 0

    return {
        "status": "drift_detected" if drifted else "stable",
        "baseline_pass_count": baseline["pass_count"],
        "current_pass_count": current_pass_count,
        "baseline_established_at": baseline["established_at"],
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "regressions": regressions,
        "new_scenarios_since_baseline": sorted(new_scenarios),
    }


if __name__ == "__main__":
    import sys
    if "--establish" in sys.argv:
        baseline = establish_baseline()
        print(f"Baseline established: {baseline['pass_count']}/{baseline['total']} passed, "
              f"saved to {BASELINE_PATH}")
    else:
        report = run_drift_check()
        print(json.dumps(report, indent=2))
        if report["status"] == "drift_detected":
            sys.exit(1)
