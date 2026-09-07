"""
Runs the golden set and prints results as JSON to stdout. Deliberately
a SEPARATE, standalone script rather than something callable in-process
from the live FastAPI app: tests/golden_set.py's scenarios each call
importlib.reload(app.core.db), swapping the database connection to a
throwaway temp SQLite file. Calling these functions directly from
within a live application would leave the app's OWN database
connection corrupted (pointed at a scenario's now-deleted temp file)
for the rest of the process's lifetime.

Running this as a subprocess (a fresh Python process) contains all of
that reloading/swapping inside a process that exits immediately after,
never touching the live app's actual database connection at all.

Usage:
    python3 scripts/run_golden_set_json.py                  # runs all 8
    python3 scripts/run_golden_set_json.py scenario_a scenario_b   # runs only the named ones
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tests.golden_set import ALL_SCENARIOS


def main():
    requested_names = sys.argv[1:]
    if requested_names:
        scenarios = [s for s in ALL_SCENARIOS if s.__name__ in requested_names]
    else:
        scenarios = ALL_SCENARIOS

    results = []
    for scenario in scenarios:
        try:
            r = scenario()
            results.append({"name": r.name, "passed": r.passed, "detail": r.detail})
        except Exception as e:
            results.append({"name": scenario.__name__, "passed": False, "detail": f"CRASHED: {e}"})

    print(json.dumps(results))


if __name__ == "__main__":
    main()
