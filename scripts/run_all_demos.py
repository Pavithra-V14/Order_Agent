"""
Runs every demo script in sequence, one after another, each as its own
subprocess (matching the same isolation pattern as the golden-set
runner) so one demo's failure or environment quirk can never affect
another's.

Usage:
    uvicorn app.main:app --reload   # separate terminal, first
    python3 scripts/run_all_demos.py

    # Skip specific ones:
    python3 scripts/run_all_demos.py --skip refund_demo,reship_demo

    # Only run specific ones:
    python3 scripts/run_all_demos.py --only fraud_pipeline_demo,inventory_pipeline_demo
"""
import sys
import os
import subprocess
import argparse

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Order matters a little: the full-pipeline demos first (they cover the
# most ground), the direct-tool demos last (narrower, quicker checks).
ALL_DEMOS = [
    "full_pipeline_demo",
    "return_pipeline_demo",
    "delivery_pipeline_demo",
    "fraud_pipeline_demo",
    "inventory_pipeline_demo",
    "refund_demo",
    "reship_demo",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", default="", help="comma-separated demo names to skip")
    parser.add_argument("--only", default="", help="comma-separated demo names to run (overrides --skip)")
    args = parser.parse_args()

    if args.only:
        demos = [d for d in ALL_DEMOS if d in {n.strip() for n in args.only.split(",")}]
    else:
        skip = {n.strip() for n in args.skip.split(",")} if args.skip else set()
        demos = [d for d in ALL_DEMOS if d not in skip]

    results = []
    for i, demo in enumerate(demos, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(demos)}] {demo}")
        print(f"{'=' * 70}\n")

        script_path = os.path.join(SCRIPTS_DIR, f"{demo}.py")
        result = subprocess.run([sys.executable, script_path], cwd=os.path.join(SCRIPTS_DIR, ".."))
        results.append((demo, result.returncode))

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for demo, code in results:
        status = "OK" if code == 0 else f"FAILED (exit {code})"
        print(f"  {demo:<30} {status}")

    failed = [d for d, c in results if c != 0]
    if failed:
        print(f"\n{len(failed)}/{len(results)} demo(s) failed: {', '.join(failed)}")
        sys.exit(1)
    print(f"\nAll {len(results)} demos completed successfully.")


if __name__ == "__main__":
    main()
