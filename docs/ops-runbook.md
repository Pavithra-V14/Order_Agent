# Operations Runbook
*Phase 17 deliverable - what happens after deployment, not just how to build it.*

---

## Weekly: Drift Watch

**What:** `app.workers.drift_watch` compares a fresh golden-set run (`tests/golden_set.py`) against the stored baseline (`data/golden_set_baseline.json`).

**Run it:**
```bash
python3 -m app.workers.drift_watch
```

**On `"status": "stable"`:** nothing to do. This is the expected weekly outcome.

**On `"status": "drift_detected"`:**
1. Read the `regressions` list - it names exactly which scenario(s) stopped passing.
2. Re-run that scenario's underlying test directly (`pytest tests/test_phase{N}_*.py -k <scenario>`) to get full traceback detail beyond the pass/fail summary.
3. Do NOT run `--establish` to silently reset the baseline - that erases the regression, not the cause. First confirm whether the regression is:
   - **A real bug** (something changed that shouldn't have) -> fix it, re-run drift-watch to confirm `stable`, done.
   - **An intentional, reviewed change** (e.g., a deliberate policy update that changes expected behavior) -> get a second person's sign-off on the change, THEN run `python3 -m app.workers.drift_watch --establish` to set the new baseline. Never self-approve this step - the same human-gate discipline as Phase 9's threshold recalibration applies here.
4. Either way, log the outcome (fixed vs. re-baselined-with-sign-off) somewhere durable - a ticket, a changelog entry - so "why did the baseline change on this date" has an answer six months later.

**On `"status": "no_baseline"`:** first-time setup only. Run `--establish` once, and only once, at initial deployment.

---

## Monthly: Cost Canvas Review

**What:** Re-check the Layer 14 cost canvas (`docs/architecture-decision.md`'s Cost Canvas section) against actual observed values, not the Phase 1 assumptions.

**Steps:**
1. Pull real numbers for **W** (actual exception volume/day) from `/api/v1/metrics/agent`'s `total_cases` over the past 30 days, not the Phase 1 assumed 50/day.
2. Pull **A** (amplification) from `/api/v1/metrics/tool`'s per-tool call counts divided by case count - confirm it's still ~8 steps/case, not drifting upward (a drift here usually means the Diagnosis Agent's loop is taking more steps than expected, which is itself worth investigating even before the cost implication).
3. Check free-tier rate-limit headroom (architecture doc 8.10's matrix) against actual W times A - this is the ceiling that matters at this project's scale, not dollar cost. If actual volume is approaching Groq's ~1K RPD or Mistral's 1 req/sec limits, that's the trigger to move off free tiers, not a fixed calendar date.
4. Re-file the cost canvas with real numbers, dated, so cost trend is visible across months, not just a point-in-time snapshot.

---

## Ongoing: Feeding Bad Outcomes Back Into the Golden Set

**What:** When a resolved case turns out to have been handled wrong - a human catches it on review, a customer disputes it, an audit flags it - that case should make the system PROVABLY better next time, not just get quietly corrected once.

**Steps:**
1. Identify the case's root cause category (which agent/guardrail/retrieval step actually went wrong - the trace viewer at `/traces/{case_id}` is the starting point for this).
2. Write a new scenario function in `tests/golden_set.py`, following the existing pattern (`scenario_<name>() -> ScenarioResult`), that reproduces the exact conditions that led to the bad outcome - same policy version, same customer history shape, same diagnosis inputs - and asserts the CORRECT outcome.
3. Add the new scenario to `ALL_SCENARIOS`. This is not optional or "nice to have" - a bad outcome that doesn't become a permanent regression test is a bad outcome that can recur silently.
4. Run `python3 -m app.workers.drift_watch --establish` to fold the new scenario into the baseline AFTER confirming the fix actually makes it pass.
5. If the bad outcome also revealed a genuinely new edge case not in the original architecture doc's inventory (Part 8.11), add it there too - the golden set should track the edge-case inventory, not drift ahead of or behind it silently.

This is the same principle this entire build followed from Phase 3 onward: a bug found once and fixed once is a patch; a bug found once and turned into a permanent test is an improvement to the system's actual guarantees.
