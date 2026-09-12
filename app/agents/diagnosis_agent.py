"""
Diagnosis Agent - the ONE genuinely agentic (open-loop) component per
Part 1's autonomy calibration; everything downstream (Resolution-Policy,
Execution) is a deterministic workflow. Implements the iterative
Planner-Worker pattern: plan ONE step -> execute -> re-evaluate -> repeat,
never a blind upfront plan.

Parallel fan-out: once the order is known, independent checks (payment
status, inventory per line item, carrier tracking) run concurrently
rather than sequentially, per architecture doc 8's parallel-fan-out design.

Hard ceilings: max_steps and wall_clock_timeout_seconds - required by
Part 1.5.3's explicit warning against a runaway diagnosis loop. Tested in
tests/test_phase6_agents.py with a deliberately non-converging scenario.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.agents.llm_client import BaseLLMClient, DiagnosisStepPlan
from app.core.tracing import record_tool_call
from app.tools import oms, wms, payment, carrier as carrier_tool


@dataclass
class DiagnosisResult:
    findings: dict = field(default_factory=dict)
    root_causes: list = field(default_factory=list)
    steps_taken: list = field(default_factory=list)
    terminated_reason: str = ""  # "concluded" | "max_steps_reached" | "timeout_reached"
    # Short-term working memory's final state (app/memory/summary_buffer.py) —
    # a bounded, human-readable recap of the diagnosis, distinct from
    # `findings` (raw structured data) — found fully implemented but
    # never actually populated or exposed anywhere during a direct audit.
    case_summary: dict = field(default_factory=dict)


def _fetch_order(db: Session, order_id: str, case_id: str = None) -> dict:
    return record_tool_call(db, case_id or order_id, "oms.get_order", False, oms.get_order, db, order_id)


def _fetch_payment(payment_intent_id: str, db: Session = None, case_id: str = None) -> dict:
    gateway = payment.get_payment_gateway()
    if db is None:
        return gateway.get_transaction_status(payment_intent_id)  # no db session available, skip tracing
    return record_tool_call(db, case_id or payment_intent_id, "payment.get_transaction_status", False,
                             gateway.get_transaction_status, payment_intent_id)


def _fetch_inventory(db: Session, line_items: list, case_id: str = None) -> list:
    from app.cache.tool_cache import get_stock_cached
    results = []
    for item in line_items:
        # Cache-aside (app/cache/tool_cache.py) - found fully
        # implemented but never actually called from anywhere during a
        # direct audit: invalidate_stock_cache() was correctly wired
        # into the inventory webhook handler, but nothing ever POPULATED
        # or READ from the cache in the first place, meaning every
        # single inventory lookup during diagnosis hit the DB directly
        # regardless — the cache existed but did nothing.
        stock = record_tool_call(db, case_id or item["sku"], "wms.get_stock", False, get_stock_cached, db, item["sku"])
        total_sellable = sum(s["sellable_qty"] for s in stock)
        total_on_hand = sum(s["on_hand_qty"] for s in stock)
        results.append({"sku": item["sku"], "sellable_qty": total_sellable, "on_hand_qty": total_on_hand})
    return results


def _fetch_carrier(tracking_number: str, db: Session = None, case_id: str = None, carrier: str = None) -> dict:
    from app.cache.tool_cache import get_tracking_cached
    # Same real gap as inventory above - get_tracking_cached() existed
    # but was never actually called; every carrier check hit the real
    # gateway directly, cache-aside in name only.
    if db is None:
        return get_tracking_cached(tracking_number, carrier=carrier)  # no db session available, skip tracing
    return record_tool_call(db, case_id or tracking_number, "carrier.get_tracking_status", False,
                             get_tracking_cached, tracking_number, carrier=carrier)


def run_diagnosis(
    db: Session,
    llm: BaseLLMClient,
    order_id: str,
    payment_intent_id: str = None,
    tracking_number: str = None,
    max_steps: int = 8,
    wall_clock_timeout_seconds: float = 30.0,
    case_id: str = None,
    carrier: str = None,
) -> DiagnosisResult:
    """Runs the iterative diagnosis loop. case_context passed to the LLM
    planner is intentionally thin (just order_id) - the planner discovers
    what it needs by requesting checks, not by being handed everything
    upfront; that's what makes this a genuine plan-execute-replan loop
    rather than a fixed pipeline with an LLM label on it."""
    start_time = time.monotonic()
    findings = {}
    steps_taken = []
    case_context = {"order_id": order_id}

    # Short-term, per-case working memory (app/memory/summary_buffer.py,
    # architecture doc 8.4's Summary Buffer) — found fully implemented
    # but never actually called from anywhere during a direct audit.
    # Reset at the start of every diagnosis run so a reopened case
    # doesn't inherit a stale summary from a previous, unrelated run.
    from app.memory.summary_buffer import get_or_create_buffer
    buffer = get_or_create_buffer(case_id or order_id)
    buffer.verbatim_items.clear()
    buffer.running_summary = ""

    def _record_step(entry: dict) -> None:
        steps_taken.append(entry)
        # Summary Buffer's fold logic expects agent/summary keys (see
        # SummaryBuffer._summarize) - mapped from this loop's own
        # action/reasoning shape here rather than changing steps_taken's
        # existing, already-tested shape everywhere else in this file.
        buffer.add({"agent": entry.get("action", "unknown"), "summary": entry.get("reasoning", "")})

    for step_num in range(1, max_steps + 1):
        elapsed = time.monotonic() - start_time
        if elapsed > wall_clock_timeout_seconds:
            return DiagnosisResult(
                findings=findings, root_causes=["diagnosis_timeout: wall-clock limit reached"],
                steps_taken=steps_taken, terminated_reason="timeout_reached",
                case_summary=buffer.get_context(),
            )

        plan = llm.plan_next_diagnosis_step(case_context, findings)

        if plan.action == "conclude":
            _record_step({"step": step_num, "action": "conclude", "reasoning": plan.reasoning})
            return DiagnosisResult(
                findings=findings, root_causes=plan.root_causes or [],
                steps_taken=steps_taken, terminated_reason="concluded",
                case_summary=buffer.get_context(),
            )

        # Defensive backstop against a real LLM re-requesting a check
        # whose data already exists — found from an actual production
        # run: a real Groq model called check_carrier 7 TIMES in a row
        # despite already having a clear "returned" status from the
        # first call, never reaching "conclude" and burning the entire
        # step ceiling. The system prompt now explicitly instructs
        # against this, but prompting alone can't be fully relied on —
        # this skips the WASTED real API call (Shippo/Stripe/etc.) a
        # redundant request would otherwise make, regardless of why the
        # model asked for it again, and nudges toward concluding instead
        # of silently repeating the same real network call for no new
        # information.
        _redundant_check_map = {"check_order": "order", "check_payment": "payment",
                                 "check_inventory": "inventory", "check_carrier": "carrier"}
        redundant_key = _redundant_check_map.get(plan.action)
        if redundant_key and redundant_key in findings:
            _record_step({
                "step": step_num, "action": f"skipped_redundant_{plan.action}",
                "reasoning": f"'{redundant_key}' was already fetched in an earlier step — "
                             f"not repeating a real API call for data that won't change.",
            })
            continue

        if plan.action == "check_order":
            try:
                order = _fetch_order(db, order_id, case_id=case_id)
                findings["order"] = order if order else {"error": "order not found"}
            except Exception as e:
                # A single check failing must NOT crash the whole
                # diagnosis loop — found from a real production crash:
                # a genuine Stripe API error (payment_intent doesn't
                # exist) propagated all the way up through diagnosis,
                # through the LangGraph pipeline, and crashed the entire
                # script with an unhandled traceback, instead of being
                # treated as "this one check failed" so diagnosis could
                # still conclude from whatever other checks succeeded.
                findings["order"] = {"error": str(e)}
            _record_step({"step": step_num, "action": "check_order", "reasoning": plan.reasoning})
            continue

        if plan.action == "check_payment" and payment_intent_id:
            try:
                findings["payment"] = _fetch_payment(payment_intent_id, db=db, case_id=case_id)
            except Exception as e:
                findings["payment"] = {"error": str(e)}
            _record_step({"step": step_num, "action": "check_payment", "reasoning": plan.reasoning})
            continue

        if plan.action == "check_inventory":
            line_items = findings.get("order", {}).get("line_items", [])
            try:
                findings["inventory"] = _fetch_inventory(db, line_items, case_id=case_id)
            except Exception as e:
                findings["inventory"] = {"error": str(e)}
            _record_step({"step": step_num, "action": "check_inventory", "reasoning": plan.reasoning})
            continue

        if plan.action == "check_carrier" and tracking_number:
            try:
                findings["carrier"] = _fetch_carrier(tracking_number, db=db, case_id=case_id, carrier=carrier)
            except Exception as e:
                findings["carrier"] = {"error": str(e)}
            _record_step({"step": step_num, "action": "check_carrier", "reasoning": plan.reasoning})
            continue

        # Planner requested a check with no data source available (e.g.
        # check_payment with no payment_intent_id) - record a placeholder
        # finding so the loop can't spin forever re-requesting the same
        # unsatisfiable check; this is what makes the non-converging test
        # case actually terminate via max_steps rather than an infinite
        # identical-action loop hiding the real ceiling test.
        findings[plan.action] = {"unavailable": True}
        _record_step({"step": step_num, "action": plan.action, "reasoning": plan.reasoning,
                      "note": "no data source available for this check"})

    return DiagnosisResult(
        findings=findings, root_causes=["diagnosis_incomplete: step ceiling reached before concluding"],
        steps_taken=steps_taken, terminated_reason="max_steps_reached",
        case_summary=buffer.get_context(),
    )


def run_parallel_initial_fanout(db: Session, order: dict, payment_intent_id: str,
                                 tracking_number: str, case_id: str = None, carrier: str = None) -> dict:
    """Once the order is known, independent reads (payment, inventory,
    carrier) run concurrently rather than sequentially. Returns a findings
    dict usable as a fast-path seed for the loop above."""
    tasks = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        if payment_intent_id:
            tasks["payment"] = executor.submit(_fetch_payment, payment_intent_id, db, case_id)
        if order.get("line_items"):
            tasks["inventory"] = executor.submit(_fetch_inventory, db, order["line_items"], case_id)
        if tracking_number:
            tasks["carrier"] = executor.submit(_fetch_carrier, tracking_number, db, case_id, carrier)

        results = {"order": order}
        for key, future in tasks.items():
            try:
                results[key] = future.result(timeout=10)
            except Exception as e:
                results[key] = {"error": str(e)}
    return results
