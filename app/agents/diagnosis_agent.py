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


def _fetch_order(db: Session, order_id: str, case_id: str = None) -> dict:
    return record_tool_call(db, case_id or order_id, "oms.get_order", False, oms.get_order, db, order_id)


def _fetch_payment(payment_intent_id: str, db: Session = None, case_id: str = None) -> dict:
    gateway = payment.get_payment_gateway()
    if db is None:
        return gateway.get_transaction_status(payment_intent_id)  # no db session available, skip tracing
    return record_tool_call(db, case_id or payment_intent_id, "payment.get_transaction_status", False,
                             gateway.get_transaction_status, payment_intent_id)


def _fetch_inventory(db: Session, line_items: list, case_id: str = None) -> list:
    results = []
    for item in line_items:
        stock = record_tool_call(db, case_id or item["sku"], "wms.get_stock", False, wms.get_stock, db, item["sku"])
        total_sellable = sum(s["sellable_qty"] for s in stock)
        total_on_hand = sum(s["on_hand_qty"] for s in stock)
        results.append({"sku": item["sku"], "sellable_qty": total_sellable, "on_hand_qty": total_on_hand})
    return results


def _fetch_carrier(tracking_number: str, db: Session = None, case_id: str = None) -> dict:
    gateway = carrier_tool.get_carrier_gateway()
    if db is None:
        return gateway.get_tracking_status(tracking_number)  # no db session available, skip tracing
    return record_tool_call(db, case_id or tracking_number, "carrier.get_tracking_status", False,
                             gateway.get_tracking_status, tracking_number)


def run_diagnosis(
    db: Session,
    llm: BaseLLMClient,
    order_id: str,
    payment_intent_id: str = None,
    tracking_number: str = None,
    max_steps: int = 8,
    wall_clock_timeout_seconds: float = 30.0,
    case_id: str = None,
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

    for step_num in range(1, max_steps + 1):
        elapsed = time.monotonic() - start_time
        if elapsed > wall_clock_timeout_seconds:
            return DiagnosisResult(
                findings=findings, root_causes=["diagnosis_timeout: wall-clock limit reached"],
                steps_taken=steps_taken, terminated_reason="timeout_reached",
            )

        plan = llm.plan_next_diagnosis_step(case_context, findings)

        if plan.action == "conclude":
            steps_taken.append({"step": step_num, "action": "conclude", "reasoning": plan.reasoning})
            return DiagnosisResult(
                findings=findings, root_causes=plan.root_causes or [],
                steps_taken=steps_taken, terminated_reason="concluded",
            )

        if plan.action == "check_order":
            order = _fetch_order(db, order_id, case_id=case_id)
            findings["order"] = order if order else {"error": "order not found"}
            steps_taken.append({"step": step_num, "action": "check_order", "reasoning": plan.reasoning})
            continue

        if plan.action == "check_payment" and payment_intent_id:
            findings["payment"] = _fetch_payment(payment_intent_id, db=db, case_id=case_id)
            steps_taken.append({"step": step_num, "action": "check_payment", "reasoning": plan.reasoning})
            continue

        if plan.action == "check_inventory":
            line_items = findings.get("order", {}).get("line_items", [])
            findings["inventory"] = _fetch_inventory(db, line_items, case_id=case_id)
            steps_taken.append({"step": step_num, "action": "check_inventory", "reasoning": plan.reasoning})
            continue

        if plan.action == "check_carrier" and tracking_number:
            findings["carrier"] = _fetch_carrier(tracking_number, db=db, case_id=case_id)
            steps_taken.append({"step": step_num, "action": "check_carrier", "reasoning": plan.reasoning})
            continue

        # Planner requested a check with no data source available (e.g.
        # check_payment with no payment_intent_id) - record a placeholder
        # finding so the loop can't spin forever re-requesting the same
        # unsatisfiable check; this is what makes the non-converging test
        # case actually terminate via max_steps rather than an infinite
        # identical-action loop hiding the real ceiling test.
        findings[plan.action] = {"unavailable": True}
        steps_taken.append({"step": step_num, "action": plan.action, "reasoning": plan.reasoning,
                             "note": "no data source available for this check"})

    return DiagnosisResult(
        findings=findings, root_causes=["diagnosis_incomplete: step ceiling reached before concluding"],
        steps_taken=steps_taken, terminated_reason="max_steps_reached",
    )


def run_parallel_initial_fanout(db: Session, order: dict, payment_intent_id: str,
                                 tracking_number: str, case_id: str = None) -> dict:
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
            tasks["carrier"] = executor.submit(_fetch_carrier, tracking_number, db, case_id)

        results = {"order": order}
        for key, future in tasks.items():
            try:
                results[key] = future.result(timeout=10)
            except Exception as e:
                results[key] = {"error": str(e)}
    return results
