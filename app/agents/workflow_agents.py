"""
Fraud/Risk, Inventory, and Customer Context agents - all WORKFLOWS per
Part 1's autonomy calibration (bounded, not open-ended reasoning loops).
Each takes structured input and produces a structured output through a
fixed sequence of steps - the "draw the full flowchart in advance" test
from the architecture guide passes for all three, unlike the Diagnosis
Agent.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.agents.llm_client import BaseLLMClient
from app.memory.episodic import summarize_customer_risk_profile
from app.core.tracing import record_tool_call
from app.tools import wms


def run_fraud_risk_agent(db: Session, llm: BaseLLMClient, customer_id: str,
                          address_changed_same_day: bool = False, order_id: str = None) -> dict:
    """Fixed sequence: pull risk profile from memory -> score via rule-based
    assessment -> return. No branching on intermediate results beyond the
    scoring rules themselves - this is a workflow, not a loop.

    order_id (Stage 1 memory upgrade, optional/backward-compatible):
    when given, looks up this order's payment_fingerprint and checks
    whether any OTHER customer sharing it has a fraud flag of their own
    (app/memory/graphiti_adapter.py's find_related_fraud_signals - Neo4j
    Aura only; returns [] on every other backend/config, same as if the
    signal were simply absent). This never raises - a lookup failure
    just means this one extra signal is unavailable for this call.
    """
    risk_profile = summarize_customer_risk_profile(db, customer_id)

    related_fraud_signals: list[dict] = []
    if order_id:
        try:
            from app.tools.oms import get_order
            order = get_order(db, order_id)
            fingerprint = order.get("payment_fingerprint") if order else None
            if fingerprint:
                from app.memory.graphiti_adapter import find_related_fraud_signals
                related_fraud_signals = find_related_fraud_signals(fingerprint, exclude_customer_id=customer_id)
        except Exception as e:
            import logging
            logging.getLogger("workflow_agents").warning(
                "related-fraud-signal lookup failed (non-fatal, continuing without it): %s", e)

    case_context = {
        "address_changed_same_day": address_changed_same_day,
        "related_fraud_signals": related_fraud_signals,
    }
    from app.core.config import get_settings
    threshold = get_settings().fraud_flag_threshold
    degraded = False
    try:
        score, model_flag, reasons = _validate_fraud_assessment(llm.assess_fraud_risk(case_context, risk_profile))
    except Exception as e:
        # LLM outage or malformed output: score with the deterministic
        # rules instead of failing the case, and mark it degraded so the
        # routing layer sends it to a human.
        from app.agents.llm_client import FakeLLMClient
        score, model_flag, reasons = _validate_fraud_assessment(
            FakeLLMClient().assess_fraud_risk(case_context, risk_profile))
        reasons = reasons + [f"LLM fraud scoring unavailable ({type(e).__name__}); deterministic rules used"]
        degraded = True

    # The threshold is owned by code, not the model: flag when the score
    # crosses it OR the model itself flags (either direction is safe).
    return {
        "customer_id": customer_id,
        "risk_score": score,
        "flag": model_flag or score >= threshold,
        "reasons": reasons,
        "degraded": degraded,
        "risk_profile_used": risk_profile,
        "related_fraud_signals": related_fraud_signals,
    }


def _validate_fraud_assessment(assessment) -> tuple[float, bool, list]:
    """Coerces an LLM fraud assessment into (score in [0,1], flag, reasons).
    Raises ValueError when the score is missing or non-numeric - a string
    "false" flag used to be truthy, and a missing key crashed the case."""
    if not isinstance(assessment, dict) or "risk_score" not in assessment:
        raise ValueError(f"fraud assessment missing risk_score: {assessment!r:.200}")
    score = min(max(float(assessment["risk_score"]), 0.0), 1.0)
    flag = assessment.get("flag", False)
    if isinstance(flag, str):
        flag = flag.strip().lower() in ("true", "1", "yes")
    reasons = assessment.get("reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return score, bool(flag), [str(r) for r in reasons]


def run_inventory_agent(db: Session, line_items: list, case_id: str = None) -> dict:
    """Fixed sequence: for each line item, query sellable stock across all
    warehouses, flag any item with insufficient sellable quantity. Always
    uses sellable_qty, never on_hand_qty alone (edge case 3.2)."""
    results = []
    any_shortfall = False
    for item in line_items:
        sku = item["sku"]
        requested_qty = item.get("qty", 1)
        stock_by_warehouse = record_tool_call(db, case_id or sku, "wms.get_stock", False, wms.get_stock, db, sku)
        total_sellable = sum(s["sellable_qty"] for s in stock_by_warehouse)
        shortfall = total_sellable < requested_qty
        any_shortfall = any_shortfall or shortfall
        results.append({
            "sku": sku,
            "requested_qty": requested_qty,
            "total_sellable_qty": total_sellable,
            "sufficient": not shortfall,
            "warehouses": stock_by_warehouse,
        })
    return {"items": results, "any_shortfall": any_shortfall}


def run_customer_context_agent(db: Session, customer_id: str) -> dict:
    """Fixed sequence: pull risk profile (reused from episodic memory) plus
    a simple LTV-tier heuristic derived from total resolved case count.
    A real deployment would pull actual LTV from a CRM/OMS; this project
    has no such system, so tier is derived from available signal only."""
    profile = summarize_customer_risk_profile(db, customer_id)
    total_cases = profile["total_return_cases"]
    if total_cases >= 20:
        tier = "high_engagement"
    elif total_cases >= 5:
        tier = "regular"
    else:
        tier = "standard"
    return {
        "customer_id": customer_id,
        "tier": tier,
        "total_return_cases": total_cases,
        "most_recent_episode": profile["most_recent_episode"],
    }
