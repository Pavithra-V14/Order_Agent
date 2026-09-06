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
from app.tools import wms


def run_fraud_risk_agent(db: Session, llm: BaseLLMClient, customer_id: str,
                          address_changed_same_day: bool = False) -> dict:
    """Fixed sequence: pull risk profile from memory -> score via rule-based
    assessment -> return. No branching on intermediate results beyond the
    scoring rules themselves - this is a workflow, not a loop."""
    risk_profile = summarize_customer_risk_profile(db, customer_id)
    case_context = {"address_changed_same_day": address_changed_same_day}
    assessment = llm.assess_fraud_risk(case_context, risk_profile)
    return {
        "customer_id": customer_id,
        "risk_score": assessment["risk_score"],
        "flag": assessment["flag"],
        "reasons": assessment["reasons"],
        "risk_profile_used": risk_profile,
    }


def run_inventory_agent(db: Session, line_items: list) -> dict:
    """Fixed sequence: for each line item, query sellable stock across all
    warehouses, flag any item with insufficient sellable quantity. Always
    uses sellable_qty, never on_hand_qty alone (edge case 3.2)."""
    results = []
    any_shortfall = False
    for item in line_items:
        sku = item["sku"]
        requested_qty = item.get("qty", 1)
        stock_by_warehouse = wms.get_stock(db, sku)
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
