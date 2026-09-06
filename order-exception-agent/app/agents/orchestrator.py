"""
Orchestrator - LangGraph state machine implementing the case lifecycle
(architecture doc Part 3): detected -> diagnosing -> [decided/escalated/
executing/verifying/resolved come in Phases 7-8, out of scope here].

Topology: hierarchical supervisor + specialist workers, per Part 3's
justification (heterogeneous systems, different write-risk profiles per
worker). The Diagnosis Agent (the one open-loop component) and the three
workflow agents (Fraud/Risk, Inventory, Customer Context) run in
parallel once the order is known - they're independent of each other,
so there's no reason to serialize them.

Every transition is persisted to ExceptionCase (app/core/db.py)
immediately, not just held in the LangGraph state - this is what makes a
case survive a process restart mid-diagnosis, per Layer 5's durable-state
requirement.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, END
from sqlalchemy.orm import Session

from app.core.db import ExceptionCase, CaseState, AuditLogEntry, SessionLocal
from app.agents.llm_client import BaseLLMClient, get_llm_client
from app.agents.diagnosis_agent import run_diagnosis, DiagnosisResult
from app.agents.workflow_agents import run_fraud_risk_agent, run_inventory_agent, run_customer_context_agent


class OrchestratorState(TypedDict, total=False):
    case_id: str
    order_id: str
    customer_id: str
    payment_intent_id: str
    tracking_number: str
    address_changed_same_day: bool

    diagnosis_findings: dict
    diagnosis_root_causes: list
    diagnosis_terminated_reason: str

    fraud_result: dict
    inventory_result: dict
    customer_context_result: dict


def _persist_transition(db: Session, case_id: str, new_state: CaseState, actor: str, detail: dict) -> None:
    case = db.get(ExceptionCase, case_id)
    if case is None:
        raise ValueError(f"No such case: {case_id}")
    old_state = case.state
    case.state = new_state
    db.add(AuditLogEntry(
        case_id=case_id, actor=actor, action="state_transition",
        detail={"from": old_state.value if old_state else None, "to": new_state.value, **detail},
    ))
    db.commit()


def make_start_node():
    def start_node(state: OrchestratorState) -> dict:
        db = SessionLocal()
        try:
            _persist_transition(db, state["case_id"], CaseState.DIAGNOSING, actor="orchestrator",
                                 detail={"reason": "diagnosis started"})
        finally:
            db.close()
        return {}
    return start_node


def make_diagnosis_node(llm: BaseLLMClient):
    def diagnosis_node(state: OrchestratorState) -> dict:
        db = SessionLocal()
        try:
            result: DiagnosisResult = run_diagnosis(
                db, llm,
                order_id=state["order_id"],
                payment_intent_id=state.get("payment_intent_id"),
                tracking_number=state.get("tracking_number"),
            )
            db.add(AuditLogEntry(
                case_id=state["case_id"], actor="diagnosis_agent", action="diagnosis_complete",
                detail={"root_causes": result.root_causes, "terminated_reason": result.terminated_reason,
                        "steps_taken": result.steps_taken},
            ))
            db.commit()
        finally:
            db.close()
        return {
            "diagnosis_findings": result.findings,
            "diagnosis_root_causes": result.root_causes,
            "diagnosis_terminated_reason": result.terminated_reason,
        }
    return diagnosis_node


def make_fraud_node(llm: BaseLLMClient):
    def fraud_node(state: OrchestratorState) -> dict:
        db = SessionLocal()
        try:
            result = run_fraud_risk_agent(
                db, llm, customer_id=state["customer_id"],
                address_changed_same_day=state.get("address_changed_same_day", False),
            )
        finally:
            db.close()
        return {"fraud_result": result}
    return fraud_node


def inventory_node(state: OrchestratorState) -> dict:
    db = SessionLocal()
    try:
        from app.tools import oms
        order = oms.get_order(db, state["order_id"])
        line_items = order.get("line_items", []) if order else []
        result = run_inventory_agent(db, line_items)
    finally:
        db.close()
    return {"inventory_result": result}


def customer_context_node(state: OrchestratorState) -> dict:
    db = SessionLocal()
    try:
        result = run_customer_context_agent(db, customer_id=state["customer_id"])
    finally:
        db.close()
    return {"customer_context_result": result}


def make_aggregate_node():
    def aggregate_node(state: OrchestratorState) -> dict:
        db = SessionLocal()
        try:
            case = db.get(ExceptionCase, state["case_id"])
            case.diagnosis = {
                "findings": state.get("diagnosis_findings", {}),
                "root_causes": state.get("diagnosis_root_causes", []),
                "terminated_reason": state.get("diagnosis_terminated_reason", ""),
            }
            fraud = state.get("fraud_result", {})
            case.fraud_risk_score = fraud.get("risk_score")
            case.fraud_flag = "flagged" if fraud.get("flag") else None
            db.add(AuditLogEntry(
                case_id=state["case_id"], actor="orchestrator", action="diagnosis_phase_aggregated",
                detail={"fraud_result": fraud, "inventory_result": state.get("inventory_result"),
                        "customer_context_result": state.get("customer_context_result")},
            ))
            db.commit()
        finally:
            db.close()
        return {}
    return aggregate_node


def build_orchestrator_graph(llm: BaseLLMClient = None):
    """Builds the compiled LangGraph. llm defaults to the fake client
    (get_llm_client()'s current swap point) - pass a real client explicitly
    once one is network-callable."""
    llm = llm or get_llm_client()

    graph = StateGraph(OrchestratorState)
    graph.add_node("start", make_start_node())
    graph.add_node("diagnosis", make_diagnosis_node(llm))
    graph.add_node("fraud", make_fraud_node(llm))
    graph.add_node("inventory", inventory_node)
    graph.add_node("customer_context", customer_context_node)
    graph.add_node("aggregate", make_aggregate_node())

    graph.set_entry_point("start")
    graph.add_edge("start", "diagnosis")
    graph.add_edge("start", "fraud")
    graph.add_edge("start", "inventory")
    graph.add_edge("start", "customer_context")
    graph.add_edge("diagnosis", "aggregate")
    graph.add_edge("fraud", "aggregate")
    graph.add_edge("inventory", "aggregate")
    graph.add_edge("customer_context", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


def run_orchestrator_for_case(case_id: str, order_id: str, customer_id: str,
                               payment_intent_id: str = None, tracking_number: str = None,
                               address_changed_same_day: bool = False,
                               llm: BaseLLMClient = None) -> dict:
    """Convenience entry point: runs the full diagnosis phase for one case
    and returns the final state (also persisted to the DB as a side effect)."""
    compiled = build_orchestrator_graph(llm)
    initial_state = {
        "case_id": case_id,
        "order_id": order_id,
        "customer_id": customer_id,
        "payment_intent_id": payment_intent_id,
        "tracking_number": tracking_number,
        "address_changed_same_day": address_changed_same_day,
    }
    return compiled.invoke(initial_state)
