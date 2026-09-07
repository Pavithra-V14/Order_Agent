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

from app.core.db import ExceptionCase, CaseState, AuditLogEntry
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
        from app.core.db import SessionLocal
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
        from app.core.db import SessionLocal
        from app.core.console_log import log_agent_step
        log_agent_step("diagnosis", "starting", case_id=state["case_id"])
        db = SessionLocal()
        try:
            result: DiagnosisResult = run_diagnosis(
                db, llm,
                order_id=state["order_id"],
                payment_intent_id=state.get("payment_intent_id"),
                tracking_number=state.get("tracking_number"),
                case_id=state["case_id"],
            )
            db.add(AuditLogEntry(
                case_id=state["case_id"], actor="diagnosis_agent", action="diagnosis_complete",
                detail={"root_causes": result.root_causes, "terminated_reason": result.terminated_reason,
                        "steps_taken": result.steps_taken},
            ))
            db.commit()
        finally:
            db.close()
        log_agent_step("diagnosis", f"root_causes={result.root_causes}", case_id=state["case_id"])
        return {
            "diagnosis_findings": result.findings,
            "diagnosis_root_causes": result.root_causes,
            "diagnosis_terminated_reason": result.terminated_reason,
        }
    return diagnosis_node


def make_fraud_node(llm: BaseLLMClient):
    def fraud_node(state: OrchestratorState) -> dict:
        from app.core.db import SessionLocal
        from app.core.console_log import log_agent_step
        log_agent_step("fraud", "starting", case_id=state["case_id"])
        db = SessionLocal()
        try:
            result = run_fraud_risk_agent(
                db, llm, customer_id=state["customer_id"],
                address_changed_same_day=state.get("address_changed_same_day", False),
            )
        finally:
            db.close()
        log_agent_step("fraud", f"risk_score={result.get('risk_score')}, flag={result.get('flag')}", case_id=state["case_id"])
        return {"fraud_result": result}
    return fraud_node


def inventory_node(state: OrchestratorState) -> dict:
    from app.core.db import SessionLocal
    from app.core.console_log import log_agent_step
    log_agent_step("inventory", "starting", case_id=state["case_id"])
    db = SessionLocal()
    try:
        from app.tools import oms
        from app.core.tracing import record_tool_call
        order = record_tool_call(db, state["case_id"], "oms.get_order", False, oms.get_order, db, state["order_id"])
        line_items = order.get("line_items", []) if order else []
        result = run_inventory_agent(db, line_items, case_id=state["case_id"])
    finally:
        db.close()
    log_agent_step("inventory", f"any_shortfall={result.get('any_shortfall')}", case_id=state["case_id"])
    return {"inventory_result": result}


def customer_context_node(state: OrchestratorState) -> dict:
    from app.core.db import SessionLocal
    from app.core.console_log import log_agent_step
    log_agent_step("customer_context", "starting", case_id=state["case_id"])
    db = SessionLocal()
    try:
        result = run_customer_context_agent(db, customer_id=state["customer_id"])
    finally:
        db.close()
    log_agent_step("customer_context", "complete", case_id=state["case_id"])
    return {"customer_context_result": result}


def make_aggregate_node():
    def aggregate_node(state: OrchestratorState) -> dict:
        from app.core.db import SessionLocal
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


def run_full_case_pipeline(
    db: Session, case_id: str, order_id: str, customer_id: str, order_amount_usd: float,
    auto_execute_confidence_threshold: float, auto_execute_value_ceiling_usd: float,
    payment_intent_id: str = None, tracking_number: str = None,
    address_changed_same_day: bool = False, retrieved_policy_doc_id: str = None,
    retrieved_policy_version: str = None, llm: BaseLLMClient = None,
) -> dict:
    """Chains diagnosis -> Resolution-Policy Workflow -> completion, all
    the way to a resolved (or escalated, or blocked) case — closing a
    real, previously-undiscovered gap: this exact sequence existed only
    as MANUAL steps in scripts/full_pipeline_demo.py, never as a single
    reusable function. In particular, if the workflow genuinely routes
    to AUTO_EXECUTE, this is the first place that actually executes and
    resolves it — prior to this function existing, there was no code
    path anywhere that completed a genuinely auto-executed case (only
    the human-escalation-decision endpoint could ever mark a case
    RESOLVED), found because a developer noticed their long-term memory
    graph stayed empty after cases that should have auto-executed.

    This is the function a real webhook-triggered background job should
    call once that wiring exists (still a separate, documented gap: the
    webhook handler itself only creates a DETECTED case today, it does
    not yet enqueue a follow-up job to run this).
    """
    diagnosis_state = run_orchestrator_for_case(
        case_id=case_id, order_id=order_id, customer_id=customer_id,
        payment_intent_id=payment_intent_id, tracking_number=tracking_number,
        address_changed_same_day=address_changed_same_day, llm=llm,
    )

    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow
    from app.core.db import ExceptionCase, CaseState

    result = run_resolution_policy_workflow(
        diagnosis_root_causes=diagnosis_state["diagnosis_root_causes"],
        inventory_result=diagnosis_state["inventory_result"],
        order_amount_usd=order_amount_usd,
        fraud_flag_present=bool(diagnosis_state["fraud_result"]["flag"]),
        auto_execute_confidence_threshold=auto_execute_confidence_threshold,
        auto_execute_value_ceiling_usd=auto_execute_value_ceiling_usd,
        retrieved_policy_doc_id=retrieved_policy_doc_id,
        retrieved_policy_version=retrieved_policy_version,
        db=db, case_id=case_id,
    )

    case = db.get(ExceptionCase, case_id)
    case.resolution_decision = result.decision.model_dump(mode="json")

    if result.routing.value == "auto_execute":
        from app.agents.resolution_completion import complete_resolution
        completion = complete_resolution(
            db, case=case, proposed_decision=result.decision, final_decision=result.decision,
            decided_by="system:auto_execute", action_label="auto_execute",
        )
        return {"routing": result.routing.value, "diagnosis": diagnosis_state, "completion": completion}
    else:
        # ESCALATE or BLOCKED: leave for human review — case.state
        # already reflects this via resolution_policy_workflow's own
        # audit logging; nothing further to execute yet.
        case.state = CaseState.ESCALATED if result.routing.value == "escalate" else case.state
        db.commit()
        return {"routing": result.routing.value, "diagnosis": diagnosis_state, "completion": None}
