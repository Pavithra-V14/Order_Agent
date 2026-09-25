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
from app.agents.llm_client import BaseLLMClient, get_llm_client, set_llm_trace, reset_llm_trace
from app.agents.diagnosis_agent import run_diagnosis, DiagnosisResult
from app.agents.workflow_agents import run_fraud_risk_agent, run_inventory_agent, run_customer_context_agent


class OrchestratorState(TypedDict, total=False):
    case_id: str
    order_id: str
    customer_id: str
    payment_intent_id: str
    tracking_number: str
    carrier: str
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
        trace_token = set_llm_trace(state["case_id"])
        try:
            result: DiagnosisResult = run_diagnosis(
                db, llm,
                order_id=state["order_id"],
                payment_intent_id=state.get("payment_intent_id"),
                tracking_number=state.get("tracking_number"),
                case_id=state["case_id"],
                carrier=state.get("carrier"),
                exception_type=getattr(db.get(ExceptionCase, state["case_id"]), "exception_type", None),
            )
            db.add(AuditLogEntry(
                case_id=state["case_id"], actor="diagnosis_agent", action="diagnosis_complete",
                detail={"root_causes": result.root_causes, "terminated_reason": result.terminated_reason,
                        "steps_taken": result.steps_taken, "case_summary": result.case_summary},
            ))
            db.commit()
        finally:
            reset_llm_trace(trace_token)
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
        trace_token = set_llm_trace(state["case_id"])
        try:
            result = run_fraud_risk_agent(
                db, llm, customer_id=state["customer_id"],
                address_changed_same_day=state.get("address_changed_same_day", False),
                # Stage 1 memory upgrade: lets the fraud agent look up
                # this order's payment_fingerprint and cross-check it
                # against OTHER customers' fraud history (see
                # find_related_fraud_signals). Optional/backward
                # compatible - run_fraud_risk_agent treats a missing
                # order_id exactly like an order with no fingerprint.
                order_id=state.get("order_id"),
            )
        finally:
            reset_llm_trace(trace_token)
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
            case.fraud_reasons = fraud.get("reasons") or None

            # Stage 1 memory upgrade: a REAL, previously-existing gap
            # found while wiring the cross-customer fraud query - the
            # "fraud_flag_raised" episode_type was referenced by
            # app/memory/episodic.py's counting logic and covered by
            # app/memory/eval.py's golden set, but nothing in the live
            # pipeline ever actually WROTE one. summarize_customer_
            # risk_profile()'s fraud_flags_raised count was therefore
            # guaranteed to read 0 for every real customer, regardless
            # of how many fraud flags had actually been raised for them
            # - the counting logic was correct, but had nothing real to
            # count. Same non-fatal discipline as resolution_completion
            # .py's log_episode call: a memory-write failure must never
            # block the diagnosis pipeline from completing.
            if fraud.get("flag"):
                try:
                    from datetime import datetime, timezone
                    from app.memory.episodic import log_episode_async
                    from app.tools.oms import get_order
                    order = get_order(db, state["order_id"])
                    log_episode_async(
                        customer_id=state["customer_id"], episode_type="fraud_flag_raised",
                        content={
                            "reason": "; ".join(fraud.get("reasons", [])) or "fraud risk threshold exceeded",
                            "risk_score": fraud.get("risk_score"),
                            # Included here too (not just on case_resolved)
                            # so a customer's VERY FIRST case - not yet
                            # resolved - is still discoverable by
                            # find_related_fraud_signals if it's the one
                            # that happens to carry the shared fingerprint.
                            "payment_fingerprint": order.get("payment_fingerprint") if order else None,
                        },
                        occurred_at=datetime.now(timezone.utc), case_id=state["case_id"],
                    )
                    # Follow-up: also feeds the shared cross-customer
                    # graph, same reasoning as resolution_completion.py's
                    # identical addition - complementary to, never a
                    # replacement for, the deterministic fraud check.
                    fingerprint = order.get("payment_fingerprint") if order else None
                    if fingerprint:
                        from app.memory.graphiti_adapter import log_cross_customer_signal_async
                        log_cross_customer_signal_async(
                            customer_id=state["customer_id"], signal_type="payment_fingerprint",
                            signal_value=fingerprint, case_id=state["case_id"],
                        )
                except Exception as e:
                    import logging
                    logging.getLogger("orchestrator").warning(
                        "log_episode(fraud_flag_raised) failed (non-fatal): %s", e)
                    try:
                        from app.core.alerting import send_alert
                        send_alert(db, "log_episode_failure", {
                            "case_id": state["case_id"], "customer_id": state["customer_id"], "error": str(e),
                        })
                    except Exception:
                        pass

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
                               llm: BaseLLMClient = None, carrier: str = None) -> dict:
    """Convenience entry point: runs the full diagnosis phase for one case
    and returns the final state (also persisted to the DB as a side effect)."""
    compiled = build_orchestrator_graph(llm)
    initial_state = {
        "case_id": case_id,
        "order_id": order_id,
        "customer_id": customer_id,
        "payment_intent_id": payment_intent_id,
        "tracking_number": tracking_number,
        "carrier": carrier,
        "address_changed_same_day": address_changed_same_day,
    }
    return compiled.invoke(initial_state)


def run_full_case_pipeline(
    db: Session, case_id: str, order_id: str, customer_id: str, order_amount_usd: float,
    auto_execute_confidence_threshold: float, auto_execute_value_ceiling_usd: float,
    payment_intent_id: str = None, tracking_number: str = None,
    address_changed_same_day: bool = False, retrieved_policy_doc_id: str = None,
    retrieved_policy_version: str = None, llm: BaseLLMClient = None, carrier: str = None,
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
        address_changed_same_day=address_changed_same_day, llm=llm, carrier=carrier,
    )

    from app.agents.resolution_policy_workflow import run_resolution_policy_workflow
    from app.core.db import ExceptionCase, CaseState

    # THE previously-missing piece: this function used to accept
    # retrieved_policy_doc_id/version as PARAMETERS, meaning nothing in
    # Always fetch order context (purchase_date, product_category) —
    # needed by the return-window check below regardless of whether RAG
    # retrieval actually runs this call (a caller may have already
    # supplied retrieved_policy_doc_id directly). Moving this outside
    # the "RAG retrieval needed" branch below fixes a real gap: these
    # were previously only computed WHEN retrieval happened, meaning a
    # caller providing the doc_id directly would silently lose access
    # to purchase_date/product_category for the return-window check.
    from app.tools.oms import get_order
    order_for_query = get_order(db, order_id)
    purchase_date = order_for_query["purchase_date"] if order_for_query else None
    if purchase_date and "T" in purchase_date:
        # get_order() returns a full ISO datetime string
        # ("2025-06-15T00:00:00"), but hybrid_search's as_of_date
        # expects a plain date ("2025-06-15") — confirmed as a real
        # bug by actually running this end to end, not assumed:
        # date.fromisoformat() inside build_temporal_filter() cannot
        # parse the full datetime form.
        purchase_date = purchase_date.split("T")[0]
    product_category = None
    if order_for_query and order_for_query.get("line_items"):
        product_category = order_for_query["line_items"][0].get("category")

    return_window_days_by_category = None

    # THE previously-missing piece: this function used to accept
    # retrieved_policy_doc_id/version as PARAMETERS, meaning nothing in
    # the real pipeline ever actually called RAG retrieval at all — a
    # caller had to already know which policy applied and hand it in
    # directly (scripts/full_pipeline_demo.py hardcoded it literally).
    # Found from a live deployment's own metrics: retrieval span count
    # sat at 0 while citing decisions sat at 3 — the resolution workflow
    # was citing policies with nothing ever having actually searched for
    # them, so groundedness could only ever report 0, correctly, for
    # exactly that reason. This block makes the pipeline actually
    # retrieve the applicable policy via a real, traced RAG call.
    if retrieved_policy_doc_id is None:
        doc_type = "fraud_policy" if bool(diagnosis_state["fraud_result"]["flag"]) else "return_policy"
        query_text = "fraud risk assessment" if doc_type == "fraud_policy" else "return window policy"

        if purchase_date:
            from app.rag.traced_retrieval import traced_hybrid_search
            retrieval_results = traced_hybrid_search(
                db, trace_id=case_id, query=query_text, as_of_date=purchase_date,
                doc_type=doc_type, product_category=product_category, top_k=1,
            )
            if retrieval_results:
                retrieved_policy_doc_id = retrieval_results[0].metadata.get("doc_id")
                retrieved_policy_version = retrieval_results[0].metadata.get("version")
                return_window_days_by_category = retrieval_results[0].metadata.get("return_window_days_by_category")

    from app.core.config import get_settings
    from app.agents.learning_loop import get_active_threshold
    settings = get_settings()
    case_row = db.get(ExceptionCase, case_id)
    # A human-accepted threshold override for this case's cluster (Phase 9
    # learning loop) replaces the caller's default. Previously nothing
    # read the override table, so an accepted proposal had no effect.
    if case_row is not None:
        auto_execute_confidence_threshold = get_active_threshold(
            db, f"{case_row.exception_type}_{case_row.channel}", auto_execute_confidence_threshold)
    extra_reasons = []
    if diagnosis_state["fraud_result"].get("degraded"):
        extra_reasons.append("fraud check ran in degraded mode (LLM unavailable or invalid output) - "
                             "human review required")

    result = run_resolution_policy_workflow(
        diagnosis_root_causes=diagnosis_state["diagnosis_root_causes"],
        inventory_result=diagnosis_state["inventory_result"],
        order_amount_usd=order_amount_usd,
        fraud_flag_present=bool(diagnosis_state["fraud_result"]["flag"]),
        auto_execute_confidence_threshold=auto_execute_confidence_threshold,
        auto_execute_value_ceiling_usd=auto_execute_value_ceiling_usd,
        retrieved_policy_doc_id=retrieved_policy_doc_id,
        retrieved_policy_version=retrieved_policy_version,
        purchase_date=purchase_date,
        product_category=product_category,
        return_window_days_by_category=return_window_days_by_category,
        payment_status=(diagnosis_state.get("diagnosis_findings", {}).get("payment") or {}).get("status"),
        diagnosis_findings=diagnosis_state.get("diagnosis_findings") or {},
        extra_escalation_reasons=extra_reasons,
        max_single_action_ceiling_usd=settings.max_single_action_ceiling_usd,
        db=db, case_id=case_id,
    )

    case = db.get(ExceptionCase, case_id)
    case.resolution_decision = result.decision.model_dump(mode="json")
    # Few-shot retrieval context (app/agents/learning_loop.py), embedded
    # alongside the decision rather than as a new DB column — this
    # project uses Base.metadata.create_all() (see main.py's lifespan
    # docstring: "Postgres/staging: swap for Alembic migrations before
    # this ships past Phase 0"), which creates tables on a fresh
    # deployment but never ALTERs an existing one, so adding a new
    # required column here would silently break any already-deployed
    # database. Embedding it in the existing JSON field needs no
    # schema change at all.
    if result.similar_past_cases:
        case.resolution_decision["similar_past_cases"] = result.similar_past_cases

    if result.routing.value == "auto_execute":
        from app.agents.resolution_completion import complete_resolution
        completion = complete_resolution(
            db, case=case, proposed_decision=result.decision, final_decision=result.decision,
            decided_by="system:auto_execute", action_label="auto_execute",
            payment_intent_id=payment_intent_id,
        )
        return {"routing": result.routing.value, "diagnosis": diagnosis_state, "completion": completion}
    else:
        # ESCALATE or BLOCKED: leave for human review — case.state
        # already reflects this via resolution_policy_workflow's own
        # audit logging; nothing further to execute yet.
        # BLOCKED gets its own state: previously it silently stayed in
        # DIAGNOSING, a state no queue, page or reconciler looks at.
        new_state = CaseState.ESCALATED if result.routing.value == "escalate" else CaseState.BLOCKED
        db.add(AuditLogEntry(case_id=case_id, actor="resolution_policy_workflow", action="state_transition",
                              detail={"from": case.state.value, "to": new_state.value,
                                      "routing_reasons": result.routing_reasons}))
        case.state = new_state
        db.commit()
        if new_state == CaseState.ESCALATED:
            from app.agents.resolution_completion import _notify_safely
            _notify_safely(case.customer_id, "escalated")
        return {"routing": result.routing.value, "diagnosis": diagnosis_state, "completion": None}
