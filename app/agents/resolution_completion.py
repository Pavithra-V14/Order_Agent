"""
Shared resolution-completion logic. Found and fixed together, since they
share the same root cause: this exact sequence (execute -> mark resolved
-> audit -> notify) previously existed ONLY inside the human-escalation-
decision endpoint - meaning a case that genuinely routed to AUTO_EXECUTE
had NO real code path that actually executed it or marked it resolved.
scripts/full_pipeline_demo.py worked around this by forcibly escalating
every case regardless of its real routing outcome, specifically so it
could route through the one completion path that existed.

The second, related gap this closes: log_episode() (the long-term-memory
WRITE path) was fully implemented and its own docstring documented it as
"Called by the Orchestrator on case resolution" - but nothing in the
real application ever called it. Every fraud/customer-context history
lookup was therefore guaranteed to return empty, regardless of how many
cases had actually been resolved for that customer - found by a
developer noticing their Neo4j graph stayed completely empty after a
real, successful pipeline run, not by inspection.
"""
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.db import ExceptionCase, CaseState, AuditLogEntry
from app.guardrails.schema import ResolutionDecision
from app.agents.execution_agent import execute_resolution, ExecutionStatus
from app.agents.learning_loop import record_resolution_outcome
from app.agents.comms_workflow import send_case_notification
from app.memory.episodic import log_episode
from app.tools.oms import get_order

_EVENT_MAP = {
    "refund": "resolved_refund", "partial_credit": "resolved_partial_credit",
    "reship": "resolved_reship", "deny": "resolved_deny",
}


def complete_resolution(
    db: Session, case: ExceptionCase, proposed_decision: ResolutionDecision,
    final_decision: ResolutionDecision, decided_by: str, action_label: str,
    payment_intent_id: str = None,
) -> dict:
    """Executes a resolution decision and completes the case's
    lifecycle - usable for BOTH the human-approved path (escalations.py)
    and the auto-execute path (orchestrator.py's run_full_case_pipeline),
    which is the whole point: previously these were two different, only
    partially-implemented code paths.

    payment_intent_id: an explicit override, preferred over the order's
    own stored value when given. Found necessary from a real production
    bug: this function previously ALWAYS re-derived payment_intent_id
    from the order's stored DB record, silently ignoring whatever was
    passed to run_full_case_pipeline()'s own payment_intent_id
    parameter. An order created once, then reused across many later
    demo runs each creating a FRESH real Stripe payment, kept forever
    executing against the ORIGINAL stale payment_intent_id stored on
    the order from whenever it was first created — a real, reproducible
    "No such payment_intent" error, since passing a new ID to the
    top-level pipeline function had no actual effect on what execution
    used. Falls back to the order's stored value when no override is
    given, preserving the existing escalations.py caller's behavior.
    """
    record_resolution_outcome(
        db, case_id=case.id, cluster_key=f"{case.exception_type}_{case.channel}",
        case_feature_summary=f"{case.exception_type} case, channel={case.channel}, fraud_flag={case.fraud_flag}",
        agent_proposed_resolution=proposed_decision.model_dump(mode="json"),
        human_final_resolution=final_decision.model_dump(mode="json"),
    )

    order = get_order(db, case.order_id)
    resolved_payment_intent_id = payment_intent_id or (order.get("payment_intent_id") if order else None)
    exec_result = execute_resolution(
        db, case_id=case.id, decision=final_decision, order_id=case.order_id,
        payment_intent_id=resolved_payment_intent_id,
    )

    case.resolution_decision = final_decision.model_dump(mode="json")
    case.execution_result = {"status": exec_result.status.value, "result": exec_result.result}

    if exec_result.status == ExecutionStatus.PENDING_RETRY:
        db.add(AuditLogEntry(case_id=case.id, actor=decided_by, action=action_label,
                              detail={"outcome": "execution_pending_retry", "error": exec_result.error}))
        db.commit()
        return {
            "case_id": case.id, "outcome": "execution_pending_retry",
            "execution": exec_result.result, "error": exec_result.error,
        }

    case.state = CaseState.RESOLVED
    db.add(AuditLogEntry(case_id=case.id, actor=decided_by, action=action_label,
                          detail={"final_resolution": final_decision.model_dump(mode="json")}))

    # THE previously-missing piece: record this resolution as an episode
    # in long-term customer memory, so future fraud/customer-context
    # lookups for this customer actually have real history to draw on.
    # Wrapped so a memory-write failure never blocks the case from
    # actually resolving.
    try:
        log_episode(
            db, customer_id=case.customer_id, episode_type="case_resolved",
            content={
                "exception_type": case.exception_type,
                "action": final_decision.action.value,
                "amount_usd": final_decision.amount_usd,
            },
            occurred_at=datetime.now(timezone.utc), case_id=case.id,
        )
    except Exception as e:
        import logging
        logging.getLogger("resolution_completion").warning("log_episode failed (non-fatal): %s", e)
        # A real production enterprise system needs this to be visible
        # and monitorable, not just a warning line in a log file nobody
        # may ever read. Found directly: a genuine Pydantic validation
        # error inside Graphiti's own internal LLM call silently meant
        # this customer's case-resolution history never made it into
        # the fraud-risk knowledge graph — correctly non-fatal to THIS
        # case, but a real, cumulative gap if it keeps happening
        # unnoticed. Uses the same alerting mechanism already in place
        # for circuit breaker trips, not a new, separate mechanism.
        from app.core.alerting import send_alert
        try:
            send_alert(db, "log_episode_failure", {
                "case_id": case.id, "customer_id": case.customer_id, "error": str(e),
            })
        except Exception:
            pass  # alerting itself must never be what breaks case resolution

    db.commit()

    send_case_notification(case.customer_id, _EVENT_MAP[final_decision.action.value],
                            amount=final_decision.amount_usd,
                            tracking_number=exec_result.result.get("tracking_number", ""))

    return {"case_id": case.id, "outcome": "resolved", "final_action": final_decision.action.value,
            "execution": exec_result.result}
