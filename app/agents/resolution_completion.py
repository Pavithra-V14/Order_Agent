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
from app.agents.verification_agent import (verify_refund, verify_reship, VerificationResult,
                                           VerificationStatus)
from app.guardrails.schema import ResolutionAction
from app.agents.learning_loop import record_resolution_outcome
from app.agents.comms_workflow import send_case_notification
from app.memory.episodic import log_episode_async
from app.tools.oms import get_order

_EVENT_MAP = {
    "refund": "resolved_refund", "partial_credit": "resolved_partial_credit",
    "reship": "resolved_reship", "deny": "resolved_deny",
}


def complete_resolution(
    db: Session, case: ExceptionCase, proposed_decision: ResolutionDecision,
    final_decision: ResolutionDecision, decided_by: str, action_label: str,
    payment_intent_id: str = None,
    record_outcome: bool = True,
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
    # record_outcome=False for reconciler retries: the same decision must
    # not be counted twice in the learning loop's overturn statistics.
    if record_outcome:
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

    # Real bug found while manually verifying Stage 3's similar_past_cases
    # in the UI, not assumed: orchestrator.py's aggregate step adds
    # similar_past_cases onto case.resolution_decision AFTER computing
    # the decision, but this function then REASSIGNED the whole
    # resolution_decision dict from final_decision alone - which has no
    # knowledge of similar_past_cases - silently discarding it for
    # EVERY resolved case, auto-executed or otherwise. Confirmed by
    # instrumenting both call sites directly: the dict correctly
    # contained similar_past_cases right up until this exact
    # reassignment ran. Preserved here explicitly rather than trusting
    # a full reassignment to keep data added elsewhere.
    existing_similar_past_cases = (case.resolution_decision or {}).get("similar_past_cases")
    case.resolution_decision = final_decision.model_dump(mode="json")
    if existing_similar_past_cases:
        case.resolution_decision["similar_past_cases"] = existing_similar_past_cases
    case.execution_result = {"status": exec_result.status.value, "result": exec_result.result,
                             "idempotency_key": exec_result.idempotency_key,
                             "payment_intent_id": resolved_payment_intent_id}

    if exec_result.status in (ExecutionStatus.PENDING_RETRY, ExecutionStatus.FAILED):
        # Transient -> PENDING_RETRY, picked up by the reconciler
        # (app/workers/reconciler.py). Permanent -> back to a human, since
        # retrying the identical request can never succeed. Previously
        # both left the case in whatever state it was in, owned by nobody.
        transient = exec_result.status == ExecutionStatus.PENDING_RETRY
        outcome = "execution_pending_retry" if transient else "execution_failed"
        new_state = CaseState.PENDING_RETRY if transient else CaseState.ESCALATED
        db.add(AuditLogEntry(case_id=case.id, actor=decided_by, action=action_label,
                              detail={"outcome": outcome, "error": exec_result.error,
                                      "from": case.state.value, "to": new_state.value}))
        case.state = new_state
        db.commit()
        if not transient:
            from app.core.alerting import send_alert
            send_alert(db, "execution_failed", {"case_id": case.id, "action": final_decision.action.value,
                                                 "error": exec_result.error})
        _notify_safely(case.customer_id, "pending_retry" if transient else "escalated")
        return {
            "case_id": case.id, "outcome": outcome,
            "execution": exec_result.result, "error": exec_result.error,
        }

    verification = verify_execution(db, final_decision, exec_result)
    case.execution_result["verification"] = {"status": verification.status.value, "detail": verification.detail}
    if verification.status == VerificationStatus.VERIFICATION_FAILED:
        db.add(AuditLogEntry(case_id=case.id, actor="verification_agent", action="verification_failed",
                              detail={"detail": verification.detail, "from": case.state.value, "to": "escalated"}))
        case.state = CaseState.ESCALATED
        db.commit()
        from app.core.alerting import send_alert
        send_alert(db, "verification_failed", {"case_id": case.id, "detail": verification.detail})
        return {"case_id": case.id, "outcome": "verification_failed",
                "execution": exec_result.result, "error": verification.detail}

    # NOT_YET_VERIFIABLE (e.g. a label the carrier hasn't scanned yet):
    # the action happened, so the customer is notified and memory is
    # written below, but the case stays VERIFYING until the reconciler
    # confirms it.
    verified = verification.status == VerificationStatus.VERIFIED
    case.state = CaseState.RESOLVED if verified else CaseState.VERIFYING
    if verified:
        case.resolved_at = datetime.now(timezone.utc)
    case.verification_result = {"status": verification.status.value, "detail": verification.detail}
    db.add(AuditLogEntry(case_id=case.id, actor=decided_by, action=action_label,
                          detail={"final_resolution": final_decision.model_dump(mode="json"),
                                  "verification": verification.status.value}))

    # Stage 2 memory upgrade: the Summary Buffer's job (recent-step
    # working memory FOR AN IN-PROGRESS diagnosis) ends once a case is
    # actually resolved - the final decision + this audit entry already
    # capture what mattered. Evicting here does two real things: drops
    # the in-process buffer (app/memory/summary_buffer.py's _buffers
    # dict would otherwise accumulate one entry per case ever diagnosed
    # for the process's whole lifetime - a real, slow memory leak) and
    # clears the now-redundant DB copy. Non-fatal, same discipline as
    # log_episode below - a cleanup failure must never block a case from
    # actually resolving.
    try:
        from app.memory.summary_buffer import evict_buffer
        evict_buffer(case.id)
        case.working_memory_summary = None
    except Exception as e:
        import logging
        logging.getLogger("resolution_completion").warning("evict_buffer failed (non-fatal): %s", e)

    # THE previously-missing piece: record this resolution as an episode
    # in long-term customer memory, so future fraud/customer-context
    # lookups for this customer actually have real history to draw on.
    # Wrapped so a memory-write failure never blocks the case from
    # actually resolving.
    try:
        from app.memory.episodic import log_episode_async
        log_episode_async(
            customer_id=case.customer_id, episode_type="case_resolved",
            content={
                "exception_type": case.exception_type,
                "action": final_decision.action.value,
                "amount_usd": final_decision.amount_usd,
                # Stage 1 memory upgrade: carries this order's payment
                # fingerprint into long-term memory so a LATER case for a
                # DIFFERENT customer can be cross-checked against it (see
                # app/memory/graphiti_adapter.py's
                # find_related_fraud_signals). `order` was already
                # fetched above for the execution step - reused here
                # rather than a second lookup. None when the order has
                # no fingerprint (no card component, or not yet wired to
                # a real gateway) - a normal, expected value.
                "payment_fingerprint": order.get("payment_fingerprint") if order else None,
            },
            occurred_at=datetime.now(timezone.utc), case_id=case.id,
        )
        # Follow-up: also logs the SAME fingerprint into the shared
        # cross-customer graph (app/memory/graphiti_adapter.py's
        # log_cross_customer_signal_async) - a genuine, best-effort
        # attempt to let Graphiti's OWN extraction relate this signal
        # across customers, complementing (not replacing) the
        # deterministic Cypher query above. Skipped when there's no
        # fingerprint - nothing shareable to log.
        fingerprint = order.get("payment_fingerprint") if order else None
        if fingerprint:
            from app.memory.graphiti_adapter import log_cross_customer_signal_async
            log_cross_customer_signal_async(
                customer_id=case.customer_id, signal_type="payment_fingerprint",
                signal_value=fingerprint, case_id=case.id,
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

    # Tier 3 judge (app/guardrails/tier3_judge.py) — deliberately
    # SAMPLED and enqueued as a background job, never run inline: per
    # architecture doc 8.6, this must never sit on the critical path
    # of an already-completed decision. Found fully implemented but
    # never actually triggered from anywhere during a direct audit —
    # this sampling call is what closes that gap. 15% sample rate is a
    # reasonable default for catching policy-citation/reasoning-quality
    # drift over time without judging every single case.
    import random
    if random.random() < 0.15:
        try:
            from app.workers.job_queue import get_job_queue
            get_job_queue().enqueue("tier3_judge_sample", {
                "case_id": case.id, "decision": final_decision.model_dump(mode="json"),
            })
        except Exception:
            pass  # sampling itself must never be what breaks case resolution

    return {"case_id": case.id, "outcome": "resolved" if verified else "awaiting_verification",
            "final_action": final_decision.action.value, "execution": exec_result.result,
            "verification": verification.status.value}


def verify_execution(db: Session, decision: ResolutionDecision, exec_result) -> VerificationResult:
    """Independent confirmation that the executed action actually took
    effect (app/agents/verification_agent.py) - previously implemented and
    tested but never called, so RESOLVED only ever meant "the call
    returned"."""
    if exec_result.status == ExecutionStatus.NO_ACTION_NEEDED:
        return VerificationResult(status=VerificationStatus.VERIFIED, detail="No action to verify.")
    try:
        if decision.action == ResolutionAction.RESHIP:
            return verify_reship(db, exec_result.idempotency_key)
        return verify_refund(db, exec_result.idempotency_key, refund_id=(exec_result.result or {}).get("id"))
    except Exception as e:
        # A verification lookup failing (provider hiccup) is not evidence
        # the action failed - leave it for the reconciler to recheck.
        return VerificationResult(status=VerificationStatus.NOT_YET_VERIFIABLE,
                                   detail=f"verification lookup failed, will retry: {e}")


def _notify_safely(customer_id: str, event: str, **kwargs) -> None:
    try:
        send_case_notification(customer_id, event, **kwargs)
    except Exception as e:
        import logging
        logging.getLogger("resolution_completion").warning("notification %s failed (non-fatal): %s", event, e)
