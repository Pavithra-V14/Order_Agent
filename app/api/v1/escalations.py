from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db, ExceptionCase, CaseState, AuditLogEntry
from app.guardrails.schema import ResolutionDecision
from app.agents.execution_agent import execute_resolution, ExecutionStatus
from app.agents.learning_loop import record_resolution_outcome
from app.agents.comms_workflow import send_case_notification
from app.tools.oms import get_order

router = APIRouter(prefix="/escalations", tags=["escalations"])


class EscalationSummary(BaseModel):
    case_id: str
    order_id: str
    customer_id: str
    exception_type: str
    fraud_flag: str | None = None
    proposed_resolution: dict | None = None
    priority_score: float


def _priority_score(case: ExceptionCase) -> float:
    """Higher = more urgent. Fraud-flagged cases and higher-value
    proposed resolutions jump the queue, per edge case 9.1's priority-
    over-FIFO requirement."""
    score = 0.0
    if case.fraud_flag:
        score += 1000.0
    if case.resolution_decision:
        score += case.resolution_decision.get("amount_usd", 0.0)
    return score


@router.get("", response_model=list[EscalationSummary])
def list_escalations(db: Session = Depends(get_db)):
    cases = db.query(ExceptionCase).filter(ExceptionCase.state == CaseState.ESCALATED).all()
    summaries = [
        EscalationSummary(
            case_id=c.id, order_id=c.order_id, customer_id=c.customer_id,
            exception_type=c.exception_type, fraud_flag=c.fraud_flag,
            proposed_resolution=c.resolution_decision, priority_score=_priority_score(c),
        )
        for c in cases
    ]
    summaries.sort(key=lambda s: s.priority_score, reverse=True)
    return summaries


class EscalationDecisionRequest(BaseModel):
    action: str
    decided_by: str
    final_resolution: dict | None = None


@router.post("/{case_id}/decision")
def decide_escalation(case_id: str, payload: EscalationDecisionRequest, db: Session = Depends(get_db)):
    case = db.get(ExceptionCase, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"No such case: {case_id}")
    if case.state != CaseState.ESCALATED:
        raise HTTPException(status_code=400,
                             detail=f"Case {case_id} is not in ESCALATED state (currently {case.state.value})")
    if case.resolution_decision is None:
        raise HTTPException(status_code=400, detail=f"Case {case_id} has no proposed resolution to decide on")

    proposed = ResolutionDecision.model_validate(case.resolution_decision)

    if payload.action == "reject":
        final = ResolutionDecision(action="deny", amount_usd=0.0, confidence=1.0,
                                    reasoning="Human reviewer rejected the proposed resolution.")
    elif payload.action == "edit":
        if not payload.final_resolution:
            raise HTTPException(status_code=400, detail="final_resolution is required when action='edit'")
        final = ResolutionDecision.model_validate(payload.final_resolution)
    else:
        final = proposed

    record_resolution_outcome(
        db, case_id=case_id, cluster_key=f"{case.exception_type}_{case.channel}",
        case_feature_summary=f"{case.exception_type} case, channel={case.channel}, "
                              f"fraud_flag={case.fraud_flag}",
        agent_proposed_resolution=proposed.model_dump(mode="json"),
        human_final_resolution=final.model_dump(mode="json"),
    )

    order = get_order(db, case.order_id)
    exec_result = execute_resolution(
        db, case_id=case_id, decision=final, order_id=case.order_id,
        payment_intent_id=order.get("payment_intent_id") if order else None,
    )

    case.resolution_decision = final.model_dump(mode="json")
    case.execution_result = {"status": exec_result.status.value, "result": exec_result.result}

    if exec_result.status == ExecutionStatus.PENDING_RETRY:
        db.add(AuditLogEntry(case_id=case_id, actor=payload.decided_by, action="human_decision",
                              detail={"action": payload.action, "outcome": "execution_pending_retry"}))
        db.commit()
        return {"case_id": case_id, "outcome": "execution_pending_retry", "execution": exec_result.result}

    case.state = CaseState.RESOLVED
    db.add(AuditLogEntry(case_id=case_id, actor=payload.decided_by, action="human_decision",
                          detail={"action": payload.action, "final_resolution": final.model_dump(mode="json")}))
    db.commit()

    event_map = {"refund": "resolved_refund", "partial_credit": "resolved_partial_credit",
                 "reship": "resolved_reship", "deny": "resolved_deny"}
    send_case_notification(case.customer_id, event_map[final.action.value],
                            amount=final.amount_usd,
                            tracking_number=exec_result.result.get("tracking_number", ""))

    return {"case_id": case_id, "outcome": "resolved", "final_action": final.action.value,
            "execution": exec_result.result}
