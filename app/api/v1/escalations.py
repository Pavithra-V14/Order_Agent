from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db, ExceptionCase, CaseState
from app.guardrails.schema import ResolutionDecision
from app.agents.resolution_completion import complete_resolution

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

    return complete_resolution(
        db, case=case, proposed_decision=proposed, final_decision=final,
        decided_by=payload.decided_by, action_label="human_decision",
    )
