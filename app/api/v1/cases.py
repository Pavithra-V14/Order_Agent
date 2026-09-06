"""
Case endpoints — Phase 0 ships a minimal create/get/list slice just to prove
the DB layer and API wiring work end to end. Full webhook-triggered case
creation (per architecture doc 8.1's /webhooks/* table) lands in Phase 12;
diagnosis/decision population lands in Phases 6-8.
"""
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db, ExceptionCase, AuditLogEntry, CaseState

router = APIRouter(prefix="/cases", tags=["cases"])


class CaseCreateRequest(BaseModel):
    order_id: str
    customer_id: str
    channel: str = Field(description="direct | amazon | walmart | ...")
    exception_type: str = Field(description="payment | inventory | carrier | return | fraud")


class CaseResponse(BaseModel):
    id: str
    order_id: str
    customer_id: str
    channel: str
    exception_type: str
    state: str
    diagnosis: dict | None = None
    fraud_risk_score: float | None = None
    fraud_flag: str | None = None
    resolution_decision: dict | None = None
    execution_result: dict | None = None
    verification_result: dict | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


@router.post("", response_model=CaseResponse, status_code=201)
def create_case(payload: CaseCreateRequest, db: Session = Depends(get_db)):
    """Manual case creation, for testing. In production this is called
    internally by the webhook handlers (Phase 12), never directly by a client."""
    case = ExceptionCase(
        order_id=payload.order_id,
        customer_id=payload.customer_id,
        channel=payload.channel,
        exception_type=payload.exception_type,
        state=CaseState.DETECTED,
    )
    db.add(case)
    db.flush()  # get case.id before the audit row references it

    # Every state transition writes an audit row — per architecture doc 8.8 / Layer 13.
    db.add(AuditLogEntry(
        case_id=case.id,
        actor="system",
        action="state_transition",
        detail={"from": None, "to": CaseState.DETECTED.value, "reason": "case created"},
    ))
    db.commit()
    db.refresh(case)
    return case


@router.get("/{case_id}", response_model=CaseResponse)
def get_case(case_id: str, db: Session = Depends(get_db)):
    case = db.get(ExceptionCase, case_id)
    if not case:
        raise HTTPException(status_code=404, detail="case not found")
    return case


@router.get("", response_model=list[CaseResponse])
def list_cases(db: Session = Depends(get_db)):
    return db.query(ExceptionCase).order_by(ExceptionCase.created_at.desc()).all()


@router.post("/{case_id}/reopen", response_model=CaseResponse)
def reopen_case(case_id: str, reason: str = "customer follow-up", db: Session = Depends(get_db)):
    """Explicit reopen path — edge case 6.3: a case reopened after
    'resolved' preserves the audit trail rather than spawning a duplicate
    case. Only RESOLVED cases can be reopened; a case that's still active
    doesn't need this path at all."""
    case = db.get(ExceptionCase, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="case not found")
    if case.state != CaseState.RESOLVED:
        raise HTTPException(status_code=400,
                             detail=f"Only RESOLVED cases can be reopened (case is currently {case.state.value})")

    old_state = case.state
    case.state = CaseState.REOPENED
    case.resolved_at = None
    db.add(AuditLogEntry(
        case_id=case.id, actor="system", action="state_transition",
        detail={"from": old_state.value, "to": CaseState.REOPENED.value, "reason": reason},
    ))
    db.commit()
    db.refresh(case)
    return case
