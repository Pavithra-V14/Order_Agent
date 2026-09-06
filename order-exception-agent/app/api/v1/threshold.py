from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db, ThresholdProposalRecord, ThresholdOverrideRecord
from app.agents.learning_loop import (
    propose_threshold_adjustments, accept_threshold_proposal, reject_threshold_proposal,
)

router = APIRouter(prefix="/threshold-proposals", tags=["threshold"])


@router.get("")
def list_threshold_proposals(status: str = "pending_review", db: Session = Depends(get_db)):
    """Per architecture doc 8.5/checklist Phase 13: 'Wire Threshold Config
    to display Phase 9's proposed-but-unapplied calibration changes with
    an explicit accept action.' This endpoint is that wiring - it reads
    ONLY from ThresholdProposalRecord, never from ThresholdOverrideRecord,
    so a proposal existing here has zero effect on system behavior until
    a human hits accept."""
    q = db.query(ThresholdProposalRecord)
    if status:
        q = q.filter(ThresholdProposalRecord.status == status)
    q = q.order_by(ThresholdProposalRecord.created_at.desc())
    return [
        {
            "id": p.id, "cluster_key": p.cluster_key, "sample_size": p.sample_size,
            "overturn_rate": p.overturn_rate, "current_threshold": p.current_threshold,
            "proposed_threshold": p.proposed_threshold, "rationale": p.rationale,
            "status": p.status, "created_at": p.created_at.isoformat(),
            "decided_by": p.decided_by,
        }
        for p in q.all()
    ]


@router.get("/active-overrides")
def list_active_overrides(db: Session = Depends(get_db)):
    """What's actually in effect right now - separate from the proposals
    list above, so the UI can show 'proposed' vs 'currently active' as
    two clearly distinct states."""
    overrides = db.query(ThresholdOverrideRecord).all()
    return [
        {
            "cluster_key": o.cluster_key, "active_threshold": o.active_threshold,
            "accepted_by": o.accepted_by, "accepted_at": o.accepted_at.isoformat(),
        }
        for o in overrides
    ]


class RunBatchJobRequest(BaseModel):
    current_threshold: float
    min_sample_size: int = 5


@router.post("/run-batch-job")
def run_batch_job(payload: RunBatchJobRequest, db: Session = Depends(get_db)):
    """Manually triggers the weekly batch job (Phase 9) - in production
    this runs on a schedule; exposed here so the UI/demo doesn't need to
    wait a week to see a proposal appear."""
    proposals = propose_threshold_adjustments(
        db, current_threshold=payload.current_threshold, min_sample_size=payload.min_sample_size,
    )
    return {"proposals_created": len(proposals), "cluster_keys": [p.cluster_key for p in proposals]}


class DecisionRequest(BaseModel):
    decided_by: str


@router.post("/{proposal_id}/accept")
def accept_proposal(proposal_id: str, payload: DecisionRequest, db: Session = Depends(get_db)):
    try:
        override = accept_threshold_proposal(db, proposal_id, accepted_by=payload.decided_by)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"cluster_key": override.cluster_key, "active_threshold": override.active_threshold,
            "accepted_by": override.accepted_by}


@router.post("/{proposal_id}/reject")
def reject_proposal(proposal_id: str, payload: DecisionRequest, db: Session = Depends(get_db)):
    try:
        proposal = reject_threshold_proposal(db, proposal_id, rejected_by=payload.decided_by)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"id": proposal.id, "status": proposal.status}
