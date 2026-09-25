from pydantic import BaseModel, ValidationError
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db, ExceptionCase, CaseState
from app.guardrails.schema import ResolutionDecision
from app.agents.resolution_completion import complete_resolution
from app.core.auth import require_readonly, require_cs_agent, CurrentPrincipal

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
def list_escalations(db: Session = Depends(get_db), _auth=Depends(require_readonly)):
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
    final_resolution: dict | None = None
    # decided_by deliberately removed as a client-supplied field — a
    # real, direct audit finding: this endpoint previously trusted
    # WHATEVER string the caller sent as "decided_by", with zero
    # verification that the caller was actually that person. Anyone
    # hitting this endpoint could approve a refund and have the audit
    # trail record it as having been decided by any named human they
    # chose to type in. It's now derived from the AUTHENTICATED API
    # key below instead — a real, verified identity, not a claim.


@router.post("/{case_id}/decision")
def decide_escalation(case_id: str, payload: EscalationDecisionRequest, db: Session = Depends(get_db),
                       current_key: CurrentPrincipal = Depends(require_cs_agent)):
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
        try:
            final = ResolutionDecision.model_validate(payload.final_resolution)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=f"final_resolution is invalid: {e.errors()}")
        # An edit that doesn't restate a citation keeps the one the
        # proposal was grounded in, rather than failing Tier 1 for it.
        if final.cited_policy is None:
            final.cited_policy = proposed.cited_policy
    elif payload.action == "approve":
        final = proposed
    else:
        raise HTTPException(status_code=400, detail="action must be one of: approve, edit, reject")
    final.requires_human_review = False  # a human has now reviewed it

    _enforce_human_decision_limits(final, current_key.role)

    # Atomic claim: only one reviewer can move the case out of ESCALATED.
    # Without this, two concurrent approvals both passed the state check
    # above and both executed.
    claimed = (db.query(ExceptionCase)
               .filter(ExceptionCase.id == case_id, ExceptionCase.state == CaseState.ESCALATED)
               .update({ExceptionCase.state: CaseState.EXECUTING}, synchronize_session=False))
    db.commit()
    if claimed == 0:
        raise HTTPException(status_code=409, detail=f"Case {case_id} was already decided by another reviewer")
    db.refresh(case)

    return complete_resolution(
        db, case=case, proposed_decision=proposed, final_decision=final,
        decided_by=f"human:{current_key.name}", action_label="human_decision",
    )


def _enforce_human_decision_limits(final: ResolutionDecision, role: str) -> None:
    """Human decisions pass the same Tier 2 (structure/PII) and Tier 1
    (hard ceiling, citation for money) checks as automated ones, plus a
    per-role approval limit. Previously an 'edit' executed whatever the
    reviewer typed - a $1,500 uncited refund was confirmed against real
    Stripe test mode."""
    from app.core.config import get_settings
    from app.guardrails.tier1_ceilings import check_tier1_ceilings
    from app.guardrails.tier2_structural import run_tier2

    settings = get_settings()
    tier2 = run_tier2(final.model_dump(mode="json"))
    if not tier2.passed:
        raise HTTPException(status_code=422, detail=f"Decision failed structural/PII checks: {tier2.errors}")
    tier1 = check_tier1_ceilings(final, settings.auto_execute_value_ceiling_usd,
                                 settings.max_single_action_ceiling_usd)
    if not tier1.passed:
        raise HTTPException(status_code=422, detail=f"Decision violates hard limits: {tier1.violations}")

    limits = {"cs_agent": settings.human_approval_limit_cs_agent_usd,
              "admin": settings.human_approval_limit_admin_usd}
    limit = limits.get(role)
    if limit is None:
        raise HTTPException(status_code=403, detail=f"Role '{role}' may not decide escalations")
    if final.amount_usd > limit:
        raise HTTPException(
            status_code=403,
            detail=f"${final.amount_usd:.2f} exceeds the ${limit:.2f} approval limit for role '{role}'. "
                   f"Ask an admin to approve this case.")
