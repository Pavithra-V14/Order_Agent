"""
Case reconciler - the owner of every case that stopped moving.

Before this existed, four kinds of case sat forever with nobody
responsible for them: PENDING_RETRY (a transient execution failure),
VERIFYING (an action taken but not yet confirmed by the provider), and
cases stuck mid-pipeline in DETECTED/DIAGNOSING/DECIDED/EXECUTING after a
crash or an unhandled error. Each pass here moves every such case forward
or hands it to a human, with an audit entry and an alert.

Like every other recurring job in this project it has no in-process
scheduler: trigger it from cron / the ops runbook via
POST /admin/reconcile, or `python -m app.workers.reconciler`.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

logger = logging.getLogger("reconciler")

_MAX_PIPELINE_RERUNS = 2


def _aware(dt: datetime | None) -> datetime | None:
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _audit(db: Session, case, action: str, detail: dict) -> None:
    from app.core.db import AuditLogEntry
    db.add(AuditLogEntry(case_id=case.id, actor="system:reconciler", action=action, detail=detail))


def _escalate(db: Session, case, reason: str) -> None:
    """Hands a case to the human queue. The escalation endpoint needs a
    proposal to decide on, so a case that never got one receives an
    explicit no-action proposal the reviewer can edit."""
    from app.core.db import CaseState
    from app.core.alerting import send_alert
    if not case.resolution_decision:
        case.resolution_decision = {
            "action": "deny", "amount_usd": 0.0, "confidence": 0.0, "requires_human_review": True,
            "reasoning": f"No automated proposal exists - {reason} A reviewer must choose the resolution.",
            "cited_policy": None,
        }
    _audit(db, case, "state_transition", {"from": case.state.value, "to": "escalated", "reason": reason})
    case.state = CaseState.ESCALATED
    db.commit()
    send_alert(db, "case_escalated_by_reconciler", {"case_id": case.id, "reason": reason})


def _retry_execution(db: Session, case) -> str:
    from app.guardrails.schema import ResolutionDecision
    from app.agents.resolution_completion import complete_resolution
    decision = ResolutionDecision.model_validate(case.resolution_decision)
    pi = (case.execution_result or {}).get("payment_intent_id")
    out = complete_resolution(db, case=case, proposed_decision=decision, final_decision=decision,
                              decided_by="system:reconciler", action_label="execution_retry",
                              payment_intent_id=pi, record_outcome=False)
    return out["outcome"]


def _recheck_verification(db: Session, case, verify_timeout_hours: float) -> str:
    from app.core.db import CaseState
    from app.guardrails.schema import ResolutionDecision
    from app.agents.execution_agent import ExecutionResult, ExecutionStatus
    from app.agents.resolution_completion import verify_execution
    from app.agents.verification_agent import VerificationStatus

    decision = ResolutionDecision.model_validate(case.resolution_decision)
    ex = case.execution_result or {}
    exec_result = ExecutionResult(status=ExecutionStatus.EXECUTED, result=ex.get("result") or {},
                                  idempotency_key=ex.get("idempotency_key") or "")
    v = verify_execution(db, decision, exec_result)
    case.verification_result = {"status": v.status.value, "detail": v.detail,
                                "checked_at": datetime.now(timezone.utc).isoformat()}
    if v.status == VerificationStatus.VERIFIED:
        _audit(db, case, "state_transition", {"from": "verifying", "to": "resolved", "detail": v.detail})
        case.state = CaseState.RESOLVED
        case.resolved_at = datetime.now(timezone.utc)
        db.commit()
        return "resolved"
    if v.status == VerificationStatus.VERIFICATION_FAILED:
        _escalate(db, case, f"verification failed: {v.detail}.")
        return "escalated"
    age = datetime.now(timezone.utc) - _aware(case.updated_at or case.created_at)
    if age > timedelta(hours=verify_timeout_hours):
        _escalate(db, case, f"still unverified after {verify_timeout_hours:.0f}h: {v.detail}.")
        return "escalated"
    db.commit()
    return "still_verifying"


def _recover_stuck(db: Session, case) -> str:
    """A case stuck before a decision is re-run through the pipeline
    (reads are safe to repeat; money movement is order-keyed and
    idempotent). After _MAX_PIPELINE_RERUNS it goes to a human."""
    from app.core.db import AuditLogEntry, CaseState
    from app.core.config import get_settings
    from app.tools.oms import get_order

    reruns = db.query(AuditLogEntry).filter(AuditLogEntry.case_id == case.id,
                                            AuditLogEntry.action == "reconciler_rerun").count()
    order = get_order(db, case.order_id)
    if case.state == CaseState.EXECUTING or reruns >= _MAX_PIPELINE_RERUNS or order is None:
        _escalate(db, case, f"stuck in {case.state.value} (pipeline reruns: {reruns}).")
        return "escalated"

    _audit(db, case, "reconciler_rerun", {"from": case.state.value, "attempt": reruns + 1})
    db.commit()
    from app.agents.orchestrator import run_full_case_pipeline
    settings = get_settings()
    try:
        run_full_case_pipeline(
            db, case_id=case.id, order_id=case.order_id, customer_id=case.customer_id,
            order_amount_usd=order["total_amount_usd"],
            auto_execute_confidence_threshold=settings.default_auto_execute_confidence_threshold,
            auto_execute_value_ceiling_usd=settings.default_auto_execute_value_ceiling_usd,
            payment_intent_id=order.get("payment_intent_id"),
        )
        return "rerun"
    except Exception as e:
        db.rollback()
        logger.warning("reconciler rerun failed for %s: %s", case.id, e)
        return "rerun_failed"


def run_reconciliation(db: Session | None = None, stale_after_minutes: float = 30.0,
                       verify_timeout_hours: float = 72.0) -> dict:
    """One reconciliation pass. Returns counts per outcome and the ids touched."""
    from app.core.db import ExceptionCase, CaseState

    own_session = db is None
    if own_session:
        from app.core.db import SessionLocal
        db = SessionLocal()
    summary = {"retried": [], "verified": [], "rerun": [], "escalated": [], "unchanged": [], "errors": []}
    try:
        stale_before = datetime.now(timezone.utc) - timedelta(minutes=stale_after_minutes)
        stuck_states = (CaseState.DETECTED, CaseState.DIAGNOSING, CaseState.DECIDED, CaseState.EXECUTING)
        candidates = db.query(ExceptionCase).filter(ExceptionCase.state.in_(
            (CaseState.PENDING_RETRY, CaseState.VERIFYING) + stuck_states)).all()

        for case in candidates:
            try:
                if case.state == CaseState.PENDING_RETRY:
                    out = _retry_execution(db, case)
                    summary["retried" if out != "execution_pending_retry" else "unchanged"].append(case.id)
                elif case.state == CaseState.VERIFYING:
                    out = _recheck_verification(db, case, verify_timeout_hours)
                    key = {"resolved": "verified", "escalated": "escalated"}.get(out, "unchanged")
                    summary[key].append(case.id)
                elif _aware(case.updated_at or case.created_at) < stale_before:
                    out = _recover_stuck(db, case)
                    summary["escalated" if out == "escalated" else "rerun"].append(case.id)
                else:
                    summary["unchanged"].append(case.id)
            except Exception as e:
                db.rollback()
                logger.exception("reconciliation failed for case %s", case.id)
                summary["errors"].append({"case_id": case.id, "error": str(e)})
        return {**{k: len(v) for k, v in summary.items()}, "case_ids": summary}
    finally:
        if own_session:
            db.close()


class ReconcilerScheduler:
    """Runs run_reconciliation() every `interval_seconds` on a daemon
    thread inside the API process. Off unless RECONCILER_INTERVAL_SECONDS
    > 0 - enable it on exactly ONE instance (or use cron against
    POST /admin/reconcile instead). A failing pass is logged and alerted,
    never allowed to kill the loop."""

    def __init__(self, interval_seconds: float):
        import threading
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = None
        self.passes = 0

    def start(self) -> None:
        import threading
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="reconciler")
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                summary = run_reconciliation()
                self.passes += 1
                if any(summary.get(k) for k in ("retried", "verified", "rerun", "escalated", "errors")):
                    logger.info("reconciler pass: %s", {k: v for k, v in summary.items() if k != "case_ids"})
            except Exception as e:
                logger.exception("reconciler pass failed")
                try:
                    from app.core.db import SessionLocal
                    from app.core.alerting import send_alert
                    db = SessionLocal()
                    try:
                        send_alert(db, "reconciler_failure", {"error": str(e)})
                    finally:
                        db.close()
                except Exception:
                    pass


if __name__ == "__main__":
    import json
    from app.core.db import init_db
    init_db()
    print(json.dumps(run_reconciliation(), indent=2, default=str))
