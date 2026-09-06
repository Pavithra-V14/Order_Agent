from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.db import get_db, AuditLogEntry

router = APIRouter(prefix="/audit-log", tags=["audit"])


@router.get("")
def list_audit_log(case_id: str = None, limit: int = 200, db: Session = Depends(get_db)):
    q = db.query(AuditLogEntry)
    if case_id:
        q = q.filter(AuditLogEntry.case_id == case_id)
    q = q.order_by(AuditLogEntry.timestamp.desc()).limit(limit)
    return [
        {
            "id": e.id, "case_id": e.case_id, "actor": e.actor, "action": e.action,
            "detail": e.detail, "timestamp": e.timestamp.isoformat(),
        }
        for e in q.all()
    ]
