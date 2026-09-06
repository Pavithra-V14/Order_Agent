from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.core.metrics import compute_agent_metrics, compute_tool_metrics, compute_rag_metrics, compute_system_metrics
from app.core.tracing import get_trace

router = APIRouter(tags=["metrics"])

_SCOPE_FUNCS = {
    "agent": compute_agent_metrics,
    "tool": compute_tool_metrics,
    "rag": compute_rag_metrics,
    "system": compute_system_metrics,
}


@router.get("/metrics/{scope}")
def get_metrics(scope: str, db: Session = Depends(get_db)):
    """scope in {agent, tool, rag, system} - per architecture doc 8.7's
    metrics categories, each computed live from trace/audit data."""
    fn = _SCOPE_FUNCS.get(scope)
    if fn is None:
        raise HTTPException(status_code=404, detail=f"Unknown metrics scope {scope!r}. Valid: {list(_SCOPE_FUNCS)}")
    return fn(db)


@router.get("/traces/{case_id}")
def get_case_trace(case_id: str, db: Session = Depends(get_db)):
    """Full input/output/metadata trace for one case."""
    spans = get_trace(db, case_id)
    if not spans:
        raise HTTPException(status_code=404, detail=f"No trace spans found for case {case_id!r}")
    return {"trace_id": case_id, "spans": spans}
