from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.db import get_db
from app.core.config import get_settings

router = APIRouter(tags=["health"])


@router.get("/health")
def health(db: Session = Depends(get_db)):
    """Phase 0 DoD check: confirms the API is up AND the DB is reachable.
    A later phase will extend this to also ping Qdrant / the embedding
    backend, once those are wired in (Phase 3)."""
    settings = get_settings()
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    return {
        "status": "ok" if db_ok else "degraded",
        "app": settings.app_name,
        "environment": settings.environment,
        "database": "ok" if db_ok else "unreachable",
        "database_url_kind": settings.database_url.split(":")[0],
    }
