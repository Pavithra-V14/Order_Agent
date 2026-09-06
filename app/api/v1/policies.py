import os
import shutil
import json

from fastapi import APIRouter, UploadFile, HTTPException

from app.workers.job_queue import get_job_queue

router = APIRouter(prefix="/policies", tags=["policies"])

POLICY_UPLOAD_DIR = "data/policies"
REINDEX_STATE_PATH = "data/reindex_state.json"


@router.get("")
def list_policies():
    """Reads the reindex-state file (Phase 3's incremental-reindex
    tracker) to report what's actually been ingested, joined with what's
    physically on disk — so the Policy Document Manager page can show
    both 'files present' and 'files actually indexed and searchable'
    (they can differ if ingestion hasn't run yet for a newly-uploaded file)."""
    files_on_disk = sorted(f for f in os.listdir(POLICY_UPLOAD_DIR) if f.lower().endswith(".pdf")) \
        if os.path.isdir(POLICY_UPLOAD_DIR) else []

    indexed_doc_ids = set()
    if os.path.exists(REINDEX_STATE_PATH):
        with open(REINDEX_STATE_PATH) as f:
            indexed_doc_ids = set(json.load(f).keys())

    return {
        "files_on_disk": files_on_disk,
        "indexed_doc_ids": sorted(indexed_doc_ids),
    }



@router.post("/upload", status_code=202)
async def upload_policy(file: UploadFile):
    """Accepts a new policy PDF. NEVER overwrites an existing file - a
    re-upload of the same filename is rejected outright rather than
    silently replacing a version that orders may already be bound to.
    Ingestion runs async (enqueued)."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    os.makedirs(POLICY_UPLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(POLICY_UPLOAD_DIR, file.filename)

    if os.path.exists(dest_path):
        raise HTTPException(
            status_code=409,
            detail=f"{file.filename} already exists - policy documents are never overwritten. "
                   f"Upload under a new versioned filename instead.",
        )

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = get_job_queue().enqueue("process_policy_upload", {"pdf_path": dest_path})
    return {"filename": file.filename, "job_id": job_id, "status": "queued_for_ingestion"}
