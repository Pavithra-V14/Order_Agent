import os
import shutil
import json
import time

from fastapi import APIRouter, UploadFile, HTTPException

router = APIRouter(prefix="/policies", tags=["policies"])

POLICY_UPLOAD_DIR = "data/policies"
REINDEX_STATE_PATH = "data/reindex_state.json"


def _ingest_with_retry(pdf_path: str, max_attempts: int = 3):
    """Retries ingestion with short backoff before giving up — a real,
    principled robustness improvement for "production enterprise level":
    ingestion depends on real cloud services (Qdrant, and Mistral if
    configured), and a single transient hiccup (a cold Qdrant Cloud
    connection, a brief rate-limit) shouldn't force a person to manually
    click "Retry ingestion" for something that would have succeeded on
    its own a second later. Genuinely bad content (a malformed PDF that
    can't be parsed) will fail identically on every attempt and still
    surfaces as a real error after exhausting retries — this doesn't
    mask real failures, only smooths over transient ones, matching the
    same retry-then-fail-honestly principle already used for tool calls
    via the circuit breaker.
    """
    from app.rag.ingestion import ingest_policy_pdf
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return ingest_policy_pdf(pdf_path)
        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                from app.core.console_log import log_warning
                log_warning("ingestion", f"attempt {attempt}/{max_attempts} failed for {pdf_path}: {e} — retrying")
                time.sleep(0.5 * attempt)
    raise last_error


@router.get("")
def list_policies():
    """Reads the reindex-state file to report what's actually been
    ingested, joined with what's physically on disk."""
    files_on_disk = sorted(f for f in os.listdir(POLICY_UPLOAD_DIR) if f.lower().endswith(".pdf")) \
        if os.path.isdir(POLICY_UPLOAD_DIR) else []

    indexed_doc_ids = set()
    if os.path.exists(REINDEX_STATE_PATH):
        with open(REINDEX_STATE_PATH) as f:
            indexed_doc_ids = set(json.load(f).keys())

    file_status = {}
    for fname in files_on_disk:
        stem = fname.replace(".pdf", "").replace(".PDF", "")
        file_status[fname] = {"status": "indexed"} if stem in indexed_doc_ids else \
            {"status": "not_indexed", "detail": "upload again, or run scripts/run_ingestion.py"}

    return {
        "files_on_disk": files_on_disk,
        "indexed_doc_ids": sorted(indexed_doc_ids),
        "file_status": file_status,
    }


@router.post("/upload")
async def upload_policy(file: UploadFile):
    """Accepts a new policy PDF and ingests it IMMEDIATELY, synchronously,
    within this same request — NOT via the async job queue.

    This is a deliberate architectural decision, not an oversight: policy
    ingestion is a rare, admin-triggered action (someone uploading a new
    policy version), not a high-throughput webhook that needs to ack in
    milliseconds. Real ingestion of one PDF takes low single-digit
    seconds. Routing this through the async job queue instead repeatedly
    caused the exact same class of confusion in practice: uploads sat at
    "pending" forever whenever REDIS_URL was configured but
    scripts/run_rq_worker.py wasn't running as a separate process — a
    dependency easy to forget for an action used only occasionally. This
    removes that entire failure mode: a successful HTTP response from
    this endpoint means the document is ALREADY searchable, no
    background worker required, no polling needed. The async job queue
    remains exactly right for OMS/payment/carrier webhooks, which
    genuinely need to ack fast under real request volume — this is a
    narrower, deliberate exception for an operation that doesn't share
    that requirement.

    NEVER overwrites an existing file - a re-upload of the same filename
    is rejected outright rather than silently replacing a version that
    orders may already be bound to.
    """
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

    try:
        summary = _ingest_with_retry(dest_path)
    except Exception as e:
        # Ingestion failed (e.g. the PDF doesn't match the expected
        # policy-header format) — the file stays on disk (so nothing is
        # silently lost), but the caller learns about the failure
        # IMMEDIATELY, in this same response, not via a job-status poll
        # that might never get checked.
        raise HTTPException(
            status_code=422,
            detail=f"{file.filename} was saved but ingestion failed: {e}",
        )

    return {"filename": file.filename, "status": "indexed", "summary": summary}


@router.post("/{filename}/reingest")
def reingest_policy(filename: str):
    """Retries ingestion for a file that's ALREADY on disk — the piece
    that was missing entirely after making uploads synchronous: if
    ingestion failed the first time (a malformed PDF, a transient RAG
    backend issue), the file was correctly kept on disk rather than
    lost, but there was no way to try again short of deleting it and
    re-uploading under a different name, which defeats the point of
    keeping the original filename. This works on the EXACT SAME file
    already present — not a new upload, so the never-overwrite guard
    (which exists to protect against silently replacing a DIFFERENT
    version) doesn't apply here at all.
    """
    dest_path = os.path.join(POLICY_UPLOAD_DIR, filename)
    if not os.path.exists(dest_path):
        raise HTTPException(status_code=404, detail=f"No such file on disk: {filename}")

    try:
        summary = _ingest_with_retry(dest_path)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Re-ingestion of {filename} failed: {e}")

    return {"filename": filename, "status": "indexed", "summary": summary}
