import os
import shutil
import json
import time
import glob

from fastapi import APIRouter, UploadFile, HTTPException, Depends
from sqlalchemy.orm import Session

from app.core.auth import require_readonly, require_admin
from app.core.db import get_db, ExceptionCase

router = APIRouter(prefix="/policies", tags=["policies"])

POLICY_UPLOAD_DIR = "data/policies"
REINDEX_STATE_PATH = "data/reindex_state.json"
POLICY_IMAGE_DIR = "data/policy_images"


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
def list_policies(_auth=Depends(require_readonly)):
    """Reads the reindex-state file to report what's actually been
    ingested, joined with what's physically on disk."""
    files_on_disk = sorted(f for f in os.listdir(POLICY_UPLOAD_DIR) if f.lower().endswith(".pdf")) \
        if os.path.isdir(POLICY_UPLOAD_DIR) else []

    state = {}
    if os.path.exists(REINDEX_STATE_PATH):
        with open(REINDEX_STATE_PATH) as f:
            state = json.load(f)
    indexed_doc_ids = set(state.keys())
    # filename -> doc_id, from the same source_filename field
    # delete_policy's _resolve_doc_id_for_filename uses - fixes the
    # same real bug: a file's real doc_id is parsed from its own PDF
    # header, never guaranteed to match its filename, so a naive
    # filename-stem comparison could report "not indexed" for a file
    # that genuinely was, whenever the two didn't happen to match.
    # Single value per doc_id, deliberately - see ingest_policy_pdf's
    # own comment on source_filename for why a later file under the
    # same doc_id is treated as superseding the earlier one, not as
    # co-existing with it.
    filename_to_doc_id = {
        entry["source_filename"]: doc_id
        for doc_id, entry in state.items()
        if isinstance(entry, dict) and entry.get("source_filename")
    }

    file_status = {}
    for fname in files_on_disk:
        matched_doc_id = filename_to_doc_id.get(fname)
        if matched_doc_id is None:
            # No source_filename recorded for this file (either it was
            # never successfully ingested, or it predates this field
            # existing) - fall back to the old stem-matching guess
            # rather than an expensive re-parse of every file on every
            # list call; this list endpoint's badge is advisory display
            # only, not what delete/ingestion logic itself relies on.
            matched_doc_id = fname.replace(".pdf", "").replace(".PDF", "")
        file_status[fname] = {"status": "indexed"} if matched_doc_id in indexed_doc_ids else \
            {"status": "not_indexed", "detail": "upload again, or run scripts/run_ingestion.py"}

    return {
        "files_on_disk": files_on_disk,
        "indexed_doc_ids": sorted(indexed_doc_ids),
        "file_status": file_status,
    }


@router.post("/upload")
async def upload_policy(file: UploadFile, _auth=Depends(require_admin)):
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
def reingest_policy(filename: str, _auth=Depends(require_admin)):
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


def _resolve_doc_id_for_filename(filename: str, full_path: str) -> str:
    """Finds the REAL doc_id for a file on disk - never guesses from
    the filename.

    Found and fixed a real, serious bug directly from a user report:
    an earlier version of this used `filename.replace(".pdf", "")` as
    the doc_id, which is only correct when someone happens to name
    their file identically to the "Document ID:" text inside the PDF
    itself - not guaranteed at all, since parse_policy_metadata parses
    doc_id entirely from the PDF's own header text, with zero
    connection to what the file is named on disk. When the two didn't
    match, deletion's Qdrant filter matched ZERO points (a filter that
    matches nothing "succeeds" silently, deleting nothing while
    reporting normally), and worse - reindex_state.json's entry for the
    file's REAL doc_id was untouched by the (wrongly-keyed) deletion, so
    re-uploading under the same filename went through the FULL, correct
    ingestion path with no old hashes to compare against, embedding
    everything as brand new ON TOP of the still-fully-intact old
    vectors that were never actually deleted - vector count only ever
    went up, exactly as reported.

    Fast path: reindex_state.json's per-doc_id 'source_filename' field
    (added specifically to fix this) gives an exact, instant lookup for
    anything ingested after this fix shipped - no need to re-open or
    re-parse the PDF at all.

    Fallback, for entries from before this field existed: re-parse the
    PDF's own header directly, the same way ingestion itself determines
    doc_id - slower, but the only genuinely correct source of truth
    when the fast path has nothing to offer.
    """
    if os.path.exists(REINDEX_STATE_PATH):
        with open(REINDEX_STATE_PATH) as f:
            state = json.load(f)
        for candidate_doc_id, entry in state.items():
            if isinstance(entry, dict) and entry.get("source_filename") == filename:
                return candidate_doc_id

    from app.rag.extraction import extract_pdf
    from app.rag.metadata import parse_policy_metadata
    elements = extract_pdf(full_path, "data/policy_images")
    header_source = next((e.content for e in elements if e.element_type == "text"), "")
    policy_meta = parse_policy_metadata(header_source)
    return policy_meta.doc_id


@router.delete("/{filename}")
def delete_policy(filename: str, confirm: bool = False, db: Session = Depends(get_db),
                    _auth=Depends(require_admin)):
    """Deletes one policy document and everything derived from it, not
    just the file on disk - the gap being closed here directly: the
    only deletion previously available was the admin page's "wipe the
    entire RAG index" button, an all-or-nothing tool wrong for the
    routine case of retiring a single superseded document.

    Cleans up, in order:
      1. Every chunk/embedding for this doc_id in Qdrant (not the whole
         collection - every OTHER document's embeddings are untouched).
      2. Its reindex_state.json entry, so a future ingestion run
         doesn't think this doc_id still exists and skip it as
         "unchanged" if the same filename is ever re-uploaded.
      3. Any extracted chart/figure images for this document.
      4. The physical PDF file itself.
      5. The retrieval cache (policy content is now genuinely
         different - a stale cached result citing this doc_id must not
         keep being served).
      6. A durable AlertRecord documenting the deletion - not tied to
         any single case (AuditLogEntry requires one), but still a
         real, queryable record of who deleted what and when.

    Deliberately does NOT touch past ExceptionCase decisions that cite
    this doc_id - a historical decision's audit trail records what
    policy justified it AT THE TIME, and must stay intact regardless of
    what happens to the document afterward; rewriting or blocking on
    history would be the wrong instinct for an audit trail. Instead,
    the response reports how many past cases cite this doc_id, so the
    person deleting it can see that real, historical impact even though
    deletion proceeds either way - removing a document from ACTIVE
    retrieval and preserving what it was CITED FOR historically are two
    different concerns, handled differently on purpose.
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="Pass ?confirm=true to actually delete this policy document")

    dest_path = os.path.join(POLICY_UPLOAD_DIR, filename)
    if not os.path.exists(dest_path):
        raise HTTPException(status_code=404, detail=f"No such file on disk: {filename}")

    doc_id = _resolve_doc_id_for_filename(filename, dest_path)

    # Checks whether THIS filename is actually the CURRENT, active
    # source for its doc_id before touching anything in Qdrant - found
    # and fixed a real, serious bug directly from a user report. A
    # SUPERSEDED file (one already showing "not indexed" because a
    # newer file took over its doc_id - see ingest_policy_pdf's
    # source_filename design) still resolves to the SAME doc_id via
    # _resolve_doc_id_for_filename's fallback (it re-parses the PDF's
    # own header, which still genuinely says the same Document ID it
    # always did). Deleting unconditionally by that resolved doc_id
    # deleted every chunk for it - including the ones actively owned by
    # the CURRENT file that superseded this one. A superseded file
    # contributes nothing to Qdrant anymore (its content was already
    # replaced or stale-cleaned when the newer file was ingested), so
    # deleting it must only ever remove the physical file itself -
    # never touch Qdrant, images, cache, or reindex_state, which all
    # correctly belong to whatever file is CURRENTLY active for this
    # doc_id, not this one.
    current_active_filename = None
    if os.path.exists(REINDEX_STATE_PATH):
        with open(REINDEX_STATE_PATH) as f:
            _state_check = json.load(f)
        current_active_filename = (_state_check.get(doc_id) or {}).get("source_filename")

    if current_active_filename is not None and current_active_filename != filename:
        os.remove(dest_path)
        from app.core.alerting import send_alert
        send_alert(db, event_type="policy_document_deleted", detail={
            "doc_id": doc_id, "filename": filename, "deleted_by": _auth.name,
            "deleted_point_count": 0,
            "note": f"superseded file removed; doc_id's active content remains '{current_active_filename}'",
        })
        return {
            "filename": filename,
            "doc_id": doc_id,
            "deleted_vector_chunk_count": 0,
            "removed_from_reindex_state": False,
            "deleted_images": [],
            "retrieval_cache_cleared": False,
            "cited_in_case_count": 0,
            "note": (
                f"'{filename}' was already superseded by '{current_active_filename}' for doc_id "
                f"'{doc_id}' - only this stale file itself was removed; the doc_id's real, active "
                f"content (owned by '{current_active_filename}') was intentionally left untouched."
            ),
        }

    # NOT bumped here - see below, right after the actual Qdrant
    # delete completes, for why "before" was the wrong place for this.

    # Best-effort citation count across past decisions - a Python-side
    # scan of a JSON column, not an indexed query. Fine at this
    # project's scale; a genuinely high-volume production deployment
    # would want cited_policy.doc_id promoted to its own indexed column
    # rather than living only inside the JSON blob, noted here as a
    # real scaling limitation rather than silently ignored.
    cited_in_case_ids = []
    for case in db.query(ExceptionCase).filter(ExceptionCase.resolution_decision.isnot(None)).all():
        cited = (case.resolution_decision or {}).get("cited_policy") or {}
        if cited.get("doc_id") == doc_id:
            cited_in_case_ids.append(case.id)

    from app.core.config import get_settings
    from app.rag.vectorstore import get_qdrant_client, delete_nodes_by_doc_id
    settings = get_settings()
    deleted_point_count = delete_nodes_by_doc_id(get_qdrant_client(), settings.qdrant_collection, doc_id)

    # Bumped HERE, right after Qdrant's own delete genuinely completes -
    # not before it, which is what an earlier version of this did.
    # Found and fixed a real, subtler gap directly from a user's own
    # careful questioning: bumping BEFORE the actual delete only
    # protects a retrieval that was ALREADY mid-flight when this
    # request started. It does NOT protect a retrieval that starts
    # AFTER the bump but BEFORE Qdrant's delete has actually finished -
    # that retrieval's own "before" check would already see the bumped
    # number, its "after" check would see the SAME number (nothing else
    # changed during ITS read), so it would conclude "no interference"
    # and cache genuinely stale, not-yet-deleted data - the exact
    # failure this whole mechanism exists to prevent, just moved to a
    # different, narrower timing window. Bumping only after the real
    # delete is confirmed done closes this correctly: any retrieval
    # whose own "after" check lands anywhere at or after this line
    # (which is every retrieval genuinely overlapping the delete
    # operation) is guaranteed to observe the change; anything that
    # finished entirely before this line read real, consistent,
    # pre-deletion data and was always safe to cache.
    from app.rag.mutation_lock import bump_rag_generation
    bump_rag_generation()

    # Uses the SAME shared, cross-process lock ingestion.py uses for
    # its own read-merge-write of this file (see
    # app/rag/reindex_state_lock.py) - not a separate or no lock at
    # all. Without this, a deletion and a concurrent ingestion of a
    # DIFFERENT document could both read the file before either
    # writes, and whichever writes last would silently overwrite the
    # other's change - either resurrecting a just-deleted doc_id's
    # entry, or losing a different document's just-added entry
    # entirely, even though its real Qdrant chunks would still exist.
    from app.rag.reindex_state_lock import reindex_state_lock
    removed_from_reindex_state = False
    with reindex_state_lock():
        if os.path.exists(REINDEX_STATE_PATH):
            with open(REINDEX_STATE_PATH) as f:
                state = json.load(f)
            if doc_id in state:
                del state[doc_id]
                with open(REINDEX_STATE_PATH, "w") as f:
                    json.dump(state, f, indent=2)
                removed_from_reindex_state = True

    deleted_images = []
    if os.path.isdir(POLICY_IMAGE_DIR):
        for img_path in glob.glob(os.path.join(POLICY_IMAGE_DIR, f"{doc_id}_*")):
            os.remove(img_path)
            deleted_images.append(os.path.basename(img_path))

    os.remove(dest_path)

    from app.cache.retrieval_cache import invalidate_retrieval_cache_for_doc_type
    invalidate_retrieval_cache_for_doc_type(doc_id)

    from app.core.alerting import send_alert
    send_alert(db, event_type="policy_document_deleted", detail={
        "doc_id": doc_id, "filename": filename,
        "deleted_by": _auth.name,
        "deleted_point_count": deleted_point_count,
        "cited_in_case_count": len(cited_in_case_ids),
    })

    return {
        "filename": filename,
        "doc_id": doc_id,
        "deleted_vector_chunk_count": deleted_point_count,
        "removed_from_reindex_state": removed_from_reindex_state,
        "deleted_images": deleted_images,
        "retrieval_cache_cleared": True,
        "cited_in_case_count": len(cited_in_case_ids),
        "note": (
            f"This document is cited in {len(cited_in_case_ids)} past resolution decision(s). "
            "Those historical audit records are untouched and remain fully intact - only this "
            "document's own file and active-retrieval index entries were removed."
            if cited_in_case_ids else
            "This document was not cited in any past resolution decisions."
        ),
    }
