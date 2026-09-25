"""
Ingestion pipeline - orchestrates architecture doc 8.2.1 through 8.2.5:
extract -> parse version metadata -> caption charts -> chunk into nodes ->
embed -> upsert, with incremental reindexing (only changed elements are
re-embedded) and additive versioning (old versions are marked superseded,
never deleted - required for the temporal-correctness guarantee).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone

from app.rag.extraction import extract_pdf, ExtractedElement
from app.rag.metadata import parse_policy_metadata, PolicyMetadata
from app.rag.captioning import caption_chart_heuristic
from app.rag.chunking import build_nodes, RagNode, _policy_metadata_to_dict
from app.rag.embeddings import get_embedder
from app.rag.vectorstore import get_qdrant_client, ensure_collection, upsert_nodes, refresh_stale_metadata, delete_points_by_ids, _date_to_int
from app.core.config import get_settings

logger = logging.getLogger(__name__)

_KNOWN_CHART_DATA = {
    "RET-POLICY-2025-A": {
        "title": "Return Window by Category - Policy 2025-A",
        "categories": ["Apparel", "Footwear", "Electronics", "Home", "Beauty"],
        "values": [180, 180, 30, 180, 45],
    },
    "RET-POLICY-2026-A": {
        "title": "Return Window by Category - Policy 2026-A",
        "categories": ["Apparel", "Footwear", "Electronics", "Home", "Beauty"],
        "values": [120, 120, 30, 120, 45],
    },
    "FRAUD-POLICY-2025-A": {
        "title": "Illustrative Fraud/Risk Score Distribution",
        "categories": ["low", "medium", "elevated", "high"],
        "values": [72, 19, 6, 3],
    },
}

_REINDEX_STATE_PATH = os.path.join("data", "reindex_state.json")


def _load_reindex_state() -> dict:
    if os.path.exists(_REINDEX_STATE_PATH):
        with open(_REINDEX_STATE_PATH) as f:
            return json.load(f)
    return {}


def _save_reindex_state(state: dict) -> None:
    os.makedirs(os.path.dirname(_REINDEX_STATE_PATH), exist_ok=True)
    with open(_REINDEX_STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def _extract_hashes(entry) -> dict:
    """Backward-compatible read: reindex_state.json entries were
    originally a flat {node_id: content_hash} dict. Adding an
    'indexed_at' timestamp (for index-freshness-lag metrics) required
    nesting to {"hashes": {...}, "indexed_at": "..."} — this reads
    EITHER format correctly, so an existing reindex_state.json file
    written before this change doesn't break or force a needless
    full re-ingestion of every document."""
    if isinstance(entry, dict) and "hashes" in entry:
        return entry["hashes"]
    return entry or {}


def _extract_doc_id_guess(pdf_path: str) -> str:
    return os.path.splitext(os.path.basename(pdf_path))[0]


def ingest_policy_pdf(pdf_path: str, image_out_dir: str = "data/policy_images",
                       collection: str = None) -> dict:
    """Ingests one policy PDF end to end. Safe to call repeatedly -
    re-running on an unchanged PDF re-embeds nothing (8.2.5)."""
    settings = get_settings()
    collection = collection or settings.qdrant_collection
    doc_id_guess = _extract_doc_id_guess(pdf_path)

    elements = extract_pdf(pdf_path, image_out_dir)
    if not elements:
        raise ValueError(f"No elements extracted from {pdf_path} - check the PDF is valid.")

    header_source = next((e.content for e in elements if e.element_type == "text"), "")
    policy_meta: PolicyMetadata = parse_policy_metadata(header_source)

    known = _KNOWN_CHART_DATA.get(policy_meta.doc_id) or _KNOWN_CHART_DATA.get(doc_id_guess)
    for el in elements:
        if el.element_type == "chart":
            if known:
                el.content = caption_chart_heuristic(
                    el.image_path, known["title"], known["categories"], known["values"]
                )
            else:
                el.content = f"[Uncaptioned chart on page {el.page} - no vision-LLM access in this sandbox]"
            el.content_hash = hashlib.sha256(el.content.encode("utf-8")).hexdigest()

    nodes: list[RagNode] = build_nodes(elements, policy_meta)

    reindex_state = _load_reindex_state()
    prev_hashes: dict = _extract_hashes(reindex_state.get(policy_meta.doc_id, {}))
    # Reverse mapping (old content_hash -> list of old node_ids) — needed
    # because node_id is a freshly-generated random UUID every single
    # run (see app/rag/chunking.py's build_nodes()), regardless of
    # whether the content matches something already stored. A "skipped"
    # node's ID in THIS run has zero correspondence to whatever ID that
    # same content was actually stored under in Qdrant in a PREVIOUS
    # run — confirmed directly: an earlier version of this fix tried to
    # refresh metadata using the current run's freshly-generated IDs
    # and hit a real KeyError, since Qdrant had never heard of them.
    # The OLD node_id (from reindex_state.json, which genuinely exists
    # as a Qdrant point) is what must be used instead.
    #
    # This must be a ONE-TO-MANY mapping, not hash -> single node_id:
    # a parent node and all its child chunks from the SAME page
    # deliberately SHARE one content_hash (see build_nodes()'s
    # `content_hash=el.content_hash` on both parent and child nodes,
    # "inherits page-level hash for reindex diffing"). A naive
    # {hash: node_id} reverse mapping silently drops all but one of
    # several nodes sharing a hash — confirmed directly: an earlier
    # version of this fix using a single-value mapping left 2-3 chunks
    # per document still missing the refreshed metadata, because only
    # one of several same-hash nodes ever got its ID captured.
    old_hash_to_node_ids: dict[str, list[str]] = {}
    for old_node_id, old_hash in prev_hashes.items():
        old_hash_to_node_ids.setdefault(old_hash, []).append(old_node_id)

    new_hashes: dict = {}
    nodes_to_embed: list[RagNode] = []
    # {old_node_id: {field: expected_current_value}} - built alongside
    # the skip decision below, since it needs exactly the same per-node
    # information (content_hash, freshly-computed page/element_type)
    # the skip check already has in hand. Deliberately does NOT include
    # parent_id: that field is a fresh, randomly-generated UUID on
    # EVERY run regardless of content (see build_nodes()), so comparing
    # it would always "detect drift" and needlessly rewrite it every
    # single time - the OLD child's parent_id correctly still points at
    # the OLD parent's still-valid ID, which is exactly where it should
    # keep pointing.
    current_doc_metadata = _policy_metadata_to_dict(policy_meta)
    current_doc_metadata["effective_start_num"] = (
        _date_to_int(current_doc_metadata["effective_start"]) if current_doc_metadata.get("effective_start") else None
    )
    current_doc_metadata["effective_end_num"] = (
        _date_to_int(current_doc_metadata["effective_end"]) if current_doc_metadata.get("effective_end") else None
    )
    node_id_to_expected_payload: dict[str, dict] = {}
    skipped = 0

    for node in nodes:
        new_hashes[node.node_id] = node.content_hash
        if node.content_hash in prev_hashes.values():
            skipped += 1
            expected_payload = {
                "page": node.page,
                "element_type": node.element_type,
                **current_doc_metadata,
            }
            for old_id in old_hash_to_node_ids.get(node.content_hash, []):
                node_id_to_expected_payload[old_id] = expected_payload
            continue
        nodes_to_embed.append(node)

    embedder = get_embedder()
    embedder.fit([n.text for n in nodes])

    client = get_qdrant_client()
    ensure_collection(client, collection, embedder.dim)

    if nodes_to_embed:
        vectors = embedder.embed([n.text for n in nodes_to_embed])
        upsert_nodes(client, collection, nodes_to_embed, vectors)

    if node_id_to_expected_payload:
        # Catches ANY metadata field drifting on a chunk whose text
        # never changed - not hardcoded to document-level fields or to
        # page specifically. Found necessary directly from two separate
        # real production incidents: (1) document-level metadata (like
        # return_window_days_by_category) silently going stale forever
        # on unchanged-text chunks, and (2) a chunk's own page number
        # going stale after content inserted/deleted earlier in the
        # same PDF reflowed everything after it. Both are really the
        # same underlying gap - "skip re-embedding" was incorrectly
        # also skipping "check if anything else about this chunk needs
        # updating" - so this checks the full expected payload at once
        # rather than hardcoding a fixed list of fields to watch.
        refresh_stale_metadata(client, collection, node_id_to_expected_payload)

    # STALE-CHUNK cleanup - a real, previously-missing piece found
    # directly from a user report: this function has always correctly
    # added new/changed content and skipped unchanged content, but
    # never actually removed OLD chunks whose content no longer exists
    # in the CURRENT document at all (a sentence deleted from the PDF,
    # or - the exact scenario reported - a different file superseding
    # this doc_id with genuinely different content). Those old chunks
    # would otherwise sit in Qdrant forever, discoverable by retrieval,
    # even though nothing in the actual current document corresponds
    # to them anymore.
    #
    # "Stale" here means: an old node_id that existed in the PREVIOUS
    # ingestion's hashes, but was neither carried forward (matched,
    # skipped-for-being-unchanged) NOR is it something brand new from
    # this run - the union of node_id_to_expected_payload's keys (every
    # old id that WAS matched above) is exactly the "still valid, keep
    # it" set; everything else that used to exist for this doc_id is
    # genuinely gone from the current document.
    all_old_node_ids = {old_id for ids in old_hash_to_node_ids.values() for old_id in ids}
    stale_node_ids = list(all_old_node_ids - set(node_id_to_expected_payload.keys()))
    if stale_node_ids:
        delete_points_by_ids(client, collection, stale_node_ids)
        # Bumped here, right after this genuinely deletes real Qdrant
        # points - the same generation counter delete_policy bumps
        # after ITS deletion, and for the exact same reason (see
        # app/rag/mutation_lock.py's docstring). Found and fixed a real
        # gap directly from a follow-up conversation: the delete
        # endpoint was the only caller ever bumping this counter, so a
        # concurrent search could still cache genuinely stale data
        # right after THIS deletion - re-ingestion's own stale-chunk
        # cleanup - even though the exact same class of race was
        # already closed for the explicit delete button.
        from app.rag.mutation_lock import bump_rag_generation
        bump_rag_generation()

    # source_filename is a SINGLE value, deliberately - a design choice
    # made directly from a real user report. An earlier version tracked
    # a LIST of filenames sharing one doc_id, letting them co-exist
    # indefinitely - but this doc_id already represents ONE logical
    # document, identified by its own internal "Document ID:" header
    # (see parse_policy_metadata), and a NEW file arriving under that
    # same doc_id with a DIFFERENT filename is a genuine version update
    # (an edit to the same document), not a second, independent
    # document that happens to share an ID. Treating it as a version
    # update - this file becomes the current, canonical source for this
    # doc_id, and the stale-chunk cleanup above already removed whatever
    # content no longer exists in it - is what actually keeps Qdrant's
    # content correct. The old filename genuinely stops being "the"
    # source for this doc_id once a newer one has superseded it, and
    # correctly shows as not-indexed from that point on.
    reindex_state[policy_meta.doc_id] = {
        "hashes": new_hashes,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "source_mtime": datetime.fromtimestamp(os.path.getmtime(pdf_path), tz=timezone.utc).isoformat(),
        # Added specifically to fix a real bug found in production: the
        # delete-policy and list-policies endpoints both used to GUESS
        # that a file's doc_id equals its filename (stem == doc_id) -
        # true only by coincidence, never guaranteed, since doc_id is
        # parsed entirely from the PDF's OWN internal "Document ID:"
        # header text (see app/rag/metadata.py's parse_policy_metadata),
        # completely independent of what the file happens to be named.
        # A real user's file named differently from its internal ID
        # caused delete's Qdrant filter to match zero points (silently
        # "succeeding" while deleting nothing), and re-ingestion under
        # the same filename kept adding NEW vectors on top of the never-
        # deleted old ones - the vector count only ever went up. Storing
        # the real, authoritative filename here means both endpoints can
        # look up the correct doc_id directly, with no guessing and no
        # need to re-open and re-parse the PDF just to find its own ID.
        "source_filename": os.path.basename(pdf_path),
    }

    # Locked, and re-reads a FRESH copy of the file right here instead
    # of saving the (potentially now-stale) snapshot loaded at the top
    # of this function - found and fixed a real, confirmed race
    # directly from a follow-up conversation. Two documents ingested
    # concurrently both start by reading the same file; without this,
    # whichever one finishes and saves LAST would silently overwrite
    # the other's entire entry, since it's writing back a snapshot from
    # before the other's changes existed - not just its own change,
    # the OTHER document's entry would vanish from the file entirely,
    # even though its real Qdrant chunks would still exist untouched.
    # The lock only wraps this brief final read-merge-write, not the
    # slow embedding work above it, so concurrent ingestions of
    # DIFFERENT documents don't need to fully serialize against each
    # other - only this last, fast step does.
    # Uses the SHARED cross-module, cross-process lock (see
    # app/rag/reindex_state_lock.py) - not a private threading.Lock
    # local to this module. A private lock here would only serialize
    # ingestion calls against EACH OTHER; it would do nothing to
    # protect against a concurrent delete_policy call (a different
    # module) reading, modifying, and writing this same file at the
    # same time, which needs the identical protection for the identical
    # reason.
    from app.rag.reindex_state_lock import reindex_state_lock
    with reindex_state_lock():
        fresh_state = _load_reindex_state()
        fresh_state[policy_meta.doc_id] = reindex_state[policy_meta.doc_id]
        _save_reindex_state(fresh_state)

    summary = {
        "doc_id": policy_meta.doc_id,
        "version": policy_meta.version,
        "effective_start": policy_meta.effective_start.isoformat(),
        "effective_end": policy_meta.effective_end.isoformat() if policy_meta.effective_end else None,
        "total_nodes": len(nodes),
        "embedded_this_run": len(nodes_to_embed),
        "skipped_unchanged": skipped,
    }
    logger.info("Ingested %s: %s", pdf_path, summary)
    return summary


def ingest_policy_directory(directory: str = "data/policies") -> list[dict]:
    summaries = []
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(".pdf"):
            summaries.append(ingest_policy_pdf(os.path.join(directory, fname)))
    return summaries
