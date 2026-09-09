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
from app.rag.vectorstore import get_qdrant_client, ensure_collection, upsert_nodes, refresh_node_metadata, _date_to_int
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
    skipped_old_node_ids: set[str] = set()
    skipped = 0

    for node in nodes:
        new_hashes[node.node_id] = node.content_hash
        if node.content_hash in prev_hashes.values():
            skipped += 1
            skipped_old_node_ids.update(old_hash_to_node_ids.get(node.content_hash, []))
            continue
        nodes_to_embed.append(node)

    embedder = get_embedder()
    embedder.fit([n.text for n in nodes])

    client = get_qdrant_client()
    ensure_collection(client, collection, embedder.dim)

    if nodes_to_embed:
        vectors = embedder.embed([n.text for n in nodes_to_embed])
        upsert_nodes(client, collection, nodes_to_embed, vectors)

    if skipped_old_node_ids:
        # THE fix for a real bug found from an actual production run:
        # skipping re-embedding (expensive) previously ALSO meant
        # skipping any metadata refresh at all (cheap, and unrelated) —
        # so a chunk whose TEXT never changed kept whatever
        # document-level metadata existed at the moment it was FIRST
        # embedded, forever, even after a real metadata schema change
        # (like adding return_window_days_by_category) and a full
        # re-ingestion run. See refresh_node_metadata()'s own docstring
        # for the full story of how this was found.
        current_metadata = _policy_metadata_to_dict(policy_meta)
        current_metadata["effective_start_num"] = (
            _date_to_int(current_metadata["effective_start"]) if current_metadata.get("effective_start") else None
        )
        current_metadata["effective_end_num"] = (
            _date_to_int(current_metadata["effective_end"]) if current_metadata.get("effective_end") else None
        )
        refresh_node_metadata(client, collection, list(skipped_old_node_ids), current_metadata)

    reindex_state[policy_meta.doc_id] = {
        "hashes": new_hashes,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
        "source_mtime": datetime.fromtimestamp(os.path.getmtime(pdf_path), tz=timezone.utc).isoformat(),
    }
    _save_reindex_state(reindex_state)

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
