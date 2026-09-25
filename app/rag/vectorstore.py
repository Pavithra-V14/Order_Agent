"""
Vector store wrapper — Qdrant embedded local mode per architecture doc
8.2.6 (self-hosted, free, real-time payload filtering, no server needed
for dev). Swap to a real Qdrant server in staging/prod by passing
`url=settings.qdrant_url` instead of `path=` — same client API either way.

Client is a process-wide singleton (see `get_qdrant_client`): Qdrant's
embedded local mode takes an exclusive file lock on its storage path, so
opening a second `QdrantClient(path=...)` against the same path from the
same process raises "already accessed by another instance" — discovered
while wiring ingestion + retrieval together in Phase 3 testing. A real
Qdrant server (staging/prod) doesn't have this constraint, but the
singleton pattern is the right one to keep regardless — one client per
process, not one per call, is standard practice either way.
"""
from __future__ import annotations

from datetime import date

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, MatchAny, Range, IsNullCondition, PayloadField, MinShould,
)

from app.core.config import get_settings
from app.rag.chunking import RagNode

_client_singleton: QdrantClient | None = None
_client_singleton_key: str | None = None


def get_qdrant_client() -> QdrantClient:
    """Returns the process-wide Qdrant client. Three modes, chosen
    automatically from settings:
      1. qdrant_url + qdrant_api_key set -> Qdrant Cloud (cloud.qdrant.io),
         no Docker needed, free tier available.
      2. qdrant_url set alone -> any self-hosted/reachable Qdrant server
         (e.g. a VM running Qdrant directly, still no Docker required).
      3. neither set -> embedded local mode (this sandbox's default),
         writes to qdrant_local_path, no server of any kind needed.
    """
    global _client_singleton, _client_singleton_key
    settings = get_settings()
    key = settings.qdrant_url or settings.qdrant_local_path
    if _client_singleton is not None and _client_singleton_key == key:
        return _client_singleton

    if _client_singleton is not None:
        _client_singleton.close()

    if settings.qdrant_url:
        _client_singleton = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    else:
        _client_singleton = QdrantClient(path=settings.qdrant_local_path)
    _client_singleton_key = key
    return _client_singleton


def _date_to_int(d: str | date) -> int:
    """Qdrant's Range filter is numeric-only (confirmed via a
    pydantic_core ValidationError when passing an ISO date string
    directly) — dates are stored/filtered as YYYYMMDD integers, e.g.
    2025-06-15 -> 20250615, which sorts and compares correctly as a plain
    number without needing a separate date-parsing step at query time."""
    if isinstance(d, str):
        d = date.fromisoformat(d)
    return d.year * 10000 + d.month * 100 + d.day


def ensure_collection(client: QdrantClient, collection: str, dim: int) -> None:
    """Creates the collection AND the payload indexes every filtered
    field needs. This is the real bug found running against Qdrant
    Cloud: a managed/remote Qdrant instance REQUIRES an explicit payload
    index before you can filter on a field at all — "Index required but
    not found for 'effective_start_num'" — whereas Qdrant's embedded
    local mode (this project's original dev/test default) filters on any
    payload field without one. This never surfaced during local-only
    development for exactly that reason; it's a genuine production-vs-
    dev-mode gap in Qdrant itself, not something either mode "does wrong."
    create_payload_index is idempotent (safe to call on an existing
    collection/index — Qdrant no-ops if it's already there), so this runs
    unconditionally rather than only on first creation.
    """
    from qdrant_client.models import PayloadSchemaType

    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    for field_name, schema_type in [
        ("effective_start_num", PayloadSchemaType.INTEGER),
        ("effective_end_num", PayloadSchemaType.INTEGER),
        ("doc_type", PayloadSchemaType.KEYWORD),
        ("channel", PayloadSchemaType.KEYWORD),
        ("product_category", PayloadSchemaType.KEYWORD),
        # Added to fix a real, confirmed production bug: delete_nodes_by_doc_id
        # (and the count-before-delete check in app/api/v1/policies.py's
        # delete_policy) both filter on this field, and Qdrant's SERVER/
        # CLOUD mode requires an explicit index to filter on any field at
        # all - it returned a real, clear 400 ("Index required but not
        # found for 'doc_id'") the moment a real user tried it against
        # real Qdrant Cloud. The LOCAL embedded mode this project's own
        # test suite runs against does NOT enforce this requirement at
        # all, which is exactly why every test passed while this was
        # completely broken against real cloud infrastructure - a real,
        # confirmed gap in test coverage, not a flaw in the delete logic
        # itself. create_payload_index is idempotent and this function
        # already runs unconditionally on every call (see this
        # function's own docstring above), so this retroactively fixes
        # any EXISTING cloud collection the next time ensure_collection
        # runs (app startup, or the next ingestion call) - no manual
        # migration step needed.
        ("doc_id", PayloadSchemaType.KEYWORD),
    ]:
        client.create_payload_index(
            collection_name=collection, field_name=field_name, field_schema=schema_type,
        )


def refresh_stale_metadata(client: QdrantClient, collection: str, node_id_to_expected_payload: dict[str, dict]) -> dict[str, list[str]]:
    """Generalizes refresh_node_metadata and refresh_stale_page_numbers
    (both retired in favor of this) into one mechanism that catches ANY
    metadata field drifting on a chunk whose TEXT is unchanged - not
    just document-level fields, not just page. Detects the need to
    update WITHOUT knowing in advance which field changed: for each
    node, it's handed the full payload that a fresh embed would write
    right now (computed the same way upsert_nodes computes it, just
    without needing a new vector for text that hasn't changed), reads
    what's ACTUALLY stored, and does a genuine field-by-field diff.
    Whatever differs gets patched via set_payload; whatever already
    matches is left alone, so this never writes more than the real
    drift requires.

    Args:
        node_id_to_expected_payload: {old_node_id: {field: value, ...}}
            — the full, freshly-computed metadata each old (still-valid,
            unchanged-text) point SHOULD currently have. Multiple old
            node_ids commonly share an identical expected dict (sibling
            chunks from the same page share both page and document-
            level metadata) — that's fine; each is still compared and
            patched independently, since what's ACTUALLY stored for
            each specific point could differ even when what SHOULD be
            stored doesn't (e.g. a prior partial write only reached
            some of them).

    Returns:
        {node_id: [field_names_that_were_actually_corrected]} — only
        for nodes where at least one field genuinely differed. A node
        with no drift at all is simply absent from this dict, not
        present with an empty list — the cheapest possible signal for
        "nothing needed doing here."
    """
    corrected: dict[str, list[str]] = {}
    node_ids = list(node_id_to_expected_payload.keys())
    if not node_ids:
        return corrected
    try:
        stored_points = client.retrieve(collection_name=collection, ids=node_ids, with_payload=True)
    except Exception:
        # Nothing stored yet to compare against (e.g. first-time path) -
        # nothing to correct.
        return corrected
    stored_by_id = {p.id: (p.payload or {}) for p in stored_points}
    for node_id, expected in node_id_to_expected_payload.items():
        stored = stored_by_id.get(node_id)
        if stored is None:
            continue
        diff = {field: value for field, value in expected.items() if stored.get(field) != value}
        if diff:
            client.set_payload(collection_name=collection, payload=diff, points=[node_id])
            corrected[node_id] = list(diff.keys())
    return corrected


def upsert_nodes(client: QdrantClient, collection: str, nodes: list[RagNode], vectors: np.ndarray) -> None:
    points = []
    for node, vec in zip(nodes, vectors):
        payload = {
            "text": node.text,
            "element_type": node.element_type,
            "parent_id": node.parent_id,
            "page": node.page,
            "content_hash": node.content_hash,
            **node.metadata,
        }
        # Numeric date fields for Qdrant's Range filter (see _date_to_int
        # docstring) — kept alongside the human-readable ISO strings
        # already in node.metadata, not instead of them, so citations can
        # still display "2025-01-01" rather than "20250101".
        #
        # effective_end_num is written explicitly as None (not omitted)
        # when there's no end date — Qdrant's IsNullCondition, per testing,
        # needs the key to genuinely be present with a null value to match;
        # a fully-absent key did not satisfy it, which silently zeroed out
        # every retrieval for a currently-active (no-end-date) policy until
        # this was root-caused against the real Phase 3 acceptance tests.
        payload["effective_start_num"] = _date_to_int(payload["effective_start"]) if payload.get("effective_start") else None
        payload["effective_end_num"] = _date_to_int(payload["effective_end"]) if payload.get("effective_end") else None
        points.append(PointStruct(id=node.node_id, vector=vec.tolist(), payload=payload))
    if points:
        client.upsert(collection_name=collection, points=points)


def delete_points_by_ids(client: QdrantClient, collection: str, point_ids: list[str]) -> int:
    """Deletes specific points by their exact IDs - the counterpart to
    delete_nodes_by_doc_id above, needed for STALE-CHUNK cleanup on
    re-ingestion: when content is genuinely removed from a document
    (a sentence deleted, or - per a real user scenario - one filename
    superseding another under the same doc_id with different content),
    the OLD chunks that no longer correspond to anything in the
    current document must be individually removed, not the whole
    doc_id's worth of points at once (which would also wipe out chunks
    that are still correct and unchanged).

    Returns the actual number of points that existed and were removed
    (Qdrant's own delete() call doesn't report a count), via the same
    count-then-delete pattern as delete_nodes_by_doc_id."""
    if not point_ids:
        return 0
    try:
        existing = client.retrieve(collection_name=collection, ids=point_ids, with_payload=False)
    except Exception:
        return 0
    if not existing:
        return 0
    client.delete(collection_name=collection, points_selector=[p.id for p in existing])
    return len(existing)


def delete_nodes_by_doc_id(client: QdrantClient, collection: str, doc_id: str) -> int:
    """Deletes every chunk/embedding belonging to one policy document,
    identified by its doc_id payload field - the counterpart to
    upsert_nodes above, needed for real per-document deletion (as
    opposed to delete_collection, which wipes every document at once).

    Returns the count of points that existed before deletion (Qdrant's
    own delete() call doesn't report how many it removed, so this does
    a count-then-delete rather than trusting an assumed return value)."""
    try:
        existing = client.count(
            collection_name=collection,
            count_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
        ).count
    except Exception:
        # Collection may not exist yet if nothing has ever been ingested -
        # nothing to delete either way.
        return 0
    if existing == 0:
        return 0
    client.delete(
        collection_name=collection,
        points_selector=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]),
    )
    return existing


def build_temporal_filter(as_of_date: str, channel: str | None = None,
                           product_category: str | None = None,
                           doc_type: str | None = None) -> Filter:
    """The core mechanism from architecture doc 8.2.4: filter to documents
    whose effective range covers `as_of_date` (the ORDER's purchase date,
    never "today"), applied BEFORE similarity search — not as a post-hoc
    re-rank. This is what makes the return-policy-changed edge case resolve
    correctly by construction rather than by hoping the LLM notices."""
    as_of_num = _date_to_int(as_of_date)
    must: list = [
        FieldCondition(key="effective_start_num", range=Range(lte=as_of_num)),
    ]
    should = [
        IsNullCondition(is_null=PayloadField(key="effective_end_num")),
        FieldCondition(key="effective_end_num", range=Range(gte=as_of_num)),
    ]
    if doc_type:
        must.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
    if channel:
        must.append(FieldCondition(key="channel", match=MatchAny(any=[channel, "all"])))
    if product_category:
        must.append(FieldCondition(key="product_category", match=MatchAny(any=[product_category, "all"])))

    return Filter(must=must, min_should=MinShould(conditions=should, min_count=1))


def search(client: QdrantClient, collection: str, query_vector: np.ndarray,
           qfilter: Filter | None, limit: int = 10):
    return client.query_points(
        collection_name=collection,
        query=query_vector.tolist(),
        query_filter=qfilter,
        limit=limit,
        with_payload=True,
    ).points
