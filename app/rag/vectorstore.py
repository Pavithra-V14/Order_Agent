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
    global _client_singleton, _client_singleton_key
    settings = get_settings()
    key = settings.qdrant_url or settings.qdrant_local_path
    if _client_singleton is not None and _client_singleton_key == key:
        return _client_singleton

    if _client_singleton is not None:
        _client_singleton.close()

    if settings.qdrant_url:
        _client_singleton = QdrantClient(url=settings.qdrant_url)
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
    existing = [c.name for c in client.get_collections().collections]
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )


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
