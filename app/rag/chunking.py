"""
Chunking per architecture doc 8.2.2: hierarchical parent-child for text
(parent = whole-page section, child = small sentence-window chunk, for
precision-at-search + full-context-at-generation), tables and images/charts
kept as atomic nodes -- never split, since a table row containing "180
days" split mid-table is exactly how a return-window number gets silently
corrupted.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from app.rag.extraction import ExtractedElement
from app.rag.metadata import PolicyMetadata


@dataclass
class RagNode:
    node_id: str
    text: str                      # the text actually embedded/searched
    element_type: str              # "text_child" | "text_parent" | "table" | "chart"
    parent_id: str | None
    page: int
    content_hash: str
    metadata: dict = field(default_factory=dict)  # flattened PolicyMetadata + doc-level fields


def _policy_metadata_to_dict(meta: PolicyMetadata) -> dict:
    return {
        "doc_id": meta.doc_id,
        "version": meta.version,
        "effective_start": meta.effective_start.isoformat(),
        "effective_end": meta.effective_end.isoformat() if meta.effective_end else None,
        "superseded_by": meta.superseded_by,
        "supersedes": meta.supersedes,
        "doc_type": meta.doc_type,
        "product_category": meta.product_category,
        "channel": meta.channel,
    }


def _split_into_child_chunks(text: str, max_sentences: int = 2) -> list[str]:
    """Splits page text into small child chunks for precise retrieval.

    Real unstructured.io hi_res output gives proper per-element
    segmentation (Title, NarrativeText, ListItem as separate elements), so
    paragraph boundaries are already known there. pdfplumber's
    extract_text() (this sandbox's substitute) returns one flat text blob
    per page with only line-wrap newlines and no paragraph markers -- so
    splitting on "\n\n" (the naive first attempt) produced exactly one
    giant "paragraph" per page, defeating the point of child-level
    chunking entirely (confirmed while testing Phase 3 against the real
    generated PDFs).

    This uses a sentence-window split instead: group every N sentences
    into one child chunk. It's a coarser signal than real layout-aware
    segmentation, but it's honest, functional, and produces genuinely
    separate, retrievable child nodes.
    """
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.replace("\n", " ").strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunks.append(" ".join(sentences[i:i + max_sentences]))
    return chunks or [text.strip()]


def build_nodes(elements: list[ExtractedElement], policy_meta: PolicyMetadata) -> list[RagNode]:
    """One parent node per page (full-page text, for generation-time
    context), several child nodes per page via sentence-window splitting
    (for precise retrieval), plus one atomic node per table and per chart
    element."""
    nodes: list[RagNode] = []
    meta_dict = _policy_metadata_to_dict(policy_meta)

    pages_seen: dict[int, str] = {}

    for el in elements:
        if el.element_type == "text":
            if el.page not in pages_seen:
                parent_id = str(uuid.uuid4())
                pages_seen[el.page] = parent_id
                nodes.append(RagNode(
                    node_id=parent_id,
                    text=el.content,
                    element_type="text_parent",
                    parent_id=None,
                    page=el.page,
                    content_hash=el.content_hash,
                    metadata=meta_dict,
                ))
            parent_id = pages_seen[el.page]
            for chunk in _split_into_child_chunks(el.content):
                nodes.append(RagNode(
                    node_id=str(uuid.uuid4()),
                    text=chunk,
                    element_type="text_child",
                    parent_id=parent_id,
                    page=el.page,
                    content_hash=el.content_hash,  # inherits page-level hash for reindex diffing
                    metadata=meta_dict,
                ))

        elif el.element_type == "table":
            nodes.append(RagNode(
                node_id=str(uuid.uuid4()),
                text=el.content,   # flattened table text, atomic -- never split
                element_type="table",
                parent_id=None,
                page=el.page,
                content_hash=el.content_hash,
                metadata=meta_dict,
            ))

        elif el.element_type in ("image", "chart"):
            nodes.append(RagNode(
                node_id=str(uuid.uuid4()),
                text=el.content,   # caption text (from captioning.py) is what's searched
                element_type=el.element_type,
                parent_id=None,
                page=el.page,
                content_hash=el.content_hash,
                metadata={**meta_dict, "image_path": el.image_path},
            ))

    return nodes
