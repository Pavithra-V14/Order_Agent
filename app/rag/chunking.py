"""
Chunking per architecture doc 8.2.2: hierarchical parent-child for text
(parent = whole-page section, child = small sentence-window chunk, for
precision-at-search + full-context-at-generation), tables and images/charts
kept as atomic nodes -- never split, since a table row containing "180
days" split mid-table is exactly how a return-window number gets silently
corrupted.
"""
from __future__ import annotations

import hashlib
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
        "return_window_days_by_category": meta.return_window_days_by_category,
    }


def _split_into_child_chunks(text: str, target_avg_sentences: int = 2, max_sentences: int = 4) -> list[str]:
    """Splits page text into small child chunks for precise retrieval,
    using CONTENT-DEFINED chunking - boundaries are placed based on
    each sentence's own hash, not by counting position from the start
    of the page.

    Real unstructured.io hi_res output gives proper per-element
    segmentation (Title, NarrativeText, ListItem as separate elements), so
    paragraph boundaries are already known there. pdfplumber's
    extract_text() (this sandbox's substitute) returns one flat text blob
    per page with only line-wrap newlines and no paragraph markers -- so
    splitting on "\n\n" (the naive first attempt) produced exactly one
    giant "paragraph" per page, defeating the point of child-level
    chunking entirely (confirmed while testing Phase 3 against the real
    generated PDFs).

    An earlier version of this grouped a FIXED count of sentences per
    chunk ("every 2 sentences, sequentially from the top of the page").
    That's position-dependent: inserting even one sentence near the top
    of a page shifts every downstream grouping boundary, since chunk N
    is defined as "the Nth pair of sentences counting from the start" -
    confirmed directly with a real simulation where a single inserted
    sentence caused every subsequent chunk on the page to look "new,"
    even though most of the underlying sentences never changed a word.

    This version instead decides each boundary from the CONTENT of the
    sentence sitting at that boundary - a real, if simplified, form of
    content-defined chunking (the same family of technique used by
    rsync/restic/Borg for exactly this stability property). Whether
    sentence S ends a chunk depends only on hash(S) itself, never on
    how many sentences came before it - so inserting a new sentence
    elsewhere on the page cannot change that decision for sentences
    that were never touched. Confirmed directly: the same insertion
    that broke every chunk under the old scheme now only affects the
    one chunk actually containing the edit (plus, occasionally, its
    immediate neighbor, if the edit lands right at an existing
    boundary) - everything else on the page hashes identically to
    before and is correctly skipped.

    max_sentences remains as a hard ceiling (not the primary mechanism
    anymore) purely to bound worst-case chunk size, since a
    content-defined boundary could theoretically not occur for an
    unusually long run of sentences.
    """
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.replace("\n", " ").strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [text.strip()]

    chunks = []
    current: list[str] = []
    for sentence in sentences:
        current.append(sentence)
        sentence_hash = int(hashlib.sha256(sentence.encode("utf-8")).hexdigest(), 16)
        # A boundary here depends ONLY on this sentence's own content
        # and how many sentences have accumulated since the last
        # boundary - never on this sentence's absolute position in the
        # page, which is the property that makes this stable across
        # insertions/deletions elsewhere on the page.
        at_content_boundary = (sentence_hash % target_avg_sentences) == 0
        at_hard_ceiling = len(current) >= max_sentences
        if at_content_boundary or at_hard_ceiling:
            chunks.append(" ".join(current))
            current = []
    if current:
        chunks.append(" ".join(current))
    return chunks


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
                    # Own, independently-computed hash - NOT inherited
                    # from the parent page anymore. This is what
                    # actually unlocks the benefit of content-defined
                    # chunking above: with the OLD position-based
                    # splitter, individual chunk boundaries weren't
                    # stable across edits, so every child had to share
                    # the whole page's hash and be treated as one
                    # coarse unit for reindex diffing (an edit ANYWHERE
                    # on the page correctly forced re-embedding
                    # EVERYWHERE on the page, since there was no
                    # reliable way to tell which specific chunk actually
                    # changed). Content-defined boundaries ARE stable
                    # for untouched regions, so each child can now be
                    # diffed independently - re-ingestion re-embeds only
                    # the handful of chunks whose own text genuinely
                    # changed, not the entire page every time.
                    content_hash=hashlib.sha256(chunk.encode("utf-8")).hexdigest(),
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
