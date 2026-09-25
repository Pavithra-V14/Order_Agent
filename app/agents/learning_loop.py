"""
Learning loop - architecture doc 8.5. Two distinct mechanisms, kept
deliberately separate:

1. Dynamic few-shot retrieval: past human-reviewed cases are retrieved by
   similarity and used as in-context examples for future Resolution-Policy
   Workflow reasoning - improves consistency without any model fine-tuning.

2. Confidence recalibration: a batch job computes per-cluster accuracy and
   PROPOSES a threshold adjustment. This is the safety-critical half - the
   architecture doc explicitly warns about the risk of a self-reflection
   loop reinforcing its own bad patterns unchecked, so the proposal step
   and the apply step are two SEPARATE functions with two SEPARATE tables,
   and only a human can bridge them (accept_threshold_proposal). The batch
   job itself has no code path that can write to ThresholdOverrideRecord.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from app.core.db import ResolutionPatternEntry, ThresholdProposalRecord, ThresholdOverrideRecord
from app.rag.embeddings import get_embedder


def record_resolution_outcome(
    db: Session,
    case_id: str,
    cluster_key: str,
    case_feature_summary: str,
    agent_proposed_resolution: dict,
    human_final_resolution: dict,
) -> ResolutionPatternEntry:
    """Called on every human escalation decision (approve/edit/reject).
    matched is computed by comparing the proposed vs. final action+amount,
    not a subjective judgment call."""
    matched = (
        agent_proposed_resolution.get("action") == human_final_resolution.get("action")
        and agent_proposed_resolution.get("amount_usd") == human_final_resolution.get("amount_usd")
    )
    entry = ResolutionPatternEntry(
        case_id=case_id,
        cluster_key=cluster_key,
        case_feature_summary=case_feature_summary,
        agent_proposed_resolution=agent_proposed_resolution,
        human_final_resolution=human_final_resolution,
        matched="match" if matched else "mismatch",
    )
    db.add(entry)
    db.commit()
    return entry


@dataclass
class SimilarPastCase:
    case_id: str
    case_feature_summary: str
    human_final_resolution: dict
    similarity: float


# --- Stage 3 memory upgrade: cached similarity index ------------------------
#
# retrieve_similar_past_resolutions() previously refit a fresh TF-IDF
# vectorizer over the ENTIRE ResolutionPatternEntry table on EVERY SINGLE
# resolution decision - an O(n) full-corpus refit call sitting in the
# live decision path, that would visibly slow down as case history grows
# into the thousands. Cached here, invalidated by ROW COUNT: this table
# is append-only in this codebase (record_resolution_outcome is the only
# writer, and nothing anywhere calls db.delete() on it) - so an unchanged
# count reliably means the cached fit is still valid, and a changed count
# reliably means new rows were added and a refit is genuinely needed.
# get_embedder() itself returns a FRESH, unfitted instance on every call
# (see app/rag/embeddings.py) - not a singleton - so this module owns its
# own fitted-instance cache rather than relying on the factory for it.
#
# ACCURACY NOTE, checked directly rather than assumed: the old code fit
# the vectorizer on [entries... , query_text] TOGETHER, so a word unique
# to the query entered the vocabulary too. This version fits ONLY on the
# entries corpus and embeds the query afterward via the already-fitted
# vectorizer (a query word absent from that vocabulary is simply
# dropped). This produces IDENTICAL similarity rankings to the old
# behavior: similarity is a dot product (entry_vecs @ query_vec), and any
# vocabulary term that appears ONLY in the query and in no entry
# contributes exactly 0 to every entry's score either way, since every
# entry's TF-IDF value for that term is 0 regardless of whether the term
# was ever added to the vectorizer's vocabulary. Verified with a
# regression test (tests/test_few_shot_retrieval_caching.py) comparing
# this cached path's ranking against the old fit-every-call approach on
# a fixed corpus, not just argued here.

_similarity_cache_lock = threading.Lock()
_similarity_cache: dict = {"row_count": None, "embedder": None, "entries": None, "entry_vectors": None}


def _get_or_refit_similarity_index(db: Session):
    """Returns (embedder, entries, entry_vectors) for the current
    ResolutionPatternEntry corpus, refitting only when the row count has
    changed since the last call. Thread-safe (plain lock) - this is
    called from resolution_policy_workflow.py's single-threaded decision
    path, not the orchestrator's parallel diagnosis fan-out, so lock
    contention here is not a real concern; the lock exists for
    correctness under concurrent requests, not for a hot loop.

    `entries` here is a list of plain dicts, NOT live ORM objects -
    deliberately. This codebase's standard pattern is `db = SessionLocal()
    ... finally: db.close()` per call (see every agent node in
    orchestrator.py), so a cache hit on a LATER call, handed a
    DIFFERENT db session than the one active when the cache was last
    populated, would otherwise return ORM objects bound to a session
    that may already be closed - a real DetachedInstanceError risk the
    first version of this cache didn't account for. Extracting plain
    data at fit time avoids this entirely; the cache never holds a
    reference to any SQLAlchemy session or its objects.
    """
    with _similarity_cache_lock:
        current_count = db.query(ResolutionPatternEntry).count()
        if _similarity_cache["embedder"] is not None and _similarity_cache["row_count"] == current_count:
            return _similarity_cache["embedder"], _similarity_cache["entries"], _similarity_cache["entry_vectors"]

        rows = db.query(ResolutionPatternEntry).all()
        entries = [
            {"case_id": e.case_id, "case_feature_summary": e.case_feature_summary,
             "human_final_resolution": e.human_final_resolution}
            for e in rows
        ]
        embedder = get_embedder()
        entry_vectors = None
        if entries:
            corpus = [e["case_feature_summary"] for e in entries]
            embedder.fit(corpus)
            entry_vectors = embedder.embed(corpus)

        _similarity_cache.update(
            row_count=current_count, embedder=embedder, entries=entries, entry_vectors=entry_vectors,
        )
        return embedder, entries, entry_vectors


def invalidate_similarity_cache() -> None:
    """Explicit invalidation - forces the next retrieve_similar_past_resolutions()
    call to refit regardless of row count. Not required for correctness
    (the row-count check in _get_or_refit_similarity_index already
    catches every real change record_resolution_outcome makes) - exists
    for tests that want a deterministic fresh state, and as a documented
    escape hatch if this table is ever mutated outside
    record_resolution_outcome (e.g. a future admin/data-cleanup script)
    in a way that coincidentally leaves the row count unchanged."""
    with _similarity_cache_lock:
        _similarity_cache.update(row_count=None, embedder=None, entries=None, entry_vectors=None)


def retrieve_similar_past_resolutions(db: Session, query_feature_summary: str, k: int = 3) -> list:
    """Dynamic few-shot retrieval - finds the k most similar PAST resolved
    cases by lexical similarity over case_feature_summary, for use as
    in-context examples. Uses the same TfidfEmbedder as the RAG layer
    rather than a separate mechanism. The corpus fit is cached (see
    _get_or_refit_similarity_index above) - only the query itself is
    embedded fresh on every call, since it's different every time by
    definition and embedding one short string is cheap."""
    embedder, entries, entry_vectors = _get_or_refit_similarity_index(db)
    if not entries:
        return []

    query_vec = embedder.embed([query_feature_summary])[0]
    similarities = entry_vectors @ query_vec

    ranked_idx = np.argsort(-similarities)[:k]
    return [
        SimilarPastCase(
            case_id=entries[i]["case_id"],
            case_feature_summary=entries[i]["case_feature_summary"],
            human_final_resolution=entries[i]["human_final_resolution"],
            similarity=float(similarities[i]),
        )
        for i in ranked_idx
    ]


@dataclass
class ClusterAccuracy:
    cluster_key: str
    sample_size: int
    overturn_rate: float


def compute_cluster_accuracy(db: Session, min_sample_size: int = 5) -> list:
    """Computes overturn rate per cluster_key. Clusters below
    min_sample_size are excluded - too few samples to propose a threshold
    change responsibly."""
    entries = db.query(ResolutionPatternEntry).all()
    by_cluster = {}
    for e in entries:
        by_cluster.setdefault(e.cluster_key, []).append(e)

    results = []
    for cluster_key, cluster_entries in by_cluster.items():
        if len(cluster_entries) < min_sample_size:
            continue
        mismatches = sum(1 for e in cluster_entries if e.matched == "mismatch")
        results.append(ClusterAccuracy(
            cluster_key=cluster_key,
            sample_size=len(cluster_entries),
            overturn_rate=mismatches / len(cluster_entries),
        ))
    return results


def propose_threshold_adjustments(
    db: Session,
    current_threshold: float,
    min_sample_size: int = 5,
    raise_step: float = 0.02,
    lower_step: float = 0.05,
) -> list:
    """THE batch job. Computes cluster accuracy and WRITES proposals to
    ThresholdProposalRecord with status='pending_review'. This function
    has no code path that touches ThresholdOverrideRecord - that table is
    only ever written by accept_threshold_proposal(), a separate function
    requiring an explicit human identity. This is not a convention to be
    remembered; it's structurally impossible for this function to
    self-apply anything, which is the actual guardrail."""
    accuracies = compute_cluster_accuracy(db, min_sample_size)
    proposals = []

    for acc in accuracies:
        if acc.overturn_rate == 0.0:
            proposed = min(current_threshold + raise_step, 0.99)
            rationale = (
                f"{acc.sample_size} cases in cluster '{acc.cluster_key}' with 0% overturn rate - "
                f"proposing to RAISE the auto-execute threshold from {current_threshold} to {proposed}, "
                f"widening the auto-execute band for this cluster. Requires human review before taking effect."
            )
        else:
            proposed = max(current_threshold - lower_step, 0.50)
            rationale = (
                f"{acc.sample_size} cases in cluster '{acc.cluster_key}' with {acc.overturn_rate:.0%} "
                f"overturn rate - proposing to LOWER the auto-execute threshold from {current_threshold} "
                f"to {proposed}, narrowing the auto-execute band for this cluster. Requires human review "
                f"before taking effect."
            )

        record = ThresholdProposalRecord(
            cluster_key=acc.cluster_key,
            sample_size=acc.sample_size,
            overturn_rate=acc.overturn_rate,
            current_threshold=current_threshold,
            proposed_threshold=proposed,
            rationale=rationale,
            status="pending_review",
        )
        db.add(record)
        proposals.append(record)

    db.commit()
    return proposals


def accept_threshold_proposal(db: Session, proposal_id: str, accepted_by: str) -> ThresholdOverrideRecord:
    """The ONLY function in this module that writes to
    ThresholdOverrideRecord. Requires a real accepted_by identity - never
    called with "system" or left to a default."""
    if not accepted_by or accepted_by.lower() in ("system", "auto", "automated"):
        raise ValueError(
            f"accept_threshold_proposal requires a real human identity for accepted_by, "
            f"got {accepted_by!r}. This function exists specifically so a human bridges "
            f"proposal -> effect; it cannot be called on the system's own behalf."
        )

    proposal = db.get(ThresholdProposalRecord, proposal_id)
    if proposal is None:
        raise ValueError(f"No such proposal: {proposal_id}")
    if proposal.status != "pending_review":
        raise ValueError(f"Proposal {proposal_id} is already {proposal.status!r}, cannot accept again")

    proposal.status = "accepted"
    proposal.decided_at = datetime.now(timezone.utc)
    proposal.decided_by = accepted_by

    override = db.get(ThresholdOverrideRecord, proposal.cluster_key)
    if override is None:
        override = ThresholdOverrideRecord(
            cluster_key=proposal.cluster_key,
            active_threshold=proposal.proposed_threshold,
            accepted_from_proposal_id=proposal.id,
            accepted_by=accepted_by,
        )
        db.add(override)
    else:
        override.active_threshold = proposal.proposed_threshold
        override.accepted_from_proposal_id = proposal.id
        override.accepted_by = accepted_by

    db.commit()
    return override


def reject_threshold_proposal(db: Session, proposal_id: str, rejected_by: str) -> ThresholdProposalRecord:
    proposal = db.get(ThresholdProposalRecord, proposal_id)
    if proposal is None:
        raise ValueError(f"No such proposal: {proposal_id}")
    proposal.status = "rejected"
    proposal.decided_at = datetime.now(timezone.utc)
    proposal.decided_by = rejected_by
    db.commit()
    return proposal


def get_active_threshold(db: Session, cluster_key: str, default_threshold: float) -> float:
    """What the Resolution-Policy Workflow would actually call to get the
    effective threshold for a cluster - checks for a human-accepted
    override first, falls back to the static default otherwise."""
    override = db.get(ThresholdOverrideRecord, cluster_key)
    return override.active_threshold if override else default_threshold
