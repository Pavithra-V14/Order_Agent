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

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from app.core.db import ResolutionPatternEntry, ThresholdProposalRecord, ThresholdOverrideRecord
from app.rag.embeddings import TfidfEmbedder


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


def retrieve_similar_past_resolutions(db: Session, query_feature_summary: str, k: int = 3) -> list:
    """Dynamic few-shot retrieval - finds the k most similar PAST resolved
    cases by lexical similarity over case_feature_summary, for use as
    in-context examples. Uses the same TfidfEmbedder as the RAG layer
    rather than a separate mechanism."""
    entries = db.query(ResolutionPatternEntry).all()
    if not entries:
        return []

    corpus = [e.case_feature_summary for e in entries] + [query_feature_summary]
    embedder = TfidfEmbedder(max_features=256)
    embedder.fit(corpus)
    vectors = embedder.embed(corpus)
    query_vec = vectors[-1]
    entry_vecs = vectors[:-1]

    similarities = entry_vecs @ query_vec

    ranked_idx = np.argsort(-similarities)[:k]
    return [
        SimilarPastCase(
            case_id=entries[i].case_id,
            case_feature_summary=entries[i].case_feature_summary,
            human_final_resolution=entries[i].human_final_resolution,
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
