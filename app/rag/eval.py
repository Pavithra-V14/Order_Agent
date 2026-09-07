"""
Labeled RAG evaluation set - query -> expected doc_id(s) pairs, used to
compute precision/recall@k against the LIVE retrieval pipeline. This is
the missing eval artifact app/core/metrics.py's docstring documented as
needed: precision/recall requires KNOWING what the "correct" answer is,
which isn't computable from telemetry alone (a retrieval that returns
5 results tells you nothing about whether they're the RIGHT 5 without
a labeled ground truth to compare against).

Each entry gives a query, the filters a real caller would apply
(as_of_date, doc_type), and the doc_id(s) that SHOULD appear in the
results for the answer to be considered relevant. Deliberately small
and hand-curated (not auto-generated) - a golden set should encode a
human's judgment about relevance, not the system's own output.
"""
from dataclasses import dataclass, field


@dataclass
class RagEvalCase:
    query: str
    as_of_date: str
    doc_type: str | None
    expected_doc_ids: list = field(default_factory=list)


RAG_EVAL_SET = [
    RagEvalCase(
        query="return window apparel",
        as_of_date="2025-06-15",
        doc_type="return_policy",
        expected_doc_ids=["RET-POLICY-2025-A"],
    ),
    RagEvalCase(
        query="return window apparel",
        as_of_date="2026-05-01",
        doc_type="return_policy",
        expected_doc_ids=["RET-POLICY-2026-A"],
    ),
    RagEvalCase(
        query="fraud risk indicators return abuse",
        as_of_date="2025-06-15",
        doc_type="fraud_policy",
        expected_doc_ids=["FRAUD-POLICY-2025-A"],
    ),
    RagEvalCase(
        query="return window comparison chart category",
        as_of_date="2025-06-15",
        doc_type=None,
        expected_doc_ids=["RET-POLICY-2025-A"],
    ),
]


def run_rag_eval(top_k: int = 5) -> dict:
    """Computes precision@k and recall@k across the whole eval set
    against the LIVE retrieval pipeline (real Qdrant + real embedder,
    whatever's currently configured). Runs in-process deliberately
    (unlike the golden set): retrieval has no equivalent of
    golden_set.py's db-module-reloading side effect, so there's no
    cross-contamination risk calling this directly.
    """
    from app.rag.retrieval import hybrid_search

    per_case = []
    for case in RAG_EVAL_SET:
        results = hybrid_search(
            query=case.query, as_of_date=case.as_of_date, doc_type=case.doc_type, top_k=top_k,
        )
        retrieved_doc_ids = [r.metadata.get("doc_id") for r in results]
        retrieved_set = set(retrieved_doc_ids)
        expected_set = set(case.expected_doc_ids)

        true_positives = len(retrieved_set & expected_set)
        precision = (true_positives / len(retrieved_set)) if retrieved_set else 0.0
        recall = (true_positives / len(expected_set)) if expected_set else None

        per_case.append({
            "query": case.query, "expected_doc_ids": case.expected_doc_ids,
            "retrieved_doc_ids": retrieved_doc_ids,
            "precision_at_k": round(precision, 3),
            "recall_at_k": round(recall, 3) if recall is not None else None,
        })

    valid_recalls = [c["recall_at_k"] for c in per_case if c["recall_at_k"] is not None]
    return {
        "top_k": top_k,
        "case_count": len(RAG_EVAL_SET),
        "avg_precision_at_k": round(sum(c["precision_at_k"] for c in per_case) / len(per_case), 3) if per_case else None,
        "avg_recall_at_k": round(sum(valid_recalls) / len(valid_recalls), 3) if valid_recalls else None,
        "per_case": per_case,
    }


def run_reranker_lift_eval(top_k: int = 5) -> dict:
    """Reranker lift: for each eval query, at what rank position did the
    expected document sit BEFORE reranking, versus AFTER? A positive
    lift means the reranker moved the correct answer higher (a smaller
    rank number is better — rank 0 is first place); zero or negative
    lift means reranking didn't help, or actively hurt.

    The previously-missing metric app/core/metrics.py's docstring
    documented as needing "the retrieval pipeline to record pre-rerank
    AND post-rerank scores" — implemented via hybrid_search's
    pre_rerank_capture callback, added specifically for this, so normal
    retrieval callers are completely unaffected.

    A document absent from a ranking entirely (not in the top candidates
    at all, pre- or post-) is reported as rank=None, not a fabricated
    worst-case number — an absence and a bad-but-present rank are
    different, meaningful outcomes that shouldn't be conflated.
    """
    from app.rag.retrieval import hybrid_search

    per_case = []
    for case in RAG_EVAL_SET:
        if not case.expected_doc_ids:
            continue
        expected_doc_id = case.expected_doc_ids[0]

        captured = {}

        def _capture(candidates, _store=captured):
            _store["pre_rerank_order"] = [c.metadata.get("doc_id") for c in candidates]

        post_rerank_results = hybrid_search(
            query=case.query, as_of_date=case.as_of_date, doc_type=case.doc_type,
            top_k=top_k, pre_rerank_capture=_capture,
        )
        post_rerank_order = [r.metadata.get("doc_id") for r in post_rerank_results]
        pre_rerank_order = captured.get("pre_rerank_order", [])

        pre_rank = pre_rerank_order.index(expected_doc_id) if expected_doc_id in pre_rerank_order else None
        post_rank = post_rerank_order.index(expected_doc_id) if expected_doc_id in post_rerank_order else None

        lift = None
        if pre_rank is not None and post_rank is not None:
            lift = pre_rank - post_rank  # positive = moved up (improved)

        per_case.append({
            "query": case.query, "expected_doc_id": expected_doc_id,
            "pre_rerank_rank": pre_rank, "post_rerank_rank": post_rank, "lift": lift,
        })

    valid_lifts = [c["lift"] for c in per_case if c["lift"] is not None]
    return {
        "top_k": top_k,
        "case_count": len(per_case),
        "avg_lift": round(sum(valid_lifts) / len(valid_lifts), 3) if valid_lifts else None,
        "per_case": per_case,
    }
