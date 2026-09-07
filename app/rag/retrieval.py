"""
Retrieval per architecture doc 8.2.4:
  1. Metadata pre-filter applied BEFORE similarity search (temporal
     correctness: 8.2.4/8.2.3) -- not a post-hoc re-rank.
  2. Hybrid dense + sparse (BM25) search, combined via Reciprocal Rank
     Fusion: RRF_Score(d) = sum(1 / (60 + rank_r(d))) across rankers.
  3. Selective MMR for broad/exploratory queries only (not single-fact
     policy lookups, where MMR's diversity goal actively works against
     retrieving the one correct, date-filtered answer).
  4. Reranking -- architecture doc specifies self-hosted BGE-Reranker;
     this sandbox substitutes a lightweight lexical-overlap rerank behind
     the same interface (see LightweightReranker docstring for the swap
     point), for the same disk-budget reason as embeddings.py.
"""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from qdrant_client.models import Filter

from app.rag.embeddings import BaseEmbedder, get_embedder
from app.rag.vectorstore import get_qdrant_client, build_temporal_filter, search
from app.core.config import get_settings


@dataclass
class RetrievedChunk:
    node_id: str
    text: str
    score: float
    metadata: dict


class BaseReranker:
    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError


class LightweightReranker(BaseReranker):
    """Sandbox substitute for architecture doc 8.10's self-hosted
    BGE-Reranker (a real cross-encoder, ~1-2GB with torch — same disk
    constraint as embeddings.py). This implementation re-scores candidates
    by exact query-term overlap density as a cheap proxy for relevance,
    which is a genuinely useful second-stage filter even without a
    cross-encoder — it just doesn't capture semantic (non-lexical) matches
    the way a real reranker would. Swap point: replace `rerank()`'s body
    with a call to a loaded `BAAI/bge-reranker-base` cross-encoder's
    `.predict([(query, c.text) for c in candidates])`, ranked descending.
    """

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        query_terms = set(query.lower().split())
        def overlap_score(c: RetrievedChunk) -> float:
            text_terms = set(c.text.lower().split())
            if not text_terms:
                return 0.0
            overlap = len(query_terms & text_terms)
            return overlap / max(len(query_terms), 1) + 0.001 * c.score  # tie-break on original score
        reranked = sorted(candidates, key=overlap_score, reverse=True)
        return reranked[:top_k]


def _rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion, per architecture doc Layer 7's exact
    formula: RRF_Score(d) = sum(1 / (60 + rank_r(d))) across each ranking
    list `rankings` the doc id appears in."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def _mmr_select(candidates: list[RetrievedChunk], embedder: BaseEmbedder,
                 top_k: int, lambda_mult: float = 0.5) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance -- used only for broad/exploratory
    queries (architecture doc 8.2.4), never for single-fact policy
    lookups. Balances relevance (original score) against diversity
    (dissimilarity to already-selected results)."""
    import numpy as np
    if len(candidates) <= top_k:
        return candidates

    texts = [c.text for c in candidates]
    vecs = embedder.embed(texts)  # re-embeds candidates for pairwise similarity

    selected_idx: list[int] = []
    remaining_idx = list(range(len(candidates)))

    # seed with the highest-scored candidate
    first = max(remaining_idx, key=lambda i: candidates[i].score)
    selected_idx.append(first)
    remaining_idx.remove(first)

    while len(selected_idx) < top_k and remaining_idx:
        def mmr_score(i):
            relevance = candidates[i].score
            diversity = max(
                float(np.dot(vecs[i], vecs[j])) for j in selected_idx
            )
            return lambda_mult * relevance - (1 - lambda_mult) * diversity

        next_idx = max(remaining_idx, key=mmr_score)
        selected_idx.append(next_idx)
        remaining_idx.remove(next_idx)

    return [candidates[i] for i in selected_idx]


def hybrid_search(
    query: str,
    as_of_date: str,
    doc_type: str | None = None,
    channel: str | None = None,
    product_category: str | None = None,
    top_k: int = 5,
    use_mmr: bool = False,
    pre_rerank_capture: callable = None,
) -> list[RetrievedChunk]:
    """Full retrieval pipeline: metadata pre-filter -> dense search (Qdrant,
    filter applied server-side) to get the filtered candidate pool -> BM25
    sparse re-ranking over that SAME filtered pool (so temporal/channel/
    category correctness holds on the sparse side too, not just dense) ->
    RRF fusion of the two rankings -> optional MMR -> lightweight rerank -> top_k.

    pre_rerank_capture: optional callback invoked with the candidate list
    exactly as it stood BEFORE the final reranking step — added
    specifically to compute reranker lift (app/rag/eval.py) without
    changing this function's return type or behavior for any other
    caller. None of the existing call sites pass this, so nothing about
    normal retrieval changes; it's purely an observation hook.
    """
    settings = get_settings()
    client = get_qdrant_client()
    qfilter: Filter = build_temporal_filter(as_of_date, channel, product_category, doc_type)

    embedder = get_embedder()
    # Fit on a representative sample so the vectorizer has vocabulary --
    # in production this isn't needed (BGE-M3 is pretrained); see embeddings.py.
    all_points, _ = client.scroll(collection_name=settings.qdrant_collection, limit=1000)
    if all_points:
        embedder.fit([p.payload.get("text", "") for p in all_points])
    query_vec = embedder.embed([query])[0]

    # Dense search -- metadata filter applied server-side, defines the filtered candidate pool
    dense_hits = search(client, settings.qdrant_collection, query_vec, qfilter, limit=top_k * 4)
    dense_ranking = [str(h.id) for h in dense_hits]
    dense_by_id = {str(h.id): h for h in dense_hits}

    # Sparse (BM25) re-ranking over the SAME filtered pool -- never over the
    # full unfiltered corpus, or a temporally-wrong doc could re-enter via
    # the sparse side even though dense correctly filtered it out.
    sparse_ranking: list[str] = []
    if dense_hits:
        pool_ids = [str(h.id) for h in dense_hits]
        pool_texts = [h.payload.get("text", "") for h in dense_hits]
        tokenized = [t.lower().split() for t in pool_texts]
        if any(tokenized):
            bm25 = BM25Okapi(tokenized)
            scores = bm25.get_scores(query.lower().split())
            order = sorted(range(len(pool_ids)), key=lambda i: scores[i], reverse=True)
            sparse_ranking = [pool_ids[i] for i in order]

    rankings = [r for r in (dense_ranking, sparse_ranking) if r]
    fused_scores = _rrf_fuse(rankings) if rankings else {}

    candidates = [
        RetrievedChunk(
            node_id=doc_id,
            text=dense_by_id[doc_id].payload.get("text", ""),
            score=fused_scores.get(doc_id, 0.0),
            metadata=dense_by_id[doc_id].payload,
        )
        for doc_id in fused_scores
        if doc_id in dense_by_id
    ]
    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[: top_k * 3]  # cap before MMR/rerank for cost

    if use_mmr:
        candidates = _mmr_select(candidates, embedder, top_k=top_k * 2)

    if pre_rerank_capture is not None:
        pre_rerank_capture(list(candidates))

    reranker = LightweightReranker()
    results = reranker.rerank(query, candidates, top_k=top_k)

    from app.core.console_log import log_rag_retrieval
    log_rag_retrieval(query, len(results), doc_type=doc_type)

    return results
