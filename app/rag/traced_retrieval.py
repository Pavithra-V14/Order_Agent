"""
Traced retrieval - wraps app.rag.retrieval.hybrid_search with a tracing
span carrying retrieval_doc_ids_used in its metadata, per architecture
doc 8.8. This is the piece that makes the RAG groundedness metric (8.7)
computable after the fact: a resolution decision's cited doc_id can be
checked against what was ACTUALLY retrieved for that case, not just
trusted at face value.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.tracing import traced_span
from app.rag.retrieval import hybrid_search
from app.cache.retrieval_cache import get_cached_retrieval, set_cached_retrieval, query_cache_key
from app.rag.mutation_lock import get_rag_generation
from app.rag.stampede_guard import run_with_stampede_protection


def traced_hybrid_search(
    db: Session,
    trace_id: str,
    query: str,
    as_of_date: str,
    doc_type: str = None,
    channel: str = None,
    product_category: str = None,
    top_k: int = 5,
) -> list:
    with traced_span(db, trace_id, "rag_retrieval", input_data={
        "query": query, "as_of_date": as_of_date, "doc_type": doc_type,
        "channel": channel, "product_category": product_category,
    }) as span:
        cache_key = query_cache_key(query, as_of_date, doc_type, channel, product_category, top_k)

        def _get_cached():
            return get_cached_retrieval(query, as_of_date, doc_type, channel, product_category, top_k)

        def _compute_and_cache():
            # Generation checked before AND after the (potentially
            # slow) Qdrant read, not just once - see
            # app/rag/mutation_lock.py's module docstring. A single
            # check before the read would miss a mutation that starts
            # and finishes entirely during this read; checking only
            # after would miss one that already finished before this
            # read even started. Comparing both bounds is what makes
            # this correct regardless of exact timing, without needing
            # a lock that would block real requests on each other.
            generation_before = get_rag_generation()
            search_results = hybrid_search(
                query=query, as_of_date=as_of_date, doc_type=doc_type,
                channel=channel, product_category=product_category, top_k=top_k,
            )
            generation_after = get_rag_generation()
            if generation_after == generation_before:
                set_cached_retrieval(query, as_of_date, search_results, doc_type, channel, product_category, top_k)
            else:
                span.metadata["cache_write_skipped_concurrent_mutation"] = True
            return search_results

        # Stampede protection - a.k.a. single-flight/request coalescing:
        # if this exact query is already being computed by another
        # concurrent caller, wait for that one to finish and reuse its
        # result instead of also hitting Qdrant. Found and built
        # directly from a follow-up conversation about a real, known
        # gap - a popular query's cache entry expiring or being
        # invalidated right as real traffic hits it used to mean every
        # concurrent caller independently repeated the same expensive
        # search. See app/rag/stampede_guard.py's own docstring for the
        # full mechanism.
        results, was_leader_cache_hit = run_with_stampede_protection(cache_key, _get_cached, _compute_and_cache)
        span.metadata["cache_hit"] = was_leader_cache_hit

        doc_ids_used = sorted({r.metadata.get("doc_id") for r in results if r.metadata.get("doc_id")})
        span.output_data = {"num_results": len(results), "top_result_texts": [r.text[:100] for r in results]}
        span.metadata["retrieval_doc_ids_used"] = doc_ids_used
        span.metadata["retrieval_doc_versions_used"] = sorted({
            r.metadata.get("version") for r in results if r.metadata.get("version")
        })
    return results
