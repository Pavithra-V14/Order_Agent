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
from app.cache.retrieval_cache import get_cached_retrieval, set_cached_retrieval


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
        cached_results = get_cached_retrieval(query, as_of_date, doc_type, channel, product_category, top_k)
        if cached_results is not None:
            results = cached_results
            span.metadata["cache_hit"] = True
        else:
            results = hybrid_search(
                query=query, as_of_date=as_of_date, doc_type=doc_type,
                channel=channel, product_category=product_category, top_k=top_k,
            )
            set_cached_retrieval(query, as_of_date, results, doc_type, channel, product_category, top_k)
            span.metadata["cache_hit"] = False

        doc_ids_used = sorted({r.metadata.get("doc_id") for r in results if r.metadata.get("doc_id")})
        span.output_data = {"num_results": len(results), "top_result_texts": [r.text[:100] for r in results]}
        span.metadata["retrieval_doc_ids_used"] = doc_ids_used
        span.metadata["retrieval_doc_versions_used"] = sorted({
            r.metadata.get("version") for r in results if r.metadata.get("version")
        })
    return results
