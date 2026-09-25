"""
MANUAL CHECK - Stage 3: cached few-shot retrieval.

Important context: Stage 3 is a PERFORMANCE change to an existing
mechanism, not a new visible feature - the actual OUTPUT (similar past
cases) is the same `similar_past_cases` field already visible via
GET /api/v1/cases/{case_id} -> resolution_decision.similar_past_cases
(see the manual-check writeup for Stage 1/2). What THIS script proves
is the thing that isn't visible through the API at all: that the
vectorizer is no longer refit on every single call.

Usage:
    python scripts/manual_check_stage3_retrieval_caching.py
"""
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import SessionLocal, init_db
from app.agents.learning_loop import (
    record_resolution_outcome, retrieve_similar_past_resolutions, invalidate_similarity_cache,
)
from app.rag.embeddings import get_embedder


def _seed_history_case(db, case_id):
    """ResolutionPatternEntry.case_id has a REAL foreign key to
    exception_cases.id (app/core/db.py) - enforced by Postgres, NOT
    by SQLite (which doesn't check FKs by default). This script's
    first version used made-up case_id strings with no backing
    ExceptionCase row at all, which worked in this sandbox's SQLite
    but failed with a real ForeignKeyViolation the first time it ran
    against a real Postgres database. Fixed by actually creating a
    minimal real case row for each seeded history entry."""
    from app.core.db import ExceptionCase, CaseState
    if not db.get(ExceptionCase, case_id):
        db.add(ExceptionCase(id=case_id, order_id=f"ORD-{case_id}", customer_id=f"CUST-{case_id}",
                              channel="direct", exception_type="payment", state=CaseState.RESOLVED))
        db.commit()


def main():
    init_db()
    db = SessionLocal()
    invalidate_similarity_cache()

    print("Seeding 5 past resolved cases into the resolution-pattern table...")
    for i in range(5):
        case_id = f"case-stage3-seed-{i}"
        _seed_history_case(db, case_id)
        record_resolution_outcome(
            db, case_id=case_id, cluster_key="payment_direct",
            case_feature_summary=f"root_causes=['payment_issue: reason {i}'], fraud_flag=False",
            agent_proposed_resolution={"action": "deny", "amount_usd": 0.0},
            human_final_resolution={"action": "deny", "amount_usd": 0.0},
        )

    # Count real .fit() calls by wrapping the actual method - this is
    # the only way to SEE caching behavior directly, since there's no
    # API surface for "was the vectorizer refit."
    # Real bug found running this against a real environment with
    # MISTRAL_API_KEY configured: this script hardcoded TfidfEmbedder as
    # the monkeypatch target, but get_embedder() (app/rag/embeddings.py)
    # auto-selects MistralEmbedder whenever a real Mistral key is set -
    # so the patch never intercepted anything at all, and the printed
    # count stayed 0 throughout regardless of real caching behavior.
    # Fixed by detecting whichever embedder class is ACTUALLY active in
    # this environment and patching that one instead of assuming.
    embedder_class = type(get_embedder())
    print(f"Active embedder class in this environment: {embedder_class.__name__}")

    fit_calls = {"n": 0}
    original_fit = embedder_class.fit

    def counting_fit(self, corpus):
        fit_calls["n"] += 1
        return original_fit(self, corpus)

    embedder_class.fit = counting_fit
    try:
        query = "root_causes=['payment_issue: reason 2'], fraud_flag=False"

        print("\nFirst call (must refit - cache is cold)...")
        results_1 = retrieve_similar_past_resolutions(db, query, k=3)
        print(f"  fit() call count so far: {fit_calls['n']} (expected 1)")
        print(f"  top match: {results_1[0].case_id if results_1 else 'none'}")

        print("\nSecond call, SAME corpus, DIFFERENT query text...")
        results_2 = retrieve_similar_past_resolutions(
            db, "root_causes=['payment_issue: reason 4'], fraud_flag=False", k=3,
        )
        print(f"  fit() call count so far: {fit_calls['n']} (expected STILL 1 - this is the cache working)")

        print("\nRecording a genuinely NEW resolution outcome (corpus changes)...")
        _seed_history_case(db, "case-stage3-new-entry")
        record_resolution_outcome(
            db, case_id="case-stage3-new-entry", cluster_key="payment_direct",
            case_feature_summary="root_causes=['payment_issue: a brand new reason'], fraud_flag=False",
            agent_proposed_resolution={"action": "deny", "amount_usd": 0.0},
            human_final_resolution={"action": "deny", "amount_usd": 0.0},
        )

        print("\nThird call, corpus HAS changed...")
        retrieve_similar_past_resolutions(db, query, k=3)
        print(f"  fit() call count so far: {fit_calls['n']} (expected 2 - a new row correctly invalidated the cache)")
    finally:
        embedder_class.fit = original_fit

    print("\n" + "=" * 70)
    if fit_calls["n"] == 2:
        print("CONFIRMED: caching works - 2 real refits for 3 calls across a corpus")
        print("change, not 3 refits (the old, pre-Stage-3 behavior would show 3).")
    else:
        print(f"UNEXPECTED: fit() was called {fit_calls['n']} times (expected exactly 2) - investigate.")
    print("=" * 70)


if __name__ == "__main__":
    main()
