"""
Stage 3 memory upgrade — caching tests for
app/agents/learning_loop.py's retrieve_similar_past_resolutions().

Scope, stated honestly (same discipline as this project's other Stage
tests): what's verified here is (1) the cache genuinely avoids refitting
when the corpus hasn't changed, (2) a genuine new entry correctly
triggers a refit rather than serving a stale index, (3) the cached
ranking is IDENTICAL to the old fit-every-call approach for a fixed
corpus (the accuracy claim in learning_loop.py's own module comment,
checked here rather than only argued in a docstring), and (4) caching
plain dicts rather than live ORM objects means a cache hit works
correctly even across a DIFFERENT SQLAlchemy session than the one that
originally populated it - the real DetachedInstanceError risk an
ORM-object cache would have had.
"""
import os
import tempfile

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_few_shot_caching_{os.getpid()}_{id(object())}.db")
    import app.core.db as db_module
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    fresh_engine = create_engine(f"sqlite:///{tmp_path}", connect_args={"check_same_thread": False})
    db_module.Base.metadata.create_all(bind=fresh_engine)
    db_module.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=fresh_engine)
    db_module.engine = fresh_engine

    yield db_module

    fresh_engine.dispose()
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except PermissionError:
        pass


@pytest.fixture(autouse=True)
def reset_similarity_cache():
    """The cache is module-level global state - must be cleared before
    AND after every test, or an earlier test's corpus leaks into a
    later one (same discipline as app.memory.summary_buffer's
    reset_buffers(), just for this cache instead)."""
    from app.agents.learning_loop import invalidate_similarity_cache
    invalidate_similarity_cache()
    yield
    invalidate_similarity_cache()


def _seed_entries(db_module, n, prefix="case-cache"):
    from app.agents.learning_loop import record_resolution_outcome
    db = db_module.SessionLocal()
    for i in range(n):
        record_resolution_outcome(
            db, case_id=f"{prefix}-{i}", cluster_key="payment_direct",
            case_feature_summary=f"root_causes=['payment_issue: reason {i}'], fraud_flag=False",
            agent_proposed_resolution={"action": "deny", "amount_usd": 0.0},
            human_final_resolution={"action": "deny", "amount_usd": 0.0},
        )
    return db


def test_second_call_with_unchanged_corpus_does_not_refit(isolated_db, monkeypatch):
    db = _seed_entries(isolated_db, 3)

    from app.agents.learning_loop import retrieve_similar_past_resolutions
    from app.rag.embeddings import TfidfEmbedder

    fit_calls = {"n": 0}
    original_fit = TfidfEmbedder.fit

    def counting_fit(self, corpus):
        fit_calls["n"] += 1
        return original_fit(self, corpus)

    monkeypatch.setattr(TfidfEmbedder, "fit", counting_fit)

    retrieve_similar_past_resolutions(db, "root_causes=['payment_issue: reason 0'], fraud_flag=False")
    assert fit_calls["n"] == 1

    # Second call, SAME corpus (no new entries recorded in between) -
    # must NOT refit again.
    retrieve_similar_past_resolutions(db, "root_causes=['payment_issue: reason 1'], fraud_flag=False")
    assert fit_calls["n"] == 1, "a call against an unchanged corpus must reuse the cached fit, not refit"


def test_new_entry_triggers_a_refit(isolated_db, monkeypatch):
    db = _seed_entries(isolated_db, 2)

    from app.agents.learning_loop import retrieve_similar_past_resolutions, record_resolution_outcome
    from app.rag.embeddings import TfidfEmbedder

    fit_calls = {"n": 0}
    original_fit = TfidfEmbedder.fit

    def counting_fit(self, corpus):
        fit_calls["n"] += 1
        return original_fit(self, corpus)

    monkeypatch.setattr(TfidfEmbedder, "fit", counting_fit)

    retrieve_similar_past_resolutions(db, "root_causes=['payment_issue: reason 0'], fraud_flag=False")
    assert fit_calls["n"] == 1

    # A genuinely new entry is recorded - the row count changes.
    record_resolution_outcome(
        db, case_id="case-cache-new", cluster_key="payment_direct",
        case_feature_summary="root_causes=['payment_issue: a brand new reason'], fraud_flag=False",
        agent_proposed_resolution={"action": "deny", "amount_usd": 0.0},
        human_final_resolution={"action": "deny", "amount_usd": 0.0},
    )

    retrieve_similar_past_resolutions(db, "root_causes=['payment_issue: reason 1'], fraud_flag=False")
    assert fit_calls["n"] == 2, "a new row must invalidate the cache and trigger exactly one refit"


def test_cached_ranking_matches_uncached_fit_every_call_approach(isolated_db):
    """THE accuracy regression test for the claim made in
    learning_loop.py's own module comment: caching must not change WHICH
    past cases get surfaced or in what order, compared to fitting fresh
    on every call (the pre-Stage-3 behavior)."""
    db = _seed_entries(isolated_db, 5, prefix="case-accuracy")
    # One genuinely different-topic entry so ranking isn't trivially uniform.
    from app.agents.learning_loop import record_resolution_outcome
    record_resolution_outcome(
        db, case_id="case-accuracy-inventory", cluster_key="inventory_direct",
        case_feature_summary="root_causes=['inventory_issue: insufficient stock'], fraud_flag=False",
        agent_proposed_resolution={"action": "reship", "amount_usd": 0.0},
        human_final_resolution={"action": "reship", "amount_usd": 0.0},
    )

    query = "root_causes=['payment_issue: reason 2'], fraud_flag=False"

    from app.agents.learning_loop import retrieve_similar_past_resolutions
    cached_result = retrieve_similar_past_resolutions(db, query, k=3)

    # Recompute the OLD way directly (fit on entries+query together,
    # every call) to compare against.
    from app.core.db import ResolutionPatternEntry
    from app.rag.embeddings import TfidfEmbedder
    rows = db.query(ResolutionPatternEntry).all()
    corpus = [r.case_feature_summary for r in rows] + [query]
    old_embedder = TfidfEmbedder()
    old_embedder.fit(corpus)
    vectors = old_embedder.embed(corpus)
    query_vec = vectors[-1]
    entry_vecs = vectors[:-1]
    similarities = entry_vecs @ query_vec
    old_ranked_idx = np.argsort(-similarities)[:3]
    old_case_ids = [rows[i].case_id for i in old_ranked_idx]

    cached_case_ids = [c.case_id for c in cached_result]
    assert cached_case_ids == old_case_ids, (
        "cached retrieval must rank identically to the old fit-every-call approach - "
        f"got {cached_case_ids}, expected {old_case_ids}"
    )


def test_cache_hit_survives_a_closed_originating_session(isolated_db):
    """THE DetachedInstanceError regression test: this codebase's normal
    pattern is db = SessionLocal(); ...; db.close() per call. A cache
    that stored live ORM objects from the FIRST session would break the
    SECOND call once that first session is closed. This proves the
    cache (plain dicts, not ORM objects) survives exactly that."""
    db1 = _seed_entries(isolated_db, 2, prefix="case-detach")

    from app.agents.learning_loop import retrieve_similar_past_resolutions
    retrieve_similar_past_resolutions(db1, "root_causes=['payment_issue: reason 0'], fraud_flag=False")
    db1.close()  # the session that populated the cache is now gone

    db2 = isolated_db.SessionLocal()
    try:
        # Must not raise sqlalchemy.orm.exc.DetachedInstanceError.
        result = retrieve_similar_past_resolutions(
            db2, "root_causes=['payment_issue: reason 1'], fraud_flag=False",
        )
        assert len(result) == 2
        assert all(isinstance(r.case_feature_summary, str) for r in result)
    finally:
        db2.close()


def test_empty_corpus_returns_empty_list_and_does_not_crash(isolated_db):
    db = isolated_db.SessionLocal()
    from app.agents.learning_loop import retrieve_similar_past_resolutions
    assert retrieve_similar_past_resolutions(db, "anything") == []


def test_invalidate_similarity_cache_forces_a_refit(isolated_db, monkeypatch):
    db = _seed_entries(isolated_db, 2)

    from app.agents.learning_loop import retrieve_similar_past_resolutions, invalidate_similarity_cache
    from app.rag.embeddings import TfidfEmbedder

    fit_calls = {"n": 0}
    original_fit = TfidfEmbedder.fit

    def counting_fit(self, corpus):
        fit_calls["n"] += 1
        return original_fit(self, corpus)

    monkeypatch.setattr(TfidfEmbedder, "fit", counting_fit)

    retrieve_similar_past_resolutions(db, "query 1")
    assert fit_calls["n"] == 1

    invalidate_similarity_cache()
    retrieve_similar_past_resolutions(db, "query 2")  # same corpus, but cache was forced clear
    assert fit_calls["n"] == 2


# --- Embedding backend: already respects MISTRAL_API_KEY, verified -----
#
# CORRECTION to an earlier (incorrect) claim made before this test
# existed: both the pre-Stage-3 code and this cache both call
# get_embedder() (app/rag/embeddings.py's factory), which ALREADY
# auto-selects MistralEmbedder when MISTRAL_API_KEY is configured -
# TF-IDF was never hardcoded here. This test proves it rather than
# asserting it.

def test_similarity_cache_uses_mistral_embedder_when_configured(isolated_db, monkeypatch):
    db = _seed_entries(isolated_db, 2)

    os.environ["MISTRAL_API_KEY"] = "fake_test_key_for_verification"
    from app.core.config import get_settings
    get_settings.cache_clear()

    captured = {}

    class FakeMistralEmbedder:
        def fit(self, corpus):
            captured["fit_called"] = True

        def embed(self, texts):
            captured["embedded_texts"] = texts
            return np.ones((len(texts), 4), dtype="float32")

        @property
        def dim(self):
            return 4

    import app.rag.embeddings as embeddings_module
    monkeypatch.setattr(embeddings_module, "MistralEmbedder", FakeMistralEmbedder)

    from app.agents.learning_loop import retrieve_similar_past_resolutions
    retrieve_similar_past_resolutions(db, "root_causes=['payment_issue: reason 0'], fraud_flag=False")

    assert captured.get("fit_called") is True, (
        "MISTRAL_API_KEY was set - get_embedder() must have selected MistralEmbedder, not TfidfEmbedder"
    )

    os.environ.pop("MISTRAL_API_KEY", None)
    get_settings.cache_clear()
