"""
Tests for the memory evaluation framework (app/memory/eval.py) - found
completely missing during a direct audit: RAG had a real, labeled
golden set (app/rag/eval.py); the memory layer had zero dedicated
evaluation of any kind, despite being just as capable of silently
regressing (a filtering-logic bug that miscounts return cases, or a
temporal-ordering bug that picks the wrong "most recent" episode,
would have shipped completely unnoticed).
"""
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_memory_eval_{os.getpid()}_{id(object())}.db")
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


def test_memory_eval_all_scenarios_pass_against_the_real_sql_backed_memory(isolated_db):
    """THE main proof: every hand-labeled scenario's expected values
    must match what summarize_customer_risk_profile() genuinely
    computes — not asserted in the abstract, run for real."""
    from app.memory.eval import run_memory_eval

    db = isolated_db.SessionLocal()
    result = run_memory_eval(db)

    assert result["total"] == len(result["results"])
    failed = [r for r in result["results"] if not r["passed"]]
    assert result["passed"] == result["total"], (
        f"expected all memory eval scenarios to pass; failures: {failed}"
    )
    db.close()


def test_memory_eval_detects_a_genuine_regression(isolated_db, monkeypatch):
    """THE regression test for the eval framework itself: if
    summarize_customer_risk_profile() were broken (e.g. miscounting
    return cases), this eval must actually catch it, not silently pass
    regardless of the real computed values."""
    import app.memory.episodic as episodic_module
    from app.memory.eval import run_memory_eval

    real_summarize = episodic_module.summarize_customer_risk_profile

    def broken_summarize(db, customer_id):
        # Deliberately wrong: always reports zero return cases,
        # regardless of real history — simulating a real regression.
        real_result = real_summarize(db, customer_id)
        real_result["total_return_cases"] = 0
        return real_result

    monkeypatch.setattr(episodic_module, "summarize_customer_risk_profile", broken_summarize)
    # run_memory_eval imports summarize_customer_risk_profile locally
    # inside the function, so patch it at the source module - re-import
    # eval fresh to make sure the patched reference is actually used.
    import importlib
    import app.memory.eval as eval_module
    importlib.reload(eval_module)

    db = isolated_db.SessionLocal()
    result = eval_module.run_memory_eval(db)

    failed_names = [r["name"] for r in result["results"] if not r["passed"]]
    assert "multiple_returns_correctly_counted" in failed_names, (
        "the eval must genuinely detect a broken total_return_cases computation, not pass regardless"
    )
    db.close()

    importlib.reload(eval_module)  # restore the real, unpatched behavior for other tests


def test_memory_eval_scenarios_use_isolated_throwaway_customer_ids(isolated_db):
    """Each scenario must use its own dedicated customer_id — running
    the eval twice in a row must not accumulate episodes across runs
    and silently change the expected counts."""
    from app.memory.eval import run_memory_eval

    db = isolated_db.SessionLocal()
    result1 = run_memory_eval(db)
    result2 = run_memory_eval(db)

    # THE regression risk this proves against: if eval customer_ids
    # were reused with data ACCUMULATING (not reset) across repeated
    # runs, "multiple_returns_correctly_counted" would grow from 3 to 6
    # on the second run and start failing its own fixed expectation.
    result2_case = next(r for r in result2["results"] if r["name"] == "multiple_returns_correctly_counted")
    assert result2_case["passed"], (
        "running the eval twice must not accumulate episodes across runs and break a scenario that "
        "passed on the first run"
    )
    db.close()


def test_memory_eval_api_endpoint_returns_real_results():
    """The actual HTTP endpoint a person would hit from the Testing
    page — proven end to end, not just the underlying function."""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        resp = client.get("/api/v1/testing/memory-eval")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] > 0
        assert data["passed"] == data["total"]
