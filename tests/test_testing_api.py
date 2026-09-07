"""
Tests for app/api/v1/testing.py - the golden-set run-and-label
endpoints. THE key property under test: triggering a golden-set run
from the live app must NOT corrupt the live app's own database
connection - the real production bug found and fixed via subprocess
isolation, not just an ordinary feature test.
"""
import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_testing_api_{os.getpid()}_{id(object())}.db")
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


def test_golden_set_run_persists_a_labeled_record(isolated_db):
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/api/v1/testing/golden-set/run", json={"label": "my-test-run"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["label"] == "my-test-run"
        assert data["total_count"] == 10
        assert len(data["results"]) == 10

        runs = client.get("/api/v1/testing/golden-set/runs").json()
        assert len(runs) == 1
        assert runs[0]["label"] == "my-test-run"


def test_golden_set_run_does_not_corrupt_the_live_apps_own_database(isolated_db):
    """THE regression test for the actual production bug found: an
    earlier version of this endpoint called the golden-set scenario
    functions IN-PROCESS. Each scenario calls
    importlib.reload(app.core.db), swapping the database connection to
    a throwaway temp file - corrupting the live app's OWN database
    connection for the rest of the process's lifetime the moment this
    endpoint was hit once. Fixed by running the golden set in a
    genuinely separate subprocess instead."""
    from app.main import app
    from app.core.db import SessionLocal, ExceptionCase, CaseState

    with TestClient(app) as client:
        resp_before = client.get("/api/v1/health")
        assert resp_before.status_code == 200

        run_resp = client.post("/api/v1/testing/golden-set/run", json={"label": "corruption-check"})
        assert run_resp.status_code == 200

        resp_after = client.get("/api/v1/health")
        assert resp_after.status_code == 200
        cases_resp = client.get("/api/v1/cases")
        assert cases_resp.status_code == 200

    db = SessionLocal()
    case = ExceptionCase(id="post-goldenset-case", order_id="ORD-X", customer_id="CUST-X",
                          channel="direct", exception_type="return", state=CaseState.DETECTED)
    db.add(case)
    db.commit()
    retrieved = db.get(ExceptionCase, "post-goldenset-case")
    assert retrieved is not None
    db.close()


def test_get_golden_set_run_detail(isolated_db):
    from app.main import app
    with TestClient(app) as client:
        run_resp = client.post("/api/v1/testing/golden-set/run", json={"label": "detail-check"})
        run_id = run_resp.json()["run_id"]

        detail_resp = client.get(f"/api/v1/testing/golden-set/runs/{run_id}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["label"] == "detail-check"
        assert len(detail["results"]) == 10
        assert all("name" in r and "passed" in r and "detail" in r for r in detail["results"])


def test_get_golden_set_run_detail_404_for_unknown_id(isolated_db):
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/testing/golden-set/runs/does-not-exist")
        assert resp.status_code == 404


def test_list_available_scenarios_returns_all_eight_with_descriptions():
    from app.main import app
    with TestClient(app) as client:
        resp = client.get("/api/v1/testing/golden-set/scenarios")
        assert resp.status_code == 200
        scenarios = resp.json()
        assert len(scenarios) == 10
        assert all("name" in s and "description" in s for s in scenarios)
        assert all(s["description"] for s in scenarios), "every scenario must have a non-empty description"


def test_golden_set_run_with_selected_scenarios_runs_only_those(isolated_db):
    """THE regression test for the actual feature request: running a
    SUBSET of scenarios must genuinely run only that subset, not
    silently fall back to running all 8."""
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/api/v1/testing/golden-set/run", json={
            "label": "selective-run", "scenario_names": ["scenario_tier1_hard_block"],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_count"] == 1, "must run ONLY the selected scenario, not all 8"
        assert data["results"][0]["name"] == "tier1_hard_block"


def test_golden_set_run_with_no_scenario_names_runs_all_eight(isolated_db):
    """Backward compatibility: omitting scenario_names entirely (or an
    empty list) must still run everything, matching the original
    behavior before selection was added."""
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/api/v1/testing/golden-set/run", json={"label": "run-everything"})
        assert resp.status_code == 200
        assert resp.json()["total_count"] == 10


def test_golden_set_run_with_unknown_scenario_name_returns_clean_error(isolated_db):
    from app.main import app
    with TestClient(app) as client:
        resp = client.post("/api/v1/testing/golden-set/run", json={
            "label": "bad-selection", "scenario_names": ["scenario_does_not_exist"],
        })
        assert resp.status_code == 400
