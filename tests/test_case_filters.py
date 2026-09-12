"""
Tests for GET /cases filtering and GET /cases/meta/exception-types -
added directly in response to a real request: "I want to view all case
types in the UI." GET /cases previously took zero query parameters,
always returning every case regardless of type or lifecycle state.
"""
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_case_filters_{os.getpid()}_{id(object())}.db")
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


def _create_case(client, order_id, customer_id, exception_type):
    resp = client.post("/api/v1/cases", json={
        "order_id": order_id, "customer_id": customer_id,
        "channel": "direct", "exception_type": exception_type,
    })
    assert resp.status_code == 201
    return resp.json()


def test_list_cases_with_no_filters_returns_everything(isolated_db):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        _create_case(client, "ORD-F1", "CUST-F1", "payment")
        _create_case(client, "ORD-F2", "CUST-F2", "return")

        resp = client.get("/api/v1/cases")
        assert resp.status_code == 200
        assert len(resp.json()) == 2


def test_list_cases_filters_by_exception_type(isolated_db):
    """THE regression test for the actual feature request: filtering
    by exception_type must return ONLY matching cases, not everything."""
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        _create_case(client, "ORD-F3", "CUST-F3", "payment")
        _create_case(client, "ORD-F4", "CUST-F4", "return")
        _create_case(client, "ORD-F5", "CUST-F5", "return")

        resp = client.get("/api/v1/cases?exception_type=return")
        assert resp.status_code == 200
        results = resp.json()
        assert len(results) == 2
        assert all(c["exception_type"] == "return" for c in results)


def test_list_cases_filters_by_state(isolated_db):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        c1 = _create_case(client, "ORD-F6", "CUST-F6", "payment")
        _create_case(client, "ORD-F7", "CUST-F7", "payment")

        resp = client.get("/api/v1/cases?state=detected")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

        resp2 = client.get("/api/v1/cases?state=resolved")
        assert resp2.status_code == 200
        assert len(resp2.json()) == 0


def test_list_cases_filter_with_no_matches_returns_empty_not_error(isolated_db):
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        _create_case(client, "ORD-F8", "CUST-F8", "payment")

        resp = client.get("/api/v1/cases?exception_type=this_type_does_not_exist")
        assert resp.status_code == 200
        assert resp.json() == []


def test_exception_types_meta_endpoint_returns_only_types_actually_in_use(isolated_db):
    """THE regression test proving this reflects REAL data, not a
    hardcoded list of documented-but-possibly-unused values from the
    API schema (which also documents inventory/carrier/fraud, none of
    which any real code path currently assigns)."""
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as client:
        resp_empty = client.get("/api/v1/cases/meta/exception-types")
        assert resp_empty.json() == []

        _create_case(client, "ORD-F9", "CUST-F9", "payment")
        _create_case(client, "ORD-F10", "CUST-F10", "return")
        _create_case(client, "ORD-F11", "CUST-F11", "payment")

        resp = client.get("/api/v1/cases/meta/exception-types")
        assert resp.status_code == 200
        assert resp.json() == ["payment", "return"], "must be de-duplicated and sorted"
