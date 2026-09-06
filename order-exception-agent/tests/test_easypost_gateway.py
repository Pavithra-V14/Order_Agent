"""
Tests for EasyPostGateway using respx to mock EasyPost's actual REST
endpoints with realistic response shapes.
"""
import os
import tempfile

import httpx
import pytest
import respx

EASYPOST_BASE = "https://api.easypost.com/v2"


@pytest.fixture(autouse=True)
def easypost_settings():
    os.environ["EASYPOST_API_KEY"] = "EZTK_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()
    from app.tools.carrier import reset_fake_carrier
    reset_fake_carrier()
    yield
    os.environ.pop("EASYPOST_API_KEY", None)
    get_settings.cache_clear()
    reset_fake_carrier()


@pytest.fixture
def isolated_db():
    """Same isolation approach as test_shippo_gateway.py — swap the
    engine/session on the ALREADY-mapped db_module classes rather than
    importlib.reload()ing the module, which was confirmed to leave stale
    mapper registrations behind when multiple test functions in one file
    each reload it within a single pytest process."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_easypost_{os.getpid()}_{id(object())}.db")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    import app.core.db as db_module
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    fresh_engine = create_engine(f"sqlite:///{tmp_path}", connect_args={"check_same_thread": False})
    db_module.Base.metadata.create_all(bind=fresh_engine)
    db_module.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=fresh_engine)
    db_module.engine = fresh_engine

    yield db_module

    fresh_engine.dispose()
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_easypost_gateway_raises_clearly_without_api_key():
    os.environ.pop("EASYPOST_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.tools.carrier import EasyPostGateway
    with pytest.raises(RuntimeError, match="EASYPOST_API_KEY not configured"):
        EasyPostGateway()


@respx.mock
def test_easypost_get_tracking_status_parses_real_response_shape(easypost_settings):
    from app.tools.carrier import EasyPostGateway

    respx.post(f"{EASYPOST_BASE}/trackers").mock(return_value=httpx.Response(200, json={
        "id": "trk_fake123",
        "object": "Tracker",
        "tracking_code": "TRK123456",
        "status": "in_transit",
        "carrier": "USPS",
    }))

    gateway = EasyPostGateway()
    result = gateway.get_tracking_status("TRK123456")

    assert result["status"] == "in_transit"
    assert result["tracking_number"] == "TRK123456"


@respx.mock
def test_easypost_get_tracking_status_unknown_number_returns_unknown_not_crash(easypost_settings):
    from app.tools.carrier import EasyPostGateway

    respx.post(f"{EASYPOST_BASE}/trackers").mock(
        return_value=httpx.Response(422, json={"error": {"message": "Invalid tracking code"}})
    )

    gateway = EasyPostGateway()
    result = gateway.get_tracking_status("NOT-A-REAL-CODE")

    assert result["status"] == "unknown"


@respx.mock
def test_easypost_generate_return_label_parses_real_response_shape(easypost_settings, isolated_db):
    from app.tools.carrier import EasyPostGateway

    respx.post(f"{EASYPOST_BASE}/shipments").mock(return_value=httpx.Response(200, json={
        "id": "shp_fake123",
        "object": "Shipment",
        "rates": [{"id": "rate_fake1", "carrier": "USPS", "service": "Priority", "rate": "7.50"}],
    }))
    respx.post(f"{EASYPOST_BASE}/shipments/shp_fake123/buy").mock(return_value=httpx.Response(200, json={
        "id": "shp_fake123",
        "object": "Shipment",
        "tracking_code": "EZ1000000001",
        "postage_label": {"label_url": "https://easypost-files.s3.amazonaws.com/files/fake_label.png"},
    }))

    db = isolated_db.SessionLocal()
    gateway = EasyPostGateway()
    result = gateway.generate_return_label(db, order_id="ORD-EP-1", idempotency_key="ep-key-1")

    assert result["tracking_number"] == "EZ1000000001"
    assert "fake_label.png" in result["label_url"]
    assert result["_was_replayed"] is False
    db.close()


@respx.mock
def test_easypost_generate_return_label_is_idempotent(easypost_settings, isolated_db):
    """Confirms the EasyPostGateway path goes through the SAME
    with_idempotency() mechanism as the fake gateway."""
    from app.tools.carrier import EasyPostGateway

    shipment_route = respx.post(f"{EASYPOST_BASE}/shipments").mock(return_value=httpx.Response(200, json={
        "id": "shp_fake456", "object": "Shipment",
        "rates": [{"id": "rate_fake1", "carrier": "USPS", "service": "Priority", "rate": "7.50"}],
    }))
    respx.post(f"{EASYPOST_BASE}/shipments/shp_fake456/buy").mock(return_value=httpx.Response(200, json={
        "id": "shp_fake456", "object": "Shipment", "tracking_code": "EZ1000000002",
        "postage_label": {"label_url": "https://easypost-files.s3.amazonaws.com/files/fake2.png"},
    }))

    db = isolated_db.SessionLocal()
    gateway = EasyPostGateway()
    r1 = gateway.generate_return_label(db, order_id="ORD-EP-2", idempotency_key="ep-key-2")
    r2 = gateway.generate_return_label(db, order_id="ORD-EP-2", idempotency_key="ep-key-2")

    assert r1["tracking_number"] == r2["tracking_number"]
    assert r2["_was_replayed"] is True
    assert shipment_route.call_count == 1, "a replayed call must NOT create a second real shipment"
    db.close()


def test_get_carrier_gateway_returns_easypost_when_key_configured(easypost_settings):
    from app.tools.carrier import get_carrier_gateway, EasyPostGateway
    gateway = get_carrier_gateway()
    assert isinstance(gateway, EasyPostGateway)


def test_get_carrier_gateway_returns_fake_when_no_key():
    os.environ.pop("EASYPOST_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.tools.carrier import get_carrier_gateway, FakeCarrierGateway
    gateway = get_carrier_gateway()
    assert isinstance(gateway, FakeCarrierGateway)
