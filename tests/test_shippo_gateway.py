"""
Tests for ShippoGateway using respx to mock Shippo's actual REST
endpoints with realistic response shapes.
"""
import os
import tempfile

import httpx
import pytest
import respx

SHIPPO_BASE = "https://api.goshippo.com"

@pytest.fixture(autouse=True)
def shippo_settings():
    os.environ["SHIPPO_API_KEY"] = "shippo_test_fake_key"
    os.environ.pop("EASYPOST_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()
    from app.tools.carrier import reset_fake_carrier
    reset_fake_carrier()
    yield
    os.environ.pop("SHIPPO_API_KEY", None)
    get_settings.cache_clear()
    reset_fake_carrier()

@pytest.fixture
def isolated_db():
    """Deliberately does NOT use importlib.reload(db_module) — repeated
    reloading of a SQLAlchemy declarative module within one pytest
    process leaves stale mapper configurations from prior reloads
    registered process-wide (each reload creates a new Base/registry,
    but old mapped classes aren't disposed), which trips SQLAlchemy's
    global configure_mappers() with a confusing, unrelated-looking error.
    Confirmed directly while adding this test file — even calling
    sqlalchemy.orm.clear_mappers() first doesn't safely fix it, since
    that unmaps classes OTHER already-imported modules (app.tools.wms,
    app.tools.idempotency) still hold direct references to.

    The actual fix: never recreate the ORM classes at all for test
    isolation — just point the EXISTING db_module.SessionLocal at a
    fresh SQLite file's engine. Same isolation guarantee (a clean,
    empty database per test), none of the reload fragility.
    """
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_shippo_{os.getpid()}_{id(object())}.db")
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

def test_shippo_gateway_raises_clearly_without_api_key():
    os.environ.pop("SHIPPO_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.tools.carrier import ShippoGateway
    with pytest.raises(RuntimeError, match="SHIPPO_API_KEY not configured"):
        ShippoGateway()

@respx.mock
def test_shippo_get_tracking_status_parses_real_response_shape(shippo_settings):
    from app.tools.carrier import ShippoGateway

    respx.get(f"{SHIPPO_BASE}/tracks/usps/9205590164917312751089").mock(
        return_value=httpx.Response(200, json={
            "carrier": "usps",
            "tracking_number": "9205590164917312751089",
            "tracking_status": {
                "object_id": "ce48ff3d52a34e91b77aa98370182624",
                "status": "DELIVERED",
                "status_details": "Your shipment has been delivered at the destination mailbox.",
                "status_date": "2023-07-23T13:03:00Z",
                "location": {"city": "Spotsylvania", "state": "VA", "zip": "22551", "country": "US"},
            },
            "tracking_history": [],
        })
    )

    gateway = ShippoGateway()
    result = gateway.get_tracking_status("9205590164917312751089")

    assert result["status"] == "delivered"
    assert result["tracking_number"] == "9205590164917312751089"

@respx.mock
def test_shippo_get_tracking_status_unknown_number_returns_unknown_not_crash(shippo_settings):
    from app.tools.carrier import ShippoGateway

    respx.get(f"{SHIPPO_BASE}/tracks/usps/NOT-A-REAL-CODE").mock(
        return_value=httpx.Response(404, json={"detail": "Not found."})
    )

    gateway = ShippoGateway()
    result = gateway.get_tracking_status("NOT-A-REAL-CODE")

    assert result["status"] == "unknown"

@respx.mock
def test_shippo_generate_return_label_parses_real_response_shape(shippo_settings, isolated_db):
    from app.tools.carrier import ShippoGateway

    respx.post(f"{SHIPPO_BASE}/shipments/").mock(return_value=httpx.Response(200, json={
        "object_id": "shp_fake123",
        "rates": [
            {"object_id": "rate_fake_cheap", "amount": "5.50", "provider": "USPS"},
            {"object_id": "rate_fake_expensive", "amount": "22.00", "provider": "FedEx"},
        ],
    }))
    respx.post(f"{SHIPPO_BASE}/transactions/").mock(return_value=httpx.Response(200, json={
        "object_id": "trans_fake123",
        "status": "SUCCESS",
        "tracking_number": "SHIPPO_TRANSACTION_DELIVERED",
        "label_url": "https://shippo-delivery.s3.amazonaws.com/fake_label.pdf",
    }))

    db = isolated_db.SessionLocal()
    gateway = ShippoGateway()
    result = gateway.generate_return_label(db, order_id="ORD-SHIPPO-1", idempotency_key="shippo-key-1")

    assert result["tracking_number"] == "SHIPPO_TRANSACTION_DELIVERED"
    assert "fake_label.pdf" in result["label_url"]
    assert result["_was_replayed"] is False
    db.close()

@respx.mock
def test_shippo_generate_return_label_picks_lowest_rate(shippo_settings, isolated_db):
    """Confirms the lowest-amount rate is selected, not just the first
    one in the list."""
    from app.tools.carrier import ShippoGateway

    respx.post(f"{SHIPPO_BASE}/shipments/").mock(return_value=httpx.Response(200, json={
        "object_id": "shp_fake456",
        "rates": [
            {"object_id": "rate_expensive", "amount": "22.00", "provider": "FedEx"},
            {"object_id": "rate_cheapest", "amount": "5.50", "provider": "USPS"},
            {"object_id": "rate_medium", "amount": "12.00", "provider": "UPS"},
        ],
    }))
    transaction_route = respx.post(f"{SHIPPO_BASE}/transactions/").mock(
        return_value=httpx.Response(200, json={
            "object_id": "trans_fake456", "status": "SUCCESS",
            "tracking_number": "TRACK456", "label_url": "https://example.com/label.pdf",
        })
    )

    db = isolated_db.SessionLocal()
    gateway = ShippoGateway()
    gateway.generate_return_label(db, order_id="ORD-SHIPPO-2", idempotency_key="shippo-key-2")

    import json
    sent_body = json.loads(transaction_route.calls[0].request.content)
    assert sent_body["rate"] == "rate_cheapest", "must select the lowest-amount rate, not just the first one"
    db.close()

@respx.mock
def test_shippo_generate_return_label_is_idempotent(shippo_settings, isolated_db):
    from app.tools.carrier import ShippoGateway

    shipment_route = respx.post(f"{SHIPPO_BASE}/shipments/").mock(return_value=httpx.Response(200, json={
        "object_id": "shp_fake789",
        "rates": [{"object_id": "rate_fake1", "amount": "7.50", "provider": "USPS"}],
    }))
    respx.post(f"{SHIPPO_BASE}/transactions/").mock(return_value=httpx.Response(200, json={
        "object_id": "trans_fake789", "status": "SUCCESS",
        "tracking_number": "TRACK789", "label_url": "https://example.com/label789.pdf",
    }))

    db = isolated_db.SessionLocal()
    gateway = ShippoGateway()
    r1 = gateway.generate_return_label(db, order_id="ORD-SHIPPO-3", idempotency_key="shippo-key-3")
    r2 = gateway.generate_return_label(db, order_id="ORD-SHIPPO-3", idempotency_key="shippo-key-3")

    assert r1["tracking_number"] == r2["tracking_number"]
    assert r2["_was_replayed"] is True
    assert shipment_route.call_count == 1, "a replayed call must NOT create a second real shipment"
    db.close()

@respx.mock
def test_shippo_generate_return_label_raises_on_failed_transaction(shippo_settings, isolated_db):
    """A transaction that doesn't succeed must surface as a clean error,
    not silently return a broken label."""
    from app.tools.carrier import ShippoGateway

    respx.post(f"{SHIPPO_BASE}/shipments/").mock(return_value=httpx.Response(200, json={
        "object_id": "shp_fake999",
        "rates": [{"object_id": "rate_fake1", "amount": "7.50", "provider": "USPS"}],
    }))
    respx.post(f"{SHIPPO_BASE}/transactions/").mock(return_value=httpx.Response(200, json={
        "object_id": "trans_fake999", "status": "ERROR",
        "messages": [{"source": "USPS", "code": "carrier_timeout", "text": "USPS API did not respond."}],
    }))

    db = isolated_db.SessionLocal()
    gateway = ShippoGateway()
    with pytest.raises(Exception):
        gateway.generate_return_label(db, order_id="ORD-SHIPPO-4", idempotency_key="shippo-key-4")
    db.close()

def test_get_carrier_gateway_returns_shippo_when_key_configured(shippo_settings):
    from app.tools.carrier import get_carrier_gateway, ShippoGateway
    gateway = get_carrier_gateway()
    assert isinstance(gateway, ShippoGateway)

def test_get_carrier_gateway_prefers_easypost_when_both_configured(shippo_settings):
    """EasyPost takes priority if both credentials happen to be set."""
    os.environ["EASYPOST_API_KEY"] = "EZTK_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.tools.carrier import get_carrier_gateway, EasyPostGateway
    gateway = get_carrier_gateway()
    assert isinstance(gateway, EasyPostGateway)

    os.environ.pop("EASYPOST_API_KEY", None)
    get_settings.cache_clear()
