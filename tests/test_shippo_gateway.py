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
def test_shippo_magic_test_tracking_token_uses_carrier_shippo(shippo_settings):
    """THE regression test for a real mistake, corrected directly: an
    earlier claim that 'no real carrier API lets you simulate tracking
    status on demand' was wrong — Shippo genuinely documents exactly
    this, via carrier='shippo' (their own reserved test-mode token, not
    a real carrier) combined with a magic tracking number like
    SHIPPO_DELIVERED. This proves get_tracking_status() actually
    threads the carrier parameter through to the real request URL —
    the previously-missing piece, since the parameter existed on the
    method already but nothing ever passed anything but the default
    'usps' through it in practice."""
    from app.tools.carrier import ShippoGateway

    route = respx.get(f"{SHIPPO_BASE}/tracks/shippo/SHIPPO_DELIVERED").mock(
        return_value=httpx.Response(200, json={
            "carrier": "shippo",
            "tracking_number": "SHIPPO_DELIVERED",
            "tracking_status": {"status": "DELIVERED"},
            "tracking_history": [],
        })
    )

    gateway = ShippoGateway()
    result = gateway.get_tracking_status("SHIPPO_DELIVERED", carrier="shippo")

    assert route.called, "must actually request the /tracks/shippo/... URL, not silently default to usps"
    assert result["status"] == "delivered"


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


@respx.mock
def test_shippo_falls_back_to_next_carrier_on_registration_error(shippo_settings, isolated_db):
    """THE regression test for a real, reproducible production failure:
    the cheapest rate happened to be UPS, which wasn't activated in the
    account's real Shippo dashboard, producing a genuine
    'ups_registration_error'. This is a carrier-ACTIVATION problem, not
    a transient one - blindly retrying the same rate fails identically
    forever. This proves the gateway now tries the next-cheapest rate
    from a DIFFERENT carrier automatically instead of failing outright,
    the same thing a person manually comparing rates would naturally do."""
    from app.tools.carrier import ShippoGateway

    respx.post(f"{SHIPPO_BASE}/shipments/").mock(return_value=httpx.Response(200, json={
        "object_id": "shp_fallback_test",
        "rates": [
            {"object_id": "rate_ups_cheap", "amount": "5.00", "provider": "UPS"},
            {"object_id": "rate_usps_pricier", "amount": "7.50", "provider": "USPS"},
        ],
    }))
    respx.post(f"{SHIPPO_BASE}/transactions/").mock(side_effect=[
        httpx.Response(200, json={
            "object_id": "trans_ups_fail", "status": "ERROR",
            "messages": [{"source": "UPS", "code": "ups_registration_error",
                          "text": "The UPS account is not yet registered."}],
        }),
        httpx.Response(200, json={
            "object_id": "trans_usps_success", "status": "SUCCESS",
            "tracking_number": "TRACKFALLBACK123", "label_url": "https://example.com/fallback_label.pdf",
        }),
    ])

    db = isolated_db.SessionLocal()
    gateway = ShippoGateway()
    result = gateway.generate_return_label(db, order_id="ORD-FALLBACK-TEST", idempotency_key="fallback-key-1")

    assert result["tracking_number"] == "TRACKFALLBACK123", (
        "must succeed using the SECOND (USPS) rate after the first (UPS) rate failed with a "
        "registration error, not raise immediately on the first failure"
    )
    db.close()


def test_shippo_raises_clean_error_when_every_carrier_is_unregistered(shippo_settings, isolated_db):
    """If literally every rate fails with a registration error (nothing
    activated at all), this must raise one clear, actionable error -
    not silently return a broken result or loop forever."""
    from app.tools.carrier import ShippoGateway

    with respx.mock:
        respx.post(f"{SHIPPO_BASE}/shipments/").mock(return_value=httpx.Response(200, json={
            "object_id": "shp_all_fail",
            "rates": [
                {"object_id": "rate_ups", "amount": "5.00", "provider": "UPS"},
                {"object_id": "rate_fedex", "amount": "6.00", "provider": "FedEx"},
            ],
        }))
        respx.post(f"{SHIPPO_BASE}/transactions/").mock(side_effect=[
            httpx.Response(200, json={
                "object_id": "t1", "status": "ERROR",
                "messages": [{"source": "UPS", "code": "ups_registration_error", "text": "not registered"}],
            }),
            httpx.Response(200, json={
                "object_id": "t2", "status": "ERROR",
                "messages": [{"source": "FedEx", "code": "fedex_registration_error", "text": "not registered"}],
            }),
        ])

        db = isolated_db.SessionLocal()
        gateway = ShippoGateway()
        with pytest.raises(RuntimeError, match="No activated carrier"):
            gateway.generate_return_label(db, order_id="ORD-ALL-FAIL", idempotency_key="all-fail-key")
        db.close()

@respx.mock
def test_shippo_generate_return_label_includes_email_and_phone_on_addresses(shippo_settings, isolated_db):
    """THE regression test for a real production error found by actually
    running against Shippo's live API: 'Seller info missing email or
    phone. Seller email and phone number required for USPS.' The
    address payloads previously had no email/phone fields at all —
    Shippo's mocked test responses never caught this because the mock
    doesn't validate the REQUEST the way Shippo's real API does; only
    running against the real backend surfaced it."""
    from app.tools.carrier import ShippoGateway

    shipment_route = respx.post(f"{SHIPPO_BASE}/shipments/").mock(return_value=httpx.Response(200, json={
        "object_id": "shp_fake_email_check",
        "rates": [{"object_id": "rate_fake1", "amount": "7.50", "provider": "USPS"}],
    }))
    respx.post(f"{SHIPPO_BASE}/transactions/").mock(return_value=httpx.Response(200, json={
        "object_id": "trans_fake_email_check", "status": "SUCCESS",
        "tracking_number": "TRACKEMAILCHECK", "label_url": "https://example.com/label.pdf",
    }))

    db = isolated_db.SessionLocal()
    gateway = ShippoGateway()
    gateway.generate_return_label(db, order_id="ORD-EMAIL-CHECK", idempotency_key="email-check-key")

    import json
    sent_body = json.loads(shipment_route.calls[0].request.content)
    assert sent_body["address_from"].get("email"), "address_from must include an email — USPS requires it"
    assert sent_body["address_from"].get("phone"), "address_from must include a phone — USPS requires it"
    assert sent_body["address_to"].get("email"), "address_to must include an email"
    assert sent_body["address_to"].get("phone"), "address_to must include a phone"
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
