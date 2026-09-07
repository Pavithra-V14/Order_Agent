"""
Carrier tool — architecture doc 8.4/Part 6 (EasyPost/Shippo sandbox via
MCP). No network access to their APIs from this sandbox, so this ships a
FakeCarrierGateway behind the same interface a real EasyPost/Shippo
wrapper would use. Swap point: `get_carrier_gateway()`.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.tools.idempotency import with_idempotency


class CarrierGateway(ABC):
    @abstractmethod
    def get_tracking_status(self, tracking_number: str) -> dict: ...

    @abstractmethod
    def generate_return_label(self, db: Session, order_id: str, idempotency_key: str) -> dict: ...


class EasyPostGateway(CarrierGateway):
    """Production implementation — real EasyPost API (cloud, no Docker; a
    free-tier/test-mode API key from easypost.com works immediately).

    Uses raw httpx REST calls against EasyPost's documented API contract
    (POST /v2/trackers, /v2/shipments, /v2/shipments/{id}/buy — Bearer
    auth, resource-name-wrapped request bodies) rather than the
    `easypost` SDK package. This is a deliberate choice, found necessary
    during testing: the SDK uses `requests` internally, not `httpx`, so
    `respx` (this project's mocking tool for every other cloud
    integration — Groq, Mistral) silently can't intercept it, and a
    "mocked" SDK test would actually hit the real network undetected
    (confirmed directly: it hit this sandbox's network egress proxy and
    failed with an allowlist error instead of using the mock). Raw httpx
    keeps this integration testable the same way as every other one in
    this project, and removes a dependency.

    Not network-tested from this sandbox (no route to api.easypost.com
    in the bash tool's allowed domains) — request/response handling is
    tested against realistic mocked EasyPost response shapes instead
    (tests/test_easypost_gateway.py).

    Honest limitation: `generate_return_label` needs real to/from
    addresses and a parcel to create an EasyPost shipment — this
    project's order model doesn't yet carry shipping-address data, so
    this uses clearly-marked placeholder addresses. Wiring real customer/
    warehouse addresses through from `MockOrderRecord` is a small,
    separate follow-up once that data exists in the order model.
    """

    def __init__(self):
        from app.core.config import get_settings
        settings = get_settings()
        self._api_key = settings.easypost_api_key
        if not self._api_key:
            raise RuntimeError("EASYPOST_API_KEY not configured - see .env.example")
        import httpx
        self._client = httpx.Client(
            base_url="https://api.easypost.com/v2",
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=30.0,
        )

    def get_tracking_status(self, tracking_number: str) -> dict:
        """EasyPost dedupes trackers by (tracking_code, carrier) — POSTing
        a tracking_code that already has a tracker attached to this
        account returns the EXISTING tracker (with its current status),
        rather than creating a duplicate. This is the correct
        EasyPost-idiomatic way to look up status, not a workaround."""
        try:
            resp = self._client.post("/trackers", json={"tracker": {"tracking_code": tracking_number}})
            resp.raise_for_status()
            data = resp.json()
            return {"tracking_number": tracking_number, "status": data["status"]}
        except Exception as e:
            # Matches FakeCarrierGateway's contract: an unrecognized/
            # malformed tracking number reports status="unknown" rather
            # than propagating a raw HTTP exception through the
            # Diagnosis Agent's tool-call loop.
            return {"tracking_number": tracking_number, "status": "unknown", "error": str(e)}

    def generate_return_label(self, db: Session, order_id: str, idempotency_key: str) -> dict:
        def _execute() -> dict:
            # Placeholder addresses — see class docstring's honest
            # limitation note.
            shipment_resp = self._client.post("/shipments", json={"shipment": {
                "to_address": {
                    "name": "Returns Processing", "street1": "417 Montgomery St",
                    "city": "San Francisco", "state": "CA", "zip": "94104", "country": "US",
                },
                "from_address": {
                    "name": "Customer", "street1": "179 N Harbor Dr",
                    "city": "Redondo Beach", "state": "CA", "zip": "90277", "country": "US",
                },
                "parcel": {"length": 10.0, "width": 8.0, "height": 4.0, "weight": 16.0},
            }})
            shipment_resp.raise_for_status()
            shipment = shipment_resp.json()

            rates = shipment.get("rates") or []
            if not rates:
                raise RuntimeError(f"EasyPost returned no rates for shipment {shipment.get('id')}")
            lowest_rate = min(rates, key=lambda r: float(r["rate"]))

            buy_resp = self._client.post(
                f"/shipments/{shipment['id']}/buy", json={"rate": {"id": lowest_rate["id"]}}
            )
            buy_resp.raise_for_status()
            bought = buy_resp.json()

            return {
                "tracking_number": bought["tracking_code"],
                "label_url": bought["postage_label"]["label_url"],
                "order_id": order_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        result, was_replayed = with_idempotency(
            db=db, idempotency_key=idempotency_key, tool_name="carrier_generate_label",
            request_args={"order_id": order_id}, execute_fn=_execute,
        )
        result["_was_replayed"] = was_replayed
        return result


class ShippoGateway(CarrierGateway):
    """Alternative production implementation — real Shippo API (cloud, no
    Docker; a free test-mode API key from goshippo.com works immediately,
    with no card required for the direct developer signup — unlike
    EasyPost's 2026 onboarding, which now gates API key access behind a
    payment method on file).

    Raw httpx REST calls against Shippo's documented API — same pattern
    as EasyPostGateway, for the same reason: consistency with this
    project's respx-based test strategy, and one fewer vendor SDK
    dependency. Shippo's auth scheme is `Authorization: ShippoToken
    <token>`, not `Bearer` — confirmed directly from their docs before
    implementing this, not assumed from the OAuth-style convention most
    other APIs in this project use.

    Not network-tested from this sandbox (no route to api.goshippo.com in
    the bash tool's allowed domains) — request/response handling is
    tested against realistic mocked Shippo response shapes instead
    (tests/test_shippo_gateway.py), same strategy as every other cloud
    integration in this project.

    Honest limitations, same as EasyPostGateway: `generate_return_label`
    uses placeholder to/from addresses (this project's order model
    doesn't carry real shipping addresses yet). Additionally,
    `get_tracking_status` defaults to carrier="usps" since Shippo's
    tracking API requires knowing the carrier explicitly (EasyPost's
    tracker lookup can work from the tracking number alone) — this
    project's CarrierGateway interface only takes a tracking_number, so a
    real deployment tracking non-USPS carriers would need to extend the
    interface to pass the carrier through, not just this gateway.
    """

    def __init__(self):
        from app.core.config import get_settings
        settings = get_settings()
        self._api_key = settings.shippo_api_key
        if not self._api_key:
            raise RuntimeError("SHIPPO_API_KEY not configured - see .env.example")
        import httpx
        self._client = httpx.Client(
            base_url="https://api.goshippo.com",
            headers={"Authorization": f"ShippoToken {self._api_key}"},
            timeout=30.0,
        )

    def get_tracking_status(self, tracking_number: str, carrier: str = "usps") -> dict:
        try:
            resp = self._client.get(f"/tracks/{carrier}/{tracking_number}")
            resp.raise_for_status()
            data = resp.json()
            status = data.get("tracking_status", {}).get("status", "unknown")
            return {"tracking_number": tracking_number, "status": status.lower() if status else "unknown"}
        except Exception as e:
            # Matches FakeCarrierGateway's contract: an unrecognized/
            # malformed tracking number reports status="unknown" rather
            # than propagating a raw HTTP exception through the
            # Diagnosis Agent's tool-call loop.
            return {"tracking_number": tracking_number, "status": "unknown", "error": str(e)}

    def generate_return_label(self, db: Session, order_id: str, idempotency_key: str) -> dict:
        def _execute() -> dict:
            # Placeholder addresses — see class docstring's honest limitation note.
            shipment_resp = self._client.post("/shipments/", json={
                "address_from": {
                    "name": "Customer", "street1": "179 N Harbor Dr",
                    "city": "Redondo Beach", "state": "CA", "zip": "90277", "country": "US",
                    "email": "returns@example.com", "phone": "+1 555 341 9393",
                },
                "address_to": {
                    "name": "Returns Processing", "street1": "417 Montgomery St",
                    "city": "San Francisco", "state": "CA", "zip": "94104", "country": "US",
                    "email": "warehouse@example.com", "phone": "+1 555 341 9393",
                },
                "parcels": [{
                    "length": "10", "width": "8", "height": "4", "distance_unit": "in",
                    "weight": "16", "mass_unit": "oz",
                }],
                "async": False,
            })
            shipment_resp.raise_for_status()
            shipment = shipment_resp.json()

            rates = shipment.get("rates") or []
            if not rates:
                raise RuntimeError(f"Shippo returned no rates for shipment {shipment.get('object_id')}")
            lowest_rate = min(rates, key=lambda r: float(r["amount"]))

            transaction_resp = self._client.post("/transactions/", json={
                "rate": lowest_rate["object_id"], "async": False, "label_file_type": "PDF",
            })
            transaction_resp.raise_for_status()
            transaction = transaction_resp.json()

            if transaction.get("status") != "SUCCESS":
                raise RuntimeError(f"Shippo transaction did not succeed: {transaction.get('messages')}")

            return {
                "tracking_number": transaction["tracking_number"],
                "label_url": transaction["label_url"],
                "order_id": order_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        result, was_replayed = with_idempotency(
            db=db, idempotency_key=idempotency_key, tool_name="carrier_generate_label",
            request_args={"order_id": order_id}, execute_fn=_execute,
        )
        result["_was_replayed"] = was_replayed
        return result


class FakeCarrierGateway(CarrierGateway):
    """Sandbox-runnable substitute. Deterministic tracking-status
    progression, real local idempotency enforcement for label generation
    (edge case: a retried label-generation call must not produce two
    labels / two postage charges for the same order)."""

    def __init__(self):
        self._tracking: dict[str, str] = {}  # tracking_number -> status
        self.label_call_count = 0

    def seed_tracking(self, tracking_number: str, status: str):
        self._tracking[tracking_number] = status

    def get_tracking_status(self, tracking_number: str) -> dict:
        status = self._tracking.get(tracking_number, "unknown")
        return {"tracking_number": tracking_number, "status": status}

    def generate_return_label(self, db: Session, order_id: str, idempotency_key: str) -> dict:
        def _execute() -> dict:
            self.label_call_count += 1
            tracking_number = f"TRK{uuid.uuid4().hex[:12].upper()}"
            self._tracking[tracking_number] = "label_created"
            return {
                "tracking_number": tracking_number,
                "label_url": f"https://fake-carrier.local/labels/{tracking_number}.pdf",
                "order_id": order_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        result, was_replayed = with_idempotency(
            db=db,
            idempotency_key=idempotency_key,
            tool_name="carrier_generate_label",
            request_args={"order_id": order_id},
            execute_fn=_execute,
        )
        result["_was_replayed"] = was_replayed
        return result


_fake_carrier_singleton: FakeCarrierGateway | None = None


_easypost_singleton: EasyPostGateway | None = None
_shippo_singleton: ShippoGateway | None = None


def get_carrier_gateway() -> CarrierGateway:
    """Auto-selects a real carrier gateway based on which credential is
    configured — EasyPost takes priority if BOTH are set (pick one in
    practice), falling back to FakeCarrierGateway if neither is. Same
    settings-driven pattern as get_llm_client()/get_embedder()."""
    from app.core.config import get_settings
    settings = get_settings()

    if settings.easypost_api_key:
        global _easypost_singleton
        if _easypost_singleton is None:
            _easypost_singleton = EasyPostGateway()
        return _easypost_singleton

    if settings.shippo_api_key:
        global _shippo_singleton
        if _shippo_singleton is None:
            _shippo_singleton = ShippoGateway()
        return _shippo_singleton

    global _fake_carrier_singleton
    if _fake_carrier_singleton is None:
        _fake_carrier_singleton = FakeCarrierGateway()
    return _fake_carrier_singleton


def reset_fake_carrier() -> None:
    global _fake_carrier_singleton, _easypost_singleton, _shippo_singleton
    _fake_carrier_singleton = None
    _easypost_singleton = None
    _shippo_singleton = None


def handle_carrier_status_webhook(tracking_number: str, new_status: str) -> dict:
    """Simulates the inbound carrier-status webhook (Phase 12 will wire
    this to a real FastAPI POST /webhooks/carrier route). Updates the
    fake gateway's tracking state AND invalidates the tool-response cache
    together, same reasoning as wms.handle_inventory_update_webhook."""
    gateway = get_carrier_gateway()
    gateway.seed_tracking(tracking_number, new_status)

    from app.cache.tool_cache import invalidate_tracking_cache
    invalidated = invalidate_tracking_cache(tracking_number)

    return {"tracking_number": tracking_number, "new_status": new_status, "cache_invalidated": invalidated}
