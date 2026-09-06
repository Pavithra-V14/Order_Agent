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
    """Production implementation — real EasyPost sandbox. Not callable in
    this sandbox (no network access). Uses EasyPost's actual REST API
    shape (label creation returns a tracking_code + label_url)."""

    def __init__(self):
        from app.core.config import get_settings
        settings = get_settings()
        self._api_key = getattr(settings, "easypost_api_key", None)
        # real implementation: import easypost; easypost.api_key = self._api_key

    def get_tracking_status(self, tracking_number: str) -> dict:
        raise NotImplementedError("Requires network access to api.easypost.com — see class docstring.")

    def generate_return_label(self, db: Session, order_id: str, idempotency_key: str) -> dict:
        raise NotImplementedError("Requires network access to api.easypost.com — see class docstring.")


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


def get_carrier_gateway() -> CarrierGateway:
    global _fake_carrier_singleton
    if _fake_carrier_singleton is None:
        _fake_carrier_singleton = FakeCarrierGateway()
    return _fake_carrier_singleton


def reset_fake_carrier() -> None:
    global _fake_carrier_singleton
    _fake_carrier_singleton = None


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
