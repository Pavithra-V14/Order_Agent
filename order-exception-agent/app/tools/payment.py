"""
Payment tool — architecture doc 8.4/Part 6 (Stripe test mode via MCP).
This sandbox has no network access to api.stripe.com (not in the bash
tool's allowed domains), so this ships a FakePaymentGateway that mirrors
the real `stripe` Python SDK's call shape exactly (same method names, same
idempotency_key parameter, same response shape) — swapping to
StripeGateway (real, using the already-installed `stripe` library) is a
one-line change in `get_payment_gateway()`, not a rewrite of any caller.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.tools.idempotency import with_idempotency


class PaymentGateway(ABC):
    @abstractmethod
    def get_transaction_status(self, payment_intent_id: str) -> dict: ...

    @abstractmethod
    def issue_refund(self, db: Session, payment_intent_id: str, amount_usd: float,
                      idempotency_key: str) -> dict: ...


class StripeGateway(PaymentGateway):
    """Production implementation — real Stripe test mode. Not callable in
    this sandbox (no network to api.stripe.com). Uses the already-installed
    `stripe` library exactly as it would be used against a live test-mode
    account; the only thing missing here is network access and a real
    STRIPE_API_KEY, both available in a normal deployment environment."""

    def __init__(self):
        settings = get_settings()
        import stripe
        stripe.api_key = getattr(settings, "stripe_api_key", None)
        self._stripe = stripe

    def get_transaction_status(self, payment_intent_id: str) -> dict:
        pi = self._stripe.PaymentIntent.retrieve(payment_intent_id)
        return {"id": pi.id, "status": pi.status, "amount": pi.amount / 100}

    def issue_refund(self, db: Session, payment_intent_id: str, amount_usd: float,
                      idempotency_key: str) -> dict:
        # Stripe's own idempotency_key param is the REAL safety net against
        # a duplicate refund on their side — see idempotency.py's docstring.
        refund = self._stripe.Refund.create(
            payment_intent=payment_intent_id,
            amount=int(round(amount_usd * 100)),
            idempotency_key=idempotency_key,
        )
        return {"id": refund.id, "status": refund.status, "amount": refund.amount / 100}


class FakePaymentGateway(PaymentGateway):
    """Sandbox-runnable substitute. Deterministic, in-memory transaction
    state keyed by payment_intent_id (seeded via `seed_transaction` for
    tests), real local idempotency enforcement via `with_idempotency` —
    genuinely functional, not a no-op mock."""

    def __init__(self):
        self._transactions: dict[str, dict] = {}
        self.refund_call_count = 0  # test hook: proves double-calls are actually prevented
        self._fail_next_n_execute_calls = 0  # chaos-test hook: simulate transient gateway failures

    def inject_transient_failures(self, n: int) -> None:
        """Chaos-test hook: the next N real execute attempts raise a
        simulated timeout instead of succeeding. Does not affect idempotent
        REPLAYS (a call whose idempotency_key already has a stored result
        never reaches _execute() at all, so injected failures only bite on
        genuinely new execution attempts, exactly like a real transient
        network failure would)."""
        self._fail_next_n_execute_calls = n

    def seed_transaction(self, payment_intent_id: str, amount_usd: float, status: str = "succeeded"):
        self._transactions[payment_intent_id] = {"id": payment_intent_id, "status": status, "amount": amount_usd}

    def get_transaction_status(self, payment_intent_id: str) -> dict:
        if payment_intent_id not in self._transactions:
            raise ValueError(f"No such payment_intent_id: {payment_intent_id}")
        return dict(self._transactions[payment_intent_id])

    def issue_refund(self, db: Session, payment_intent_id: str, amount_usd: float,
                      idempotency_key: str) -> dict:
        def _execute() -> dict:
            self.refund_call_count += 1  # only increments on a REAL execution, not a replay
            if self._fail_next_n_execute_calls > 0:
                self._fail_next_n_execute_calls -= 1
                raise TimeoutError("Simulated payment gateway timeout (chaos test fault injection)")
            if payment_intent_id not in self._transactions:
                raise ValueError(f"No such payment_intent_id: {payment_intent_id}")
            refund_id = f"re_fake_{uuid.uuid4().hex[:16]}"
            return {
                "id": refund_id,
                "status": "succeeded",
                "amount": amount_usd,
                "payment_intent": payment_intent_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        result, was_replayed = with_idempotency(
            db=db,
            idempotency_key=idempotency_key,
            tool_name="payment_refund",
            request_args={"payment_intent_id": payment_intent_id, "amount_usd": amount_usd},
            execute_fn=_execute,
        )
        result["_was_replayed"] = was_replayed  # visible in tests/audit trail, not sent to Stripe itself
        return result


_fake_gateway_singleton: FakePaymentGateway | None = None


def get_payment_gateway() -> PaymentGateway:
    """Swap point: return StripeGateway() here once running with real
    network access + STRIPE_API_KEY. Kept as a module-level singleton for
    the fake so its in-memory transaction/call-count state persists across
    calls within a process — required for the idempotency test to mean
    anything (a fresh instance per call would trivially "pass" by having
    no memory of the first call at all)."""
    global _fake_gateway_singleton
    if _fake_gateway_singleton is None:
        _fake_gateway_singleton = FakePaymentGateway()
    return _fake_gateway_singleton


def reset_fake_gateway() -> None:
    """Test helper — forces a fresh FakePaymentGateway (fresh transaction
    state + call counter) between test cases."""
    global _fake_gateway_singleton
    _fake_gateway_singleton = None
