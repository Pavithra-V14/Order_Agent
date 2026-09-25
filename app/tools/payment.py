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

    def get_refund(self, refund_id: str) -> dict | None:
        """Independent lookup of a refund at the provider, used by the
        Verification Agent. None means this gateway can't look refunds up
        (the fake gateway verifies from its local idempotency record)."""
        return None

    def seed_transaction(self, payment_intent_id: str, amount_usd: float, status: str = "succeeded") -> None:
        """No-op on the base class, overridden with real behavior only by
        FakePaymentGateway. A real gateway's transaction state is
        determined by the real provider (Stripe), not something this
        code can declare into existence — calling this on a real
        StripeGateway is a no-op with a clear explanation rather than a
        crash, so demo/seed scripts written against the fake gateway
        don't break the moment a real STRIPE_API_KEY gets configured.
        Found necessary directly: get_payment_gateway() now auto-selects
        StripeGateway when a key is present, and every seed/demo script
        in this project calls seed_transaction() unconditionally.
        """
        from app.core.console_log import log_warning
        log_warning(
            "payment",
            f"seed_transaction() called on {type(self).__name__} — this is a no-op on real gateways, "
            f"since a real payment's status is determined by the real provider, not declared by this code. "
            f"Use scripts/create_real_stripe_test_payment.py to set up a genuinely refundable payment instead.",
        )


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

    def get_refund(self, refund_id: str) -> dict | None:
        refund = self._stripe.Refund.retrieve(refund_id)
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
_stripe_singleton: StripeGateway | None = None


def get_payment_gateway() -> PaymentGateway:
    """Auto-selects StripeGateway when STRIPE_API_KEY is configured,
    falling back to FakePaymentGateway otherwise — same settings-driven
    pattern as get_llm_client()/get_embedder()/get_carrier_gateway().

    Found and fixed a real gap here: StripeGateway was fully written and
    tested for construction, but this factory function NEVER actually
    checked for STRIPE_API_KEY at all — it unconditionally returned the
    fake gateway regardless of what was configured. Every refund in
    every environment running this project, including ones with a real
    Stripe key set, was silently using the fake gateway the whole time.
    """
    from app.core.config import get_settings
    settings = get_settings()

    if settings.stripe_api_key:
        global _stripe_singleton
        if _stripe_singleton is None:
            _stripe_singleton = StripeGateway()
        return _stripe_singleton

    global _fake_gateway_singleton
    if _fake_gateway_singleton is None:
        _fake_gateway_singleton = FakePaymentGateway()
    return _fake_gateway_singleton


def reset_fake_gateway() -> None:
    """Test helper — forces a fresh FakePaymentGateway (fresh transaction
    state + call counter) between test cases."""
    global _fake_gateway_singleton, _stripe_singleton
    _fake_gateway_singleton = None
    _stripe_singleton = None
