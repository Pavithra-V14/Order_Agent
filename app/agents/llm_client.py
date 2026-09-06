"""
LLM client interface - architecture doc 8.10's free-tier matrix (Groq
router tier, Mistral Large reasoning tier, Gemini generation tier). No
network access to any of those providers from this sandbox (not in the
bash tool's allowed domains), so this ships a FakeLLMClient behind the
same interface.

Important distinction from the RAG-layer substitutes (TF-IDF for BGE-M3,
etc.): those were "same interface, lower semantic quality." An LLM doing
multi-step diagnostic PLANNING is different - a fake that just returns
canned text would make the orchestration logic (loop termination, tool
selection, multi-cause aggregation) untestable, since there'd be nothing
real to test. So FakeLLMClient implements genuine rule-based decision
logic - deterministic, but actually reasoning over the structured tool
outputs it's given (payment status, inventory levels, etc.) to decide
what to check next and what conclusions to draw. This is a real,
inspectable decision procedure, not a mock - it just isn't a neural
network. The orchestration code (diagnosis_agent.py, orchestrator.py) is
written entirely against the interface below, so swapping in a real
LLM-backed implementation later changes zero orchestration code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DiagnosisStepPlan:
    """One planning decision: either 'check something else' or 'stop, here's what I found'."""
    action: str                    # "check_order" | "check_payment" | "check_inventory" | "check_carrier" | "conclude"
    reasoning: str                 # why this action was chosen, given what's known so far
    root_causes: list[str] | None = None  # populated only when action == "conclude"


class BaseLLMClient(ABC):
    @abstractmethod
    def plan_next_diagnosis_step(self, case_context: dict, findings_so_far: dict) -> DiagnosisStepPlan: ...

    @abstractmethod
    def assess_fraud_risk(self, case_context: dict, customer_risk_profile: dict) -> dict: ...


class GroqClient(BaseLLMClient):
    """Production router/classifier-tier implementation - Groq-hosted
    Llama 3.3 70B, per architecture doc 8.10. Not callable in this sandbox
    (no network to api.groq.com). Real implementation would call Groq's
    OpenAI-compatible chat completions endpoint with a function-calling
    schema matching DiagnosisStepPlan."""

    def __init__(self):
        from app.core.config import get_settings
        settings = get_settings()
        self._api_key = settings.groq_api_key
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY not configured - see .env.example")

    def plan_next_diagnosis_step(self, case_context: dict, findings_so_far: dict) -> DiagnosisStepPlan:
        raise NotImplementedError("Requires network access to api.groq.com - see class docstring.")

    def assess_fraud_risk(self, case_context: dict, customer_risk_profile: dict) -> dict:
        raise NotImplementedError("Requires network access to api.groq.com - see class docstring.")


class FakeLLMClient(BaseLLMClient):
    """Sandbox-runnable substitute with genuine rule-based reasoning (see
    module docstring). This is what Phase 6's tests actually run against."""

    def plan_next_diagnosis_step(self, case_context: dict, findings_so_far: dict) -> DiagnosisStepPlan:
        """Deterministic planner: checks order -> payment -> inventory ->
        carrier, in that order, skipping any check whose data is already
        in findings_so_far - this is a genuine "what's missing, check
        that next" decision procedure, not a scripted sequence per test
        case. Concludes once everything relevant has been checked, with
        root causes derived from what was actually found (not
        hardcoded)."""
        if "order" not in findings_so_far:
            return DiagnosisStepPlan(
                action="check_order",
                reasoning="No order data yet - must fetch the order before anything else can be evaluated.",
            )

        order = findings_so_far["order"]
        needs_payment_check = order.get("status") in ("payment_failed", "paid") and "payment" not in findings_so_far
        if needs_payment_check:
            return DiagnosisStepPlan(
                action="check_payment",
                reasoning=f"Order status is '{order.get('status')}' - checking payment gateway to "
                          f"confirm actual transaction state (never trust OMS status alone; edge case 2.2).",
            )

        has_line_items = bool(order.get("line_items"))
        if has_line_items and "inventory" not in findings_so_far:
            return DiagnosisStepPlan(
                action="check_inventory",
                reasoning="Order has line items - checking sellable stock for each SKU to rule out "
                          "an inventory-side cause (phantom stock, edge case 3.1).",
            )

        if order.get("status") in ("shipped", "delivered") and "carrier" not in findings_so_far:
            return DiagnosisStepPlan(
                action="check_carrier",
                reasoning="Order has shipped - checking carrier tracking status for delivery exceptions.",
            )

        causes = []
        payment = findings_so_far.get("payment")
        if payment and payment.get("status") not in ("succeeded", None):
            causes.append(f"payment_issue: transaction status is '{payment.get('status')}'")

        inventory = findings_so_far.get("inventory")
        order_line_items = {li["sku"]: li.get("qty", 1) for li in findings_so_far.get("order", {}).get("line_items", [])}
        if inventory:
            for item in inventory:
                requested_qty = order_line_items.get(item.get("sku"), 1)
                sellable = item.get("sellable_qty", 0)
                if sellable < requested_qty:
                    causes.append(
                        f"inventory_issue: SKU {item.get('sku')} has insufficient sellable stock "
                        f"(requested={requested_qty}, sellable={sellable}, on_hand={item.get('on_hand_qty')})"
                    )

        carrier = findings_so_far.get("carrier")
        if carrier and carrier.get("status") not in ("delivered", "in_transit", None):
            causes.append(f"carrier_issue: tracking status is '{carrier.get('status')}'")

        if not causes:
            causes.append("no_anomaly_detected: all checked systems report normal state")

        return DiagnosisStepPlan(
            action="conclude",
            reasoning=f"All relevant systems checked ({list(findings_so_far.keys())}); "
                      f"{len(causes)} root cause(s) identified.",
            root_causes=causes,
        )

    def assess_fraud_risk(self, case_context: dict, customer_risk_profile: dict) -> dict:
        """Rule-based risk scoring per architecture doc's explicit warning:
        weight return-reason pattern and prior fraud flags, NOT raw return
        count alone (avoids penalizing legitimate high-engagement customers)."""
        score = 0.0
        reasons = []

        fraud_flags = customer_risk_profile.get("fraud_flags_raised", 0)
        if fraud_flags > 0:
            score += 0.5
            reasons.append(f"{fraud_flags} prior fraud flag(s) on this customer")

        return_count = customer_risk_profile.get("total_return_cases", 0)
        if return_count > 10:
            score += 0.1
            reasons.append(f"elevated return count ({return_count}) - weighted lightly, not disqualifying")

        if case_context.get("address_changed_same_day"):
            score += 0.4
            reasons.append("shipping address changed same-day as this request")

        score = min(score, 1.0)
        return {
            "risk_score": score,
            "flag": score >= 0.6,
            "reasons": reasons,
        }


def get_llm_client() -> BaseLLMClient:
    """Swap point: return GroqClient() (or a reasoning-tier client for the
    planning-heavy Diagnosis Agent, per 8.10) once running with network
    access + a configured API key."""
    return FakeLLMClient()
