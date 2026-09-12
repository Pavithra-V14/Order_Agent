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


def _diagnosis_system_prompt() -> str:
    """Shared between GroqClient and LiteLLMClient - this exact prompt
    text has several real, bug-fix-driven improvements baked in (e.g.
    the explicit "don't re-request already-fetched data" instruction,
    added after a real Groq model looped calling check_carrier 7 times
    in production). Extracted here so both clients use the identical,
    already-battle-tested text rather than risking drift between two
    copies."""
    return (
        "You are the Diagnosis Agent for an order-exception resolution system. Given the "
        "case context and findings gathered so far, decide the NEXT single action to take. "
        "Valid actions: check_order, check_payment, check_inventory, check_carrier, conclude. "
        "CRITICAL: never choose an action whose corresponding key already exists in "
        "findings_so_far (e.g. if findings_so_far already has a 'carrier' key, do NOT choose "
        "check_carrier again — that data has already been fetched once and will not change on "
        "a second identical call). If every check relevant to this case's context already has "
        "an entry in findings_so_far, you MUST choose 'conclude' — do not re-request a check "
        "just to double-check or confirm data you already have. "
        "Only choose 'conclude' once you have enough evidence to state root causes (or state "
        "'no_anomaly_detected: ...' if nothing is wrong). Respond ONLY with JSON: "
        '{"action": str, "reasoning": str, "root_causes": [str] or null}. '
        "Each root cause MUST start with one of these exact prefixes — this contract is "
        "required by the calling system, not a style preference: payment_issue:, "
        "inventory_issue:, carrier_issue:, no_anomaly_detected:, diagnosis_incomplete:, "
        "diagnosis_timeout:"
    )


def _fraud_system_prompt() -> str:
    """Shared between GroqClient and LiteLLMClient - see _diagnosis_system_prompt()'s docstring for why."""
    return (
        "You are the Fraud/Risk Agent for an order-exception resolution system. Score risk "
        "from 0.0-1.0 given the customer's risk profile and case context. IMPORTANT: weight "
        "prior fraud flags and return-reason-pattern consistency heavily; weight raw return "
        "COUNT only lightly — a high-volume but legitimate customer must NOT be penalized for "
        "return frequency alone. Respond ONLY with JSON: "
        '{"risk_score": float, "flag": bool, "reasons": [str]}.'
    )


class BaseLLMClient(ABC):
    @abstractmethod
    def plan_next_diagnosis_step(self, case_context: dict, findings_so_far: dict) -> DiagnosisStepPlan: ...

    @abstractmethod
    def assess_fraud_risk(self, case_context: dict, customer_risk_profile: dict) -> dict: ...


class GroqClient(BaseLLMClient):
    """Production router/classifier-tier implementation - Groq-hosted
    Llama 3.3 70B, per architecture doc 8.10. Real HTTP implementation
    (Groq's API is OpenAI-compatible chat completions with JSON mode) —
    not network-tested from this sandbox (no route to api.groq.com in
    the bash tool's allowed domains), but the request/response handling
    itself IS tested against a realistic mocked response shape via respx
    (see tests/test_groq_client.py), which is the strongest verification
    possible without real network access: it proves the parsing/retry
    logic is correct, leaving only "can this sandbox reach the internet"
    as the untested variable — not "is the code right."
    """

    def __init__(self):
        from app.core.config import get_settings
        settings = get_settings()
        self._api_key = settings.groq_api_key
        self._model = settings.router_model
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY not configured - see .env.example")
        import httpx
        self._client = httpx.Client(
            base_url="https://api.groq.com/openai/v1",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            timeout=30.0,
        )

    def _chat_json(self, system_prompt: str, user_prompt: str, max_retries: int = 2) -> dict:
        """Calls Groq's chat completions endpoint in JSON mode. Retries
        on transient failures (network error, malformed JSON response) —
        an LLM call is exactly the kind of thing that occasionally needs
        a retry, same reliability discipline as the tool layer (Layer 5)."""
        import json
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                resp = self._client.post("/chat/completions", json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.0,
                })
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"]
                return json.loads(content)
            except Exception as e:
                last_error = e
                continue
        raise RuntimeError(f"Groq API call failed after {max_retries + 1} attempts: {last_error}")

    def plan_next_diagnosis_step(self, case_context: dict, findings_so_far: dict) -> DiagnosisStepPlan:
        import json
        system_prompt = _diagnosis_system_prompt()
        user_prompt = json.dumps({"case_context": case_context, "findings_so_far": findings_so_far}, default=str)
        result = self._chat_json(system_prompt, user_prompt)
        return DiagnosisStepPlan(
            action=result["action"],
            reasoning=result.get("reasoning", ""),
            root_causes=result.get("root_causes"),
        )

    def assess_fraud_risk(self, case_context: dict, customer_risk_profile: dict) -> dict:
        import json
        system_prompt = _fraud_system_prompt()
        user_prompt = json.dumps(
            {"case_context": case_context, "customer_risk_profile": customer_risk_profile}, default=str
        )
        return self._chat_json(system_prompt, user_prompt)


class LiteLLMClient(BaseLLMClient):
    """The real LLM gateway - wraps Groq calls through litellm.Router
    rather than calling Groq's API directly (GroqClient, kept in this
    file but no longer used by get_llm_client() - see that function's
    docstring). Adds three real, verified capabilities GroqClient never
    had: genuine exponential backoff with jitter (litellm's own
    _calculate_retry_after, respecting real Retry-After response
    headers when Groq sends one), per-deployment rate limiting (rpm),
    and a cooldown mechanism (allowed_fails/cooldown_time) that
    temporarily stops routing to Groq after repeated failures - all
    three checked directly against litellm's actual source before
    building this, not assumed from documentation.

    ALSO wrapped in this project's own CircuitBreaker (app/core/
    circuit_breaker.py) - the SAME mechanism payment/carrier/wms
    already use - rather than relying solely on litellm's internal
    cooldown. This is deliberate: litellm's cooldown is real but
    invisible to this project's own observability (the /admin/alerts
    panel, the circuit_breaker_trip alert type) — wrapping it in the
    project's own breaker means an LLM outage shows up the exact same
    way every other dependency's outage does, not as a separate,
    unmonitored subsystem.
    """

    def __init__(self):
        from app.core.config import get_settings
        settings = get_settings()
        self._api_key = settings.groq_api_key
        self._model_name = settings.router_model
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY not configured - see .env.example")

        # Skips litellm's own remote model-cost-map fetch from GitHub
        # on first use (found directly: this added several real
        # seconds of network latency on every fresh instantiation,
        # purely for cost-tracking metadata this project doesn't use).
        # Real, documented flag - see litellm/litellm_core_utils/
        # get_model_cost_map.py's own module docstring.
        import os
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

        from litellm import Router
        litellm_model_name = f"groq/{self._model_name}"
        self._router = Router(
            model_list=[{
                "model_name": "diagnosis-router",
                "litellm_params": {
                    "model": litellm_model_name,
                    "api_key": self._api_key,
                    # A conservative default for Groq's free tier —
                    # real deployments on a paid tier should raise this
                    # to match their actual plan limits. Rate-limiting
                    # client-side (rather than only reacting to Groq's
                    # own 429s after the fact) is the actual point of
                    # setting this at all.
                    "rpm": 30,
                },
            }],
            num_retries=2,
            timeout=30.0,
            # allowed_fails/cooldown_time mirror this project's own
            # CircuitBreaker defaults (failure_threshold=3,
            # reset_timeout_seconds=30.0) for consistency, even though
            # the project's own breaker (below) is the primary
            # mechanism actually observed/alerted on.
            allowed_fails=3,
            cooldown_time=30,
        )
        self._litellm_model_name = litellm_model_name

    def _chat_json(self, system_prompt: str, user_prompt: str) -> dict:
        import json
        from app.core.circuit_breaker import get_circuit_breaker
        breaker = get_circuit_breaker("llm_groq", failure_threshold=3, reset_timeout_seconds=30.0)

        def _do_call():
            response = self._router.completion(
                model="diagnosis-router",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return response.choices[0].message.content

        content = breaker.call(_do_call)
        return json.loads(content)

    def plan_next_diagnosis_step(self, case_context: dict, findings_so_far: dict) -> DiagnosisStepPlan:
        import json
        user_prompt = json.dumps({"case_context": case_context, "findings_so_far": findings_so_far}, default=str)
        result = self._chat_json(_diagnosis_system_prompt(), user_prompt)
        return DiagnosisStepPlan(
            action=result["action"],
            reasoning=result.get("reasoning", ""),
            root_causes=result.get("root_causes"),
        )

    def assess_fraud_risk(self, case_context: dict, customer_risk_profile: dict) -> dict:
        import json
        user_prompt = json.dumps(
            {"case_context": case_context, "customer_risk_profile": customer_risk_profile}, default=str
        )
        return self._chat_json(_fraud_system_prompt(), user_prompt)


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
    """Auto-selects the real LiteLLMClient when GROQ_API_KEY is
    configured, falling back to FakeLLMClient otherwise — mirrors the
    same settings-driven pattern as get_cache() (Redis) and
    get_qdrant_client() (Qdrant Cloud): nothing crashes on a partial
    cloud configuration, the real implementation activates
    automatically once its credential is present.

    LiteLLMClient (routes through litellm.Router), not GroqClient
    (direct HTTP calls, kept in this file but no longer used here) —
    the real LLM gateway upgrade: genuine exponential backoff, per-
    deployment rate limiting, and a cooldown mechanism on top of
    litellm's own retry logic, plus this project's own CircuitBreaker
    wrapping the whole thing for consistent observability with every
    other external dependency (payment/carrier/wms). See
    LiteLLMClient's docstring for what was actually verified, not just
    assumed, about each of these three capabilities.
    """
    from app.core.config import get_settings
    settings = get_settings()
    if settings.groq_api_key:
        return LiteLLMClient()
    return FakeLLMClient()
