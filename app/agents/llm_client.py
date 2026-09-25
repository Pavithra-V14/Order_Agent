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

import contextvars
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Bump the version whenever a prompt's text changes, so traces and evals
# can attribute a behaviour change to the prompt that caused it.
PROMPT_VERSIONS = {"diagnosis": "diagnosis@4", "fraud": "fraud@1", "summary": "summary@1"}

# The case the current LLM call belongs to. Set by the orchestrator nodes;
# when unset (a direct call outside a case) no span is recorded.
_llm_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_trace_id", default=None)


def set_llm_trace(trace_id: str | None):
    return _llm_trace_id.set(trace_id)


def reset_llm_trace(token) -> None:
    _llm_trace_id.reset(token)


def _record_llm_span(purpose: str, model: str | None, user_prompt: str, content: str | None,
                     error: str | None, latency_ms: int, usage) -> None:
    """One trace span per real LLM call: which model answered, under which
    prompt version, with what tokens and latency, and whether the output
    parsed. Previously LLM calls were the one step in a case with no span
    at all. Never raises - tracing must not break the call it observes."""
    trace_id = _llm_trace_id.get()
    if trace_id is None:
        return
    try:
        from app.core.db import SessionLocal
        from app.core.tracing import record_span
        db = SessionLocal()
        try:
            output = {"content": (content or "")[:4000]}
            if error:
                output["error"] = error[:500]
            record_span(db, trace_id=trace_id, agent_or_tool_name=f"llm.{purpose}",
                        input_data={"prompt_version": PROMPT_VERSIONS.get(purpose), "user_prompt": user_prompt[:4000]},
                        output_data=output,
                        metadata={"model": model, "prompt_version": PROMPT_VERSIONS.get(purpose),
                                  "latency_ms": latency_ms, "ok": error is None,
                                  "prompt_tokens": getattr(usage, "prompt_tokens", None),
                                  "completion_tokens": getattr(usage, "completion_tokens", None)})
        finally:
            db.close()
    except Exception as e:
        import logging
        logging.getLogger("llm_client").warning("LLM span recording failed (non-fatal): %s", e)


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
        "case_context.exception_type (when present) says why the case was opened - payment, "
        "return or carrier - so prioritise the checks relevant to it. "
        "Valid actions: check_order, check_payment, check_inventory, check_carrier, conclude. "
        "CRITICAL: never choose an action whose corresponding key already exists in "
        "findings_so_far (e.g. if findings_so_far already has a 'carrier' key, do NOT choose "
        "check_carrier again — that data has already been fetched once and will not change on "
        "a second identical call). A finding marked \"unavailable\": true means no data source "
        "exists for that check on this order - it is NOT evidence of a fault and must never be "
        "reported as a root cause. If every check relevant to this case's context already has "
        "an entry in findings_so_far, you MUST choose 'conclude' — do not re-request a check "
        "just to double-check or confirm data you already have. "
        "Only choose 'conclude' once you have enough evidence to state root causes (or state "
        "'no_anomaly_detected: ...' if nothing is wrong). Before concluding no_anomaly_detected, "
        "verify ALL of: the payment status is 'succeeded' (or payment is unavailable); every "
        "inventory item has \"sufficient\": true; the carrier status (if any) is not a fault such "
        "as lost, damaged or returned_to_sender. If any of these fails, report the matching issue "
        "(payment_issue / inventory_issue / carrier_issue) instead. Respond ONLY with JSON: "
        '{"action": str, "reasoning": str, "root_causes": [str] or null}. '
        "Each root cause MUST start with one of these exact prefixes — this contract is "
        "required by the calling system, not a style preference: payment_issue:, "
        "inventory_issue:, carrier_issue:, no_anomaly_detected:, diagnosis_incomplete:, "
        "diagnosis_timeout:"
    )


def _summary_fold_system_prompt() -> str:
    """Stage 2 memory upgrade - the Summary Buffer's real summarization
    call (app/memory/summary_buffer.py), replacing what was previously a
    deterministic string-concatenation stub. Shared between GroqClient
    and LiteLLMClient, same reasoning as _diagnosis_system_prompt()'s
    docstring for why a single, already-battle-tested prompt is used by
    both rather than two copies that could drift."""
    return (
        "You maintain a running summary of an order-exception diagnosis in progress. "
        "You are given the EXISTING summary (may be empty, on the very first fold) and ONE "
        "new diagnostic step that just aged out of the recent-items window. Fold the new "
        "step into the summary. Keep it SHORT (1-3 sentences) and preserve concrete facts "
        "(specific findings, root causes, amounts) - drop redundant phrasing rather than "
        "restating what the existing summary already says. This summary may later be read "
        "by a human reviewing the case, so it must remain a faithful, readable account, not "
        "a compressed code. Respond ONLY with JSON: {\"summary\": str}."
    )


def _fraud_system_prompt() -> str:
    """Shared between GroqClient and LiteLLMClient - see _diagnosis_system_prompt()'s docstring for why."""
    return (
        "You are the Fraud/Risk Agent for an order-exception resolution system. Score risk "
        "from 0.0-1.0 given the customer's risk profile and case context. IMPORTANT: weight "
        "prior fraud flags and return-reason-pattern consistency heavily; weight raw return "
        "COUNT only lightly — a high-volume but legitimate customer must NOT be penalized for "
        "return frequency alone. If case_context includes a non-empty related_fraud_signals "
        "list, this means one or more OTHER customers share this order's payment method and "
        "have a confirmed prior fraud flag of their own — a strong, cross-account risk signal "
        "(e.g. the same card used under different declared identities). Weight this heavily: "
        "a non-empty related_fraud_signals list should push risk_score well above the auto-"
        "execute threshold and set flag=true, and MUST be cited in reasons. Respond ONLY with "
        'JSON: {"risk_score": float, "flag": bool, "reasons": [str]}.'
    )


def _parse_json_response(content: str) -> dict:
    """Parses an LLM's JSON response, with a fallback extraction step
    for providers that don't enforce response_format the way Groq and
    Gemini do. Added specifically for Cohere - confirmed directly
    against litellm's own get_supported_openai_params(model="cohere/...")
    that response_format is NOT in Cohere's supported param list, unlike
    Groq and Gemini (both confirmed to support it) - so a Cohere
    response depends entirely on the system prompt's own "Respond ONLY
    with JSON" instruction, with no structural guarantee from the API
    itself. A direct json.loads() still succeeds for the common case
    (Cohere follows the instruction correctly), and this only engages
    its fallback extraction (stripping a ```json code fence, or pulling
    the first balanced {...} block) when that fails - so Groq/Gemini's
    already-clean responses take the exact same fast path as before
    this function existed; nothing changes for them."""
    import json
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse a JSON object from LLM response: {content!r}")


class BaseLLMClient(ABC):
    @abstractmethod
    def plan_next_diagnosis_step(self, case_context: dict, findings_so_far: dict) -> DiagnosisStepPlan: ...

    @abstractmethod
    def assess_fraud_risk(self, case_context: dict, customer_risk_profile: dict) -> dict: ...

    def summarize_context(self, existing_summary: str, new_item: dict) -> str:
        """Stage 2 memory upgrade: folds one aged-out diagnostic step into
        the running summary (app/memory/summary_buffer.py). GroqClient/
        LiteLLMClient/FakeLLMClient each override this with their own
        implementation (a real LLM fold, or FakeLLMClient's deterministic
        fold). Deliberately NOT an @abstractmethod, with this same
        deterministic fold as the base-class default: an earlier version
        of this made it abstract, which broke every existing test file's
        minimal custom BaseLLMClient subclass (e.g.
        tests/test_phase6_agents.py's NeverConcludesLLM) with
        "Can't instantiate abstract class ... without an implementation
        for 'summarize_context'" - a real backward-compatibility break
        found by running the FULL suite, not just this feature's own
        dedicated test file. A concrete default here means any existing
        or future minimal test subclass that only cares about
        plan_next_diagnosis_step/assess_fraud_risk keeps working
        unchanged, while SummaryBuffer.add() still gets a usable
        summary string from every BaseLLMClient without needing to
        branch on client type."""
        item_desc = f"[{new_item.get('agent', 'unknown')}] {new_item.get('summary', str(new_item))}"
        if not existing_summary:
            return item_desc
        return f"{existing_summary}; {item_desc}"


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

    def summarize_context(self, existing_summary: str, new_item: dict) -> str:
        import json
        user_prompt = json.dumps({"existing_summary": existing_summary, "new_item": new_item}, default=str)
        result = self._chat_json(_summary_fold_system_prompt(), user_prompt)
        return result.get("summary", existing_summary)


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

        # Skips litellm's own remote model-cost-map fetch from GitHub
        # on first use (found directly: this added several real
        # seconds of network latency on every fresh instantiation,
        # purely for cost-tracking metadata this project doesn't use).
        # Real, documented flag - see litellm/litellm_core_utils/
        # get_model_cost_map.py's own module docstring.
        import os
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

        # Silently drops a provider-unsupported request param instead
        # of raising - needed specifically because Cohere's chat API,
        # verified directly against litellm's own
        # get_supported_openai_params(model="cohere/...") rather than
        # assumed, does NOT support response_format at all (unlike Groq
        # and Gemini, both confirmed to support it the same way). Set
        # globally, once, rather than per-call - see _chat_json()'s own
        # comment for how this project compensates for Cohere's weaker
        # structured-output guarantee.
        import litellm
        litellm.drop_params = True

        # Multi-provider fallback, built DYNAMICALLY from whichever
        # keys are actually configured - NOT hardcoded to assume any
        # one provider is always present, matching this project's own
        # settings-driven pattern everywhere else (get_embedder(),
        # get_payment_gateway(), get_carrier_gateway()). Ordered by
        # structured-output reliability, most to least reliable, not
        # arbitrarily: Groq and Gemini both genuinely enforce
        # response_format="json_object" (confirmed via litellm's own
        # get_supported_openai_params for each); Cohere does not, so it
        # is deliberately last - a real last resort, not an equal peer.
        model_list = []
        fallback_chain = []

        if settings.groq_api_key:
            model_list.append({
                "model_name": "diagnosis-router",
                "litellm_params": {
                    "model": f"groq/{settings.router_model}",
                    "api_key": settings.groq_api_key,
                    # A conservative default for Groq's free tier —
                    # real deployments on a paid tier should raise this
                    # to match their actual plan limits. Rate-limiting
                    # client-side (rather than only reacting to Groq's
                    # own 429s after the fact) is the actual point of
                    # setting this at all.
                    "rpm": 30,
                },
            })
            fallback_chain.append("diagnosis-router")

        if settings.google_api_key:
            model_list.append({
                "model_name": "diagnosis-router-gemini",
                "litellm_params": {
                    "model": f"gemini/{settings.generation_model}",
                    "api_key": settings.google_api_key,
                },
            })
            fallback_chain.append("diagnosis-router-gemini")

        if settings.cohere_api_key:
            model_list.append({
                "model_name": "diagnosis-router-cohere",
                "litellm_params": {
                    "model": f"cohere_chat/{settings.cohere_generation_model}",
                    "api_key": settings.cohere_api_key,
                },
            })
            fallback_chain.append("diagnosis-router-cohere")

        if not model_list:
            raise RuntimeError(
                "No LLM provider configured - set GROQ_API_KEY, GOOGLE_API_KEY, "
                "or COHERE_API_KEY (see .env.example)."
            )

        from litellm import Router
        self._router = Router(
            model_list=model_list,
            # {primary: [rest, in order]} - litellm tries the primary
            # first, then walks the fallback list in order if it fails.
            # Only meaningful with 2+ providers configured; a single-
            # provider setup (e.g. Groq only, this project's original
            # behavior) gets an empty fallback list and behaves exactly
            # as before this change.
            fallbacks=[{fallback_chain[0]: fallback_chain[1:]}] if len(fallback_chain) > 1 else [],
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
        self._primary_model_name = fallback_chain[0]

    def _chat_json(self, system_prompt: str, user_prompt: str, purpose: str = "unknown") -> dict:
        from app.core.circuit_breaker import get_circuit_breaker
        # Name kept for alert/test continuity; it guards the whole provider
        # chain (Groq -> Gemini -> Cohere), not only Groq.
        breaker = get_circuit_breaker("llm_groq", failure_threshold=3, reset_timeout_seconds=30.0)
        meta = {"model": None, "usage": None}

        def _do_call():
            response = self._router.completion(
                model=self._primary_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            meta["model"] = getattr(response, "model", None)
            meta["usage"] = getattr(response, "usage", None)
            return response.choices[0].message.content

        started = time.monotonic()
        content, error = None, None
        try:
            content = breaker.call(_do_call)
            return _parse_json_response(content)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            raise
        finally:
            _record_llm_span(purpose, meta["model"], user_prompt, content, error,
                             int((time.monotonic() - started) * 1000), meta["usage"])

    def plan_next_diagnosis_step(self, case_context: dict, findings_so_far: dict) -> DiagnosisStepPlan:
        import json
        user_prompt = json.dumps({"case_context": case_context, "findings_so_far": findings_so_far}, default=str)
        result = self._chat_json(_diagnosis_system_prompt(), user_prompt, purpose="diagnosis")
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
        return self._chat_json(_fraud_system_prompt(), user_prompt, purpose="fraud")

    def summarize_context(self, existing_summary: str, new_item: dict) -> str:
        """Stage 2 memory upgrade. A deliberately cheap call - folding one
        item is a small, low-stakes piece of text work, not a reasoning
        step - so this goes through the SAME circuit-breaker-wrapped
        _chat_json() as diagnosis/fraud calls (consistent observability),
        but callers (SummaryBuffer.add()) are expected to treat a failure
        here as non-fatal and fall back to a deterministic fold rather
        than blocking the diagnosis loop on a summarization hiccup."""
        import json
        user_prompt = json.dumps({"existing_summary": existing_summary, "new_item": new_item}, default=str)
        result = self._chat_json(_summary_fold_system_prompt(), user_prompt, purpose="summary")
        return result.get("summary", existing_summary)


class FakeLLMClient(BaseLLMClient):
    """Sandbox-runnable substitute with genuine rule-based reasoning (see
    module docstring). This is what Phase 6's tests actually run against."""

    def summarize_context(self, existing_summary: str, new_item: dict) -> str:
        """Deterministic text fold - the SAME logic previously living
        directly inside SummaryBuffer._summarize() as a private stub
        (Stage 2 memory upgrade moved it here, matching every other
        piece of "real vs fake" reasoning in this project: the fake
        client's job is genuine, inspectable behavior, not a stand-in
        with different semantics from the real path). Kept intentionally
        simple - concatenation, not compression - since a real semantic
        fold requires an actual LLM; this is what runs when none is
        configured, or as the fallback when a real call fails."""
        item_desc = f"[{new_item.get('agent', 'unknown')}] {new_item.get('summary', str(new_item))}"
        if not existing_summary:
            return item_desc
        return f"{existing_summary}; {item_desc}"

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

        # Stage 1 memory upgrade: a non-empty related_fraud_signals list
        # means another customer sharing this order's payment fingerprint
        # has a confirmed prior fraud flag of their own (see
        # app/memory/graphiti_adapter.py's find_related_fraud_signals -
        # Neo4j Aura only). Weighted heavily and deliberately pushed past
        # a typical 0.6 flag threshold on its own - unlike raw return
        # count, a shared-card-with-a-flagged-account signal is exact,
        # not a fuzzy correlation, so it should not need to stack with
        # other weaker signals to trigger a flag.
        related_signals = case_context.get("related_fraud_signals") or []
        if related_signals:
            score += 0.7
            other_customer_ids = ", ".join(sorted({s["customer_id"] for s in related_signals}))
            reasons.append(
                f"payment fingerprint shared with {len(related_signals)} other customer(s) "
                f"with a confirmed prior fraud flag ({other_customer_ids})"
            )

        score = min(score, 1.0)
        return {
            "risk_score": score,
            "flag": score >= 0.6,
            "reasons": reasons,
        }


def get_llm_client() -> BaseLLMClient:
    """Auto-selects the real LiteLLMClient when ANY of GROQ_API_KEY,
    GOOGLE_API_KEY, or COHERE_API_KEY is configured, falling back to
    FakeLLMClient only when none are — mirrors the same settings-driven
    pattern as get_cache() (Redis) and get_qdrant_client() (Qdrant
    Cloud): nothing crashes on a partial cloud configuration, the real
    implementation activates automatically once a credential is
    present. Multi-provider fallback (which of the three actually gets
    used, and in what order) is built dynamically inside
    LiteLLMClient.__init__ - see its own docstring and this project's
    memory-upgrade README section for why that couldn't be hardcoded to
    assume any one provider.

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
    if settings.groq_api_key or settings.google_api_key or settings.cohere_api_key:
        return LiteLLMClient()
    return FakeLLMClient()
