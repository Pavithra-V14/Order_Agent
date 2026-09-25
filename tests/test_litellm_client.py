"""
Tests for LiteLLMClient (app/agents/llm_client.py) - the real LLM
gateway, replacing direct GroqClient calls with litellm.Router. Uses
respx to mock the actual HTTP layer (litellm's own Groq provider calls
the same real https://api.groq.com/openai/v1/chat/completions endpoint
under the hood), same verification strategy as test_groq_client.py.

Found and fixed two real things while building this: (1) litellm's
Groq response parser requires a "service_tier" field that a
hand-written mock easily omits (a real Groq response always includes
it, defaulting to "on_demand" - see Groq's own service-tier docs), and
(2) litellm attempts a real network fetch of its remote model cost map
on first use, adding several real seconds of latency for cost-tracking
metadata this project doesn't use - disabled via LITELLM_LOCAL_MODEL_COST_MAP.
"""
import json

import httpx
import pytest
import respx

from tests.test_groq_client import GROQ_URL, _groq_response


@pytest.fixture(autouse=True)
def groq_settings():
    import os
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key_for_mocked_requests"
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    os.environ.pop("GROQ_API_KEY", None)
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def reset_llm_circuit_breaker():
    from app.core.circuit_breaker import reset_all_breakers
    reset_all_breakers()
    yield
    reset_all_breakers()


@respx.mock
def test_litellm_client_plan_next_diagnosis_step_parses_real_response_shape(groq_settings):
    from app.agents.llm_client import LiteLLMClient

    mocked_content = {"action": "check_order", "reasoning": "need order details first", "root_causes": None}
    respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_groq_response(mocked_content)))

    client = LiteLLMClient()
    plan = client.plan_next_diagnosis_step(case_context={"order_id": "ORD-1"}, findings_so_far={})

    assert plan.action == "check_order"
    assert plan.reasoning == "need order details first"
    assert plan.root_causes is None


@respx.mock
def test_litellm_client_assess_fraud_risk_parses_real_response_shape(groq_settings):
    from app.agents.llm_client import LiteLLMClient

    mocked_content = {"risk_score": 0.85, "flag": True, "reasons": ["prior fraud flag"]}
    respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_groq_response(mocked_content)))

    client = LiteLLMClient()
    result = client.assess_fraud_risk(case_context={}, customer_risk_profile={"fraud_flags_raised": 1})

    assert result["risk_score"] == 0.85
    assert result["flag"] is True


@respx.mock
def test_litellm_client_sends_correct_request_shape_to_groq(groq_settings):
    """Proves litellm.Router genuinely routes to the real Groq endpoint
    with the correct model name and auth - not a different provider or
    a malformed request."""
    from app.agents.llm_client import LiteLLMClient
    from app.core.config import get_settings

    mocked_content = {"action": "conclude", "reasoning": "ok", "root_causes": ["no_anomaly_detected: clean"]}
    route = respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_groq_response(mocked_content)))

    client = LiteLLMClient()
    client.plan_next_diagnosis_step(case_context={"order_id": "ORD-1"}, findings_so_far={})

    assert route.called
    sent_request = route.calls[0].request
    assert sent_request.headers["Authorization"] == "Bearer gsk_fake_test_key_for_mocked_requests"
    body = json.loads(sent_request.content)
    # Asserted against the actual configured setting, not a hardcoded
    # literal - same fix as test_groq_client.py's equivalent test, for
    # the same reason: a stale hardcoded value drifting out of sync
    # with config.py's real default produces a confusing failure that
    # looks like a code bug but isn't one.
    assert body["model"] == get_settings().router_model
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.0


@respx.mock
def test_litellm_client_retries_on_transient_failure_then_succeeds(groq_settings):
    """Real exponential-backoff retry, verified against litellm's OWN
    genuine implementation (checked directly against source before
    building this integration) - not this project re-implementing
    retry logic itself."""
    from app.agents.llm_client import LiteLLMClient

    mocked_content = {"action": "conclude", "reasoning": "recovered", "root_causes": ["no_anomaly_detected: ok"]}
    route = respx.post(GROQ_URL).mock(side_effect=[
        httpx.Response(500, json={"error": {"message": "internal server error"}}),
        httpx.Response(200, json=_groq_response(mocked_content)),
    ])

    client = LiteLLMClient()
    result = client.plan_next_diagnosis_step(case_context={"order_id": "ORD-1"}, findings_so_far={})

    assert result.action == "conclude"
    assert route.call_count == 2, "expected exactly one real retry after the first transient failure"


@respx.mock
def test_litellm_client_wraps_calls_in_the_projects_own_circuit_breaker(groq_settings):
    """THE regression test for the actual point of this integration:
    a sustained Groq outage must trip THIS PROJECT'S OWN circuit
    breaker (app/core/circuit_breaker.py) - the same one payment/
    carrier/wms use - not just litellm's internal, separately-observed
    cooldown mechanism. This is what makes an LLM outage show up in
    /admin/alerts the same way every other dependency's outage does."""
    from app.agents.llm_client import LiteLLMClient
    from app.core.circuit_breaker import get_circuit_breaker, CircuitState

    # Every call fails - permanently, not transiently - matching a
    # genuine sustained outage.
    respx.post(GROQ_URL).mock(return_value=httpx.Response(500, json={"error": {"message": "down"}}))

    client = LiteLLMClient()
    failures = 0
    for _ in range(3):
        try:
            client.plan_next_diagnosis_step(case_context={"order_id": "ORD-1"}, findings_so_far={})
        except Exception:
            failures += 1

    assert failures == 3
    breaker = get_circuit_breaker("llm_groq", failure_threshold=3, reset_timeout_seconds=30.0)
    assert breaker.state == CircuitState.OPEN, (
        "the project's own circuit breaker must trip OPEN after sustained LLM failures, exactly "
        "like it does for payment/carrier/wms"
    )


def test_litellm_client_configures_real_rate_limiting_and_cooldown(groq_settings):
    """Verifies the actual Router configuration (checked directly
    against litellm's real, documented model_list/Router parameters
    before building this - not assumed) includes genuine rate limiting
    (rpm) and cooldown (allowed_fails/cooldown_time) settings, without
    needing a real network call to check construction-time config."""
    from app.agents.llm_client import LiteLLMClient

    client = LiteLLMClient()
    deployment = client._router.get_model_list(model_name="diagnosis-router")[0]
    assert deployment["litellm_params"]["rpm"] == 30
    assert client._router.allowed_fails == 3
    assert client._router.cooldown_time == 30


def test_litellm_client_raises_clearly_without_api_key():
    """Updated for the multi-provider fallback upgrade: LiteLLMClient
    now activates on ANY of Groq/Gemini/Cohere, so it only raises when
    ALL THREE are absent - not just Groq."""
    import os
    os.environ.pop("GROQ_API_KEY", None)
    os.environ.pop("GOOGLE_API_KEY", None)
    os.environ.pop("COHERE_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.agents.llm_client import LiteLLMClient
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        LiteLLMClient()


# --- Multi-provider fallback (Groq -> Gemini -> Cohere), built dynamically ---
#
# Found and fixed a real gap: GOOGLE_API_KEY/COHERE_API_KEY were
# declared in config.py but never wired into any LLM client anywhere -
# a real Groq rate-limit error reported "Available Model Group
# Fallbacks=None", which was litellm telling the truth, not a
# misconfiguration. Fixed by building the Router's model_list and
# fallback chain DYNAMICALLY from whichever keys are actually
# configured, not hardcoded to assume any one provider - same pattern
# this project already uses for get_embedder()/get_payment_gateway().

def _clear_llm_keys():
    import os
    for key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "COHERE_API_KEY"):
        os.environ.pop(key, None)
    from app.core.config import get_settings
    get_settings.cache_clear()


def test_dynamic_model_list_groq_only(groq_settings):
    """The original, pre-upgrade behavior must be byte-identical when
    only Groq is configured - no fallbacks, single deployment."""
    from app.agents.llm_client import LiteLLMClient
    client = LiteLLMClient()

    assert client._primary_model_name == "diagnosis-router"
    assert len(client._router.model_list) == 1
    assert client._router.model_list[0]["model_name"] == "diagnosis-router"
    assert not client._router.fallbacks


def test_dynamic_model_list_groq_and_gemini():
    _clear_llm_keys()
    import os
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    os.environ["GOOGLE_API_KEY"] = "fake_google_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.agents.llm_client import LiteLLMClient
    client = LiteLLMClient()

    assert client._primary_model_name == "diagnosis-router"
    model_names = {d["model_name"] for d in client._router.model_list}
    assert model_names == {"diagnosis-router", "diagnosis-router-gemini"}
    gemini_deployment = client._router.get_model_list(model_name="diagnosis-router-gemini")[0]
    assert gemini_deployment["litellm_params"]["model"] == "gemini/gemini-2.0-flash"
    assert client._router.fallbacks == [{"diagnosis-router": ["diagnosis-router-gemini"]}]

    _clear_llm_keys()


def test_dynamic_model_list_all_three_providers():
    _clear_llm_keys()
    import os
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    os.environ["GOOGLE_API_KEY"] = "fake_google_key"
    os.environ["COHERE_API_KEY"] = "fake_cohere_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.agents.llm_client import LiteLLMClient
    client = LiteLLMClient()

    assert client._primary_model_name == "diagnosis-router"
    model_names = {d["model_name"] for d in client._router.model_list}
    assert model_names == {"diagnosis-router", "diagnosis-router-gemini", "diagnosis-router-cohere"}
    cohere_deployment = client._router.get_model_list(model_name="diagnosis-router-cohere")[0]
    assert cohere_deployment["litellm_params"]["model"] == "cohere_chat/command-r-plus"
    # Fallback chain must be ORDERED: Groq first (primary), then Gemini,
    # then Cohere - matching structured-output reliability, not
    # arbitrary or reversed.
    assert client._router.fallbacks == [
        {"diagnosis-router": ["diagnosis-router-gemini", "diagnosis-router-cohere"]}
    ]

    _clear_llm_keys()


def test_dynamic_model_list_gemini_only_no_groq():
    """A real, previously-impossible configuration: no Groq at all,
    just Gemini. Must work as a standalone primary, not require Groq to
    exist first."""
    _clear_llm_keys()
    import os
    os.environ["GOOGLE_API_KEY"] = "fake_google_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.agents.llm_client import LiteLLMClient
    client = LiteLLMClient()

    assert client._primary_model_name == "diagnosis-router-gemini"
    assert len(client._router.model_list) == 1
    assert not client._router.fallbacks

    _clear_llm_keys()


def test_get_llm_client_activates_litellm_for_gemini_only():
    """get_llm_client()'s OWN activation condition must also recognize
    a Gemini-only (no Groq) configuration - the dynamic model_list
    building above is useless if the factory function never gets that
    far."""
    _clear_llm_keys()
    import os
    os.environ["GOOGLE_API_KEY"] = "fake_google_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.agents.llm_client import get_llm_client, LiteLLMClient
    client = get_llm_client()
    assert isinstance(client, LiteLLMClient)

    _clear_llm_keys()


@respx.mock
def test_genuinely_falls_back_to_gemini_when_groq_is_rate_limited():
    """THE end-to-end proof, at the real HTTP layer (not just
    inspecting Router config): when Groq's real endpoint returns a 429,
    litellm's Router must actually retry through the Gemini deployment
    and the final result must reflect GEMINI's response, not fail
    outright the way this project's real Groq rate-limit error did
    before any fallback existed."""
    _clear_llm_keys()
    import os
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    os.environ["GOOGLE_API_KEY"] = "fake_google_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    groq_route = respx.post(GROQ_URL).mock(
        return_value=httpx.Response(429, json={"error": {"message": "rate limit", "type": "tokens"}})
    )
    gemini_route = respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*").mock(
        return_value=httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": json.dumps(
                {"action": "conclude", "reasoning": "gemini fallback response",
                 "root_causes": ["no_anomaly_detected: gemini handled this"]}
            )}], "role": "model"}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 10, "totalTokenCount": 20},
        })
    )

    from app.agents.llm_client import LiteLLMClient
    from app.core.circuit_breaker import reset_all_breakers
    reset_all_breakers()
    client = LiteLLMClient()
    result = client.plan_next_diagnosis_step(case_context={"order_id": "ORD-1"}, findings_so_far={})

    assert groq_route.called
    assert gemini_route.called
    assert "gemini" in result.reasoning.lower()

    _clear_llm_keys()


# --- _parse_json_response resilience (for Cohere's weaker guarantee) --------

def test_parse_json_response_handles_clean_json():
    from app.agents.llm_client import _parse_json_response
    assert _parse_json_response('{"action": "conclude"}') == {"action": "conclude"}


def test_parse_json_response_handles_markdown_fenced_json():
    """Cohere (confirmed via litellm's own get_supported_openai_params
    to NOT support response_format) may wrap its output in a code
    fence despite the prompt's instruction - this must still parse."""
    from app.agents.llm_client import _parse_json_response
    content = '```json\n{"action": "conclude"}\n```'
    assert _parse_json_response(content) == {"action": "conclude"}


def test_parse_json_response_handles_surrounding_prose():
    from app.agents.llm_client import _parse_json_response
    content = 'Here is the JSON you requested:\n{"action": "conclude"}\nLet me know if you need anything else.'
    assert _parse_json_response(content) == {"action": "conclude"}


def test_parse_json_response_raises_clearly_on_unparseable_content():
    from app.agents.llm_client import _parse_json_response
    with pytest.raises(ValueError, match="Could not parse a JSON object"):
        _parse_json_response("I cannot help with that request.")


@respx.mock
def test_every_llm_call_inside_a_case_records_a_traced_span(groq_settings, tmp_path):
    """R9: LLM calls were the one step of a case with no trace span - no
    model, prompt version, tokens or latency were ever recorded."""
    import importlib, os
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path / 'llm_trace.db'}"
    from app.core.config import get_settings
    get_settings.cache_clear()
    import app.core.db as db_module
    importlib.reload(db_module)
    db_module.init_db()
    from app.agents.llm_client import LiteLLMClient, set_llm_trace, reset_llm_trace, PROMPT_VERSIONS
    from app.core.metrics import compute_llm_metrics

    respx.post(GROQ_URL).mock(return_value=httpx.Response(
        200, json=_groq_response({"risk_score": 0.1, "flag": False, "reasons": []})))
    token = set_llm_trace("case-llm-trace")
    try:
        LiteLLMClient().assess_fraud_risk(case_context={}, customer_risk_profile={})
    finally:
        reset_llm_trace(token)
    LiteLLMClient().assess_fraud_risk(case_context={}, customer_risk_profile={})   # outside a case: no span

    db = db_module.SessionLocal()
    spans = db.query(db_module.TraceSpanRecord).filter_by(trace_id="case-llm-trace").all()
    assert len(spans) == 1 and spans[0].agent_or_tool_name == "llm.fraud"
    meta = spans[0].span_metadata
    assert meta["prompt_version"] == PROMPT_VERSIONS["fraud"] and meta["ok"] is True
    assert meta["latency_ms"] >= 0 and meta["prompt_tokens"] is not None
    metrics = compute_llm_metrics(db)
    assert metrics["total_llm_calls"] == 1 and metrics["by_model_and_prompt"][0]["failure_rate"] == 0.0
    db.close()
