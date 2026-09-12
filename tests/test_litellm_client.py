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
    import os
    os.environ.pop("GROQ_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.agents.llm_client import LiteLLMClient
    with pytest.raises(RuntimeError, match="GROQ_API_KEY not configured"):
        LiteLLMClient()
