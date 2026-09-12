"""
Tests for GroqClient (app/agents/llm_client.py) using respx to mock the
actual HTTP layer with a realistic Groq API response shape. This proves
request construction, response parsing, and retry logic are correct,
without needing real network access to api.groq.com.
"""
import json

import httpx
import pytest
import respx

@pytest.fixture(autouse=True)
def groq_settings():
    import os
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key_for_mocked_requests"
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    os.environ.pop("GROQ_API_KEY", None)
    get_settings.cache_clear()

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

def _groq_response(content_dict):
    return {
        "id": "chatcmpl-fake123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "llama-3.3-70b-versatile",
        "service_tier": "on_demand",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": json.dumps(content_dict)},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
    }

def test_groq_client_raises_clearly_without_api_key():
    import os
    os.environ.pop("GROQ_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.agents.llm_client import GroqClient
    with pytest.raises(RuntimeError, match="GROQ_API_KEY not configured"):
        GroqClient()

@respx.mock
def test_groq_client_plan_next_diagnosis_step_parses_real_response_shape(groq_settings):
    from app.agents.llm_client import GroqClient

    mocked_content = {
        "action": "check_payment",
        "reasoning": "Order status is 'payment_failed' - checking the payment gateway directly.",
        "root_causes": None,
    }
    respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_groq_response(mocked_content)))

    client = GroqClient()
    plan = client.plan_next_diagnosis_step(
        case_context={"order_id": "ORD-1"},
        findings_so_far={"order": {"status": "payment_failed"}},
    )

    assert plan.action == "check_payment"
    assert "payment gateway" in plan.reasoning
    assert plan.root_causes is None

@respx.mock
def test_groq_client_conclude_with_root_causes(groq_settings):
    from app.agents.llm_client import GroqClient

    mocked_content = {
        "action": "conclude",
        "reasoning": "All systems checked.",
        "root_causes": ["payment_issue: transaction status is 'declined'"],
    }
    respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_groq_response(mocked_content)))

    client = GroqClient()
    plan = client.plan_next_diagnosis_step(case_context={}, findings_so_far={"payment": {"status": "declined"}})

    assert plan.action == "conclude"
    assert plan.root_causes == ["payment_issue: transaction status is 'declined'"]

@respx.mock
def test_groq_client_assess_fraud_risk_parses_real_response_shape(groq_settings):
    from app.agents.llm_client import GroqClient

    mocked_content = {"risk_score": 0.15, "flag": False, "reasons": ["high return count, weighted lightly"]}
    respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_groq_response(mocked_content)))

    client = GroqClient()
    result = client.assess_fraud_risk(case_context={}, customer_risk_profile={"total_return_cases": 15})

    assert result["risk_score"] == 0.15
    assert result["flag"] is False

@respx.mock
def test_groq_client_retries_on_transient_failure_then_succeeds(groq_settings):
    """Proves the retry logic actually retries - first call times out,
    second call (the retry) succeeds."""
    from app.agents.llm_client import GroqClient

    mocked_content = {"action": "conclude", "reasoning": "ok", "root_causes": ["no_anomaly_detected: clean"]}
    route = respx.post(GROQ_URL)
    route.side_effect = [
        httpx.TimeoutException("simulated timeout"),
        httpx.Response(200, json=_groq_response(mocked_content)),
    ]

    client = GroqClient()
    plan = client.plan_next_diagnosis_step(case_context={}, findings_so_far={})

    assert plan.action == "conclude"
    assert route.call_count == 2, "expected exactly one retry after the first simulated failure"

@respx.mock
def test_groq_client_raises_after_exhausting_retries(groq_settings):
    from app.agents.llm_client import GroqClient

    respx.post(GROQ_URL).mock(side_effect=httpx.TimeoutException("simulated permanent timeout"))

    client = GroqClient()
    with pytest.raises(RuntimeError, match="Groq API call failed"):
        client.plan_next_diagnosis_step(case_context={}, findings_so_far={})

@respx.mock
def test_groq_client_sends_correct_request_shape(groq_settings):
    from app.agents.llm_client import GroqClient
    from app.core.config import get_settings

    mocked_content = {"action": "conclude", "reasoning": "ok", "root_causes": ["no_anomaly_detected: clean"]}
    route = respx.post(GROQ_URL).mock(return_value=httpx.Response(200, json=_groq_response(mocked_content)))

    client = GroqClient()
    client.plan_next_diagnosis_step(case_context={"order_id": "ORD-1"}, findings_so_far={})

    sent_request = route.calls[0].request
    assert sent_request.headers["Authorization"] == "Bearer gsk_fake_test_key_for_mocked_requests"
    body = json.loads(sent_request.content)
    # Asserted against the actual configured setting, not a hardcoded
    # literal - found necessary directly from a real user report where
    # a stale/mixed file state made a hardcoded literal here drift out
    # of sync with config.py's real default, producing a confusing
    # failure that looked like a code bug but was actually a fixture-
    # isolation gap (ROUTER_MODEL leaking in from a real .env via `uv
    # run`, since fixed in conftest.py). Asserting against the live
    # setting makes this test correct regardless of what that default
    # is configured to be, as long as the request genuinely reflects it.
    assert body["model"] == get_settings().router_model
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.0

def test_get_llm_client_returns_litellm_client_when_key_configured(groq_settings):
    """Updated after wiring in a real LLM gateway (litellm.Router) -
    get_llm_client() now returns LiteLLMClient, not GroqClient
    directly. GroqClient itself is kept in the codebase (and its own
    tests above still cover its request/response handling), but is no
    longer what the application actually uses in production - see
    get_llm_client()'s own docstring for why."""
    from app.agents.llm_client import get_llm_client, LiteLLMClient
    client = get_llm_client()
    assert isinstance(client, LiteLLMClient)

def test_get_llm_client_returns_fake_when_no_key():
    import os
    os.environ.pop("GROQ_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.agents.llm_client import get_llm_client, FakeLLMClient
    client = get_llm_client()
    assert isinstance(client, FakeLLMClient)
