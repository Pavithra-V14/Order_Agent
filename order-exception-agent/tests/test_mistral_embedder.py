"""
Tests for MistralEmbedder using respx to mock a realistic Mistral
embeddings API response.
"""
import numpy as np
import httpx
import pytest
import respx

MISTRAL_URL = "https://api.mistral.ai/v1/embeddings"


@pytest.fixture(autouse=True)
def mistral_settings():
    import os
    os.environ["MISTRAL_API_KEY"] = "fake_mistral_key_for_mocked_requests"
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    os.environ.pop("MISTRAL_API_KEY", None)
    get_settings.cache_clear()


def _mistral_response(vectors):
    return {
        "id": "embd-fake123",
        "object": "list",
        "data": [{"object": "embedding", "embedding": v, "index": i} for i, v in enumerate(vectors)],
        "model": "mistral-embed",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }


def test_mistral_embedder_raises_clearly_without_api_key():
    import os
    os.environ.pop("MISTRAL_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.rag.embeddings import MistralEmbedder
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY not configured"):
        MistralEmbedder()


@respx.mock
def test_mistral_embedder_parses_real_response_shape(mistral_settings):
    from app.rag.embeddings import MistralEmbedder

    fake_vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    respx.post(MISTRAL_URL).mock(return_value=httpx.Response(200, json=_mistral_response(fake_vectors)))

    embedder = MistralEmbedder()
    result = embedder.embed(["first text", "second text"])

    assert result.shape == (2, 3)
    assert np.allclose(result[0], [0.1, 0.2, 0.3])
    assert np.allclose(result[1], [0.4, 0.5, 0.6])


@respx.mock
def test_mistral_embedder_sends_correct_request_shape(mistral_settings):
    from app.rag.embeddings import MistralEmbedder

    route = respx.post(MISTRAL_URL).mock(
        return_value=httpx.Response(200, json=_mistral_response([[1.0, 2.0]]))
    )

    embedder = MistralEmbedder()
    embedder.embed(["some text"])

    sent = route.calls[0].request
    assert sent.headers["Authorization"] == "Bearer fake_mistral_key_for_mocked_requests"
    import json
    body = json.loads(sent.content)
    assert body["model"] == "mistral-embed"
    assert body["input"] == ["some text"]


@respx.mock
def test_mistral_embedder_retries_on_transient_failure(mistral_settings):
    from app.rag.embeddings import MistralEmbedder

    route = respx.post(MISTRAL_URL)
    route.side_effect = [
        httpx.TimeoutException("simulated timeout"),
        httpx.Response(200, json=_mistral_response([[1.0, 2.0]])),
    ]

    embedder = MistralEmbedder()
    result = embedder.embed(["text"])

    assert result.shape == (1, 2)
    assert route.call_count == 2


def test_mistral_embedder_empty_input_returns_empty_array(mistral_settings):
    from app.rag.embeddings import MistralEmbedder
    embedder = MistralEmbedder()
    result = embedder.embed([])
    assert result.shape == (0, 1024)


def test_get_embedder_returns_mistral_when_key_configured(mistral_settings):
    from app.rag.embeddings import get_embedder, MistralEmbedder
    embedder = get_embedder()
    assert isinstance(embedder, MistralEmbedder)


def test_get_embedder_returns_tfidf_when_no_key():
    import os
    os.environ.pop("MISTRAL_API_KEY", None)
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.rag.embeddings import get_embedder, TfidfEmbedder
    embedder = get_embedder()
    assert isinstance(embedder, TfidfEmbedder)
