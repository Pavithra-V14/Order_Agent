"""
DeepEval custom model, backed by Groq - not OpenAI, which DeepEval
defaults to and this project has no credential or need for. Wires the
SAME real Groq integration already used for diagnosis/fraud assessment
(app/agents/llm_client.py's GroqClient) into DeepEval's judge-model
interface, so DeepEval's LLM-based metrics (FaithfulnessMetric,
AnswerRelevancyMetric, etc.) run against the project's own actual
model, not a different provider entirely.

HONEST LIMITATION, same as GroqClient's own docstring: this sandbox has
no network route to api.groq.com, so this cannot be network-tested from
here. The request/response handling follows the exact same pattern
already verified against a realistic mocked response shape
(tests/test_groq_client.py) - what's untested is "can this sandbox
reach the internet," not "is the code right." Tests using this model
are marked to skip cleanly when GROQ_API_KEY isn't configured, rather
than fail with a confusing network error.
"""
import json

from deepeval.models.base_model import DeepEvalBaseLLM


class GroqDeepEvalModel(DeepEvalBaseLLM):
    def __init__(self):
        from app.core.config import get_settings
        settings = get_settings()
        self._api_key = settings.groq_api_key
        self._model_name = settings.router_model
        if not self._api_key:
            raise RuntimeError("GROQ_API_KEY not configured - see .env.example")
        super().__init__(model=self._model_name)

    def load_model(self):
        import httpx
        return httpx.Client(
            base_url="https://api.groq.com/openai/v1",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            timeout=60.0,
        )

    def generate(self, prompt: str) -> str:
        # DeepEval's metrics send free-form evaluation prompts (often
        # asking for a JSON-shaped verdict within the prompt text
        # itself) - no response_format constraint here, unlike
        # GroqClient._chat_json, since DeepEval controls the exact
        # output contract via its own prompt engineering and parses
        # the raw text response itself.
        resp = self.model.post("/chat/completions", json={
            "model": self._model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        })
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    async def a_generate(self, prompt: str) -> str:
        # DeepEval's async path - a real async httpx client, not just
        # wrapping the sync call, so metrics that genuinely run
        # concurrently (e.g. evaluating multiple test cases in
        # parallel) don't block on each other unnecessarily.
        import httpx
        async with httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            timeout=60.0,
        ) as client:
            resp = await client.post("/chat/completions", json={
                "model": self._model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            })
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def get_model_name(self) -> str:
        return f"groq/{self._model_name}"
