"""
Isolated Groq API test - tests ONLY the Groq connection/model, completely
independent of Graphiti/Neo4j, to pinpoint a 404 error.

A 404 from https://api.groq.com/openai/v1/chat/completions (the correct,
real endpoint - this is not a URL bug) usually means one of:
  1. GROQ_API_KEY is invalid, revoked, or has a typo.
  2. The MODEL NAME is wrong or no longer available. Groq's model
     catalog changes over time - a model string that worked when this
     project was built may since have been renamed or deprecated.
     Check console.groq.com/docs/models for the CURRENT list and update
     ROUTER_MODEL in .env if needed (defaults to
     "llama-3.3-70b-versatile" - verify this is still listed there).

Usage:
    python3 scripts/test_groq_connection.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import get_settings


def main():
    settings = get_settings()
    if not settings.groq_api_key:
        print("GROQ_API_KEY is not set in .env - nothing to test.")
        sys.exit(1)

    print(f"Testing model: {settings.router_model}")
    print(f"API key (first 10 chars): {settings.groq_api_key[:10]}...")
    print()

    import httpx
    try:
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}", "Content-Type": "application/json"},
            json={
                "model": settings.router_model,
                "messages": [{"role": "user", "content": "Say 'test successful' and nothing else."}],
                "max_tokens": 200,  # generous headroom — reasoning models (e.g. gpt-oss-120b)
                                     # spend tokens on internal chain-of-thought BEFORE visible
                                     # output; a tight cap here can consume the whole budget on
                                     # reasoning and leave nothing for the actual answer, which
                                     # looks like a "success but empty response" — confirmed
                                     # directly: this happened with max_tokens=10 against
                                     # gpt-oss-120b specifically. The real production code
                                     # (app/agents/llm_client.py) never caps max_tokens at all,
                                     # so this was a test-script-only issue, not a production one.
            },
            timeout=15.0,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            if not content.strip():
                print("Connected successfully (200 OK), but the response content was empty.")
                print("This is expected for reasoning models under a tight max_tokens cap —")
                print("increase max_tokens further if this still happens, or check")
                print("response['choices'][0]['message'] for a 'reasoning' field consuming")
                print("the budget. The real app's own llm_client.py sets no cap at all, so")
                print("this is very unlikely to affect production calls.")
            else:
                print(f"SUCCESS - model responded: {content!r}")
        elif resp.status_code == 401:
            print("FAILED (401 Unauthorized): your GROQ_API_KEY is invalid or revoked.")
            print("Get a fresh one from console.groq.com/keys")
        elif resp.status_code == 404:
            print(f"FAILED (404 Not Found): the model '{settings.router_model}' is not")
            print("recognized by Groq's API. Check the CURRENT model list at:")
            print("  https://console.groq.com/docs/models")
            print("and update ROUTER_MODEL in .env to whatever's currently listed there")
            print("(model names/versions change over time - this is not a bug in this")
            print(" project's code, it's Groq's model catalog having moved on).")
        else:
            print(f"FAILED ({resp.status_code}): {resp.text[:500]}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
