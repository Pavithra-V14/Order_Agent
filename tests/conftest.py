"""
Session-wide test isolation from a real .env file AND from real
environment variables already present in the process.

Found running this project's real test suite against a real,
fully-configured .env (Groq/Mistral/EasyPost/Shippo/Stripe/Neo4j
credentials all filled in for actual cloud usage, via `uv run`): every
test written as "confirm this raises/falls back to the local substitute
when the credential is NOT configured" was popping the *process
environment variable* (os.environ.pop("GROQ_API_KEY", None)) to
simulate an absent credential - but pydantic-settings' Settings class
reads a real .env FILE directly on every construction, independent of
the current process environment. Popping an os.environ var does not
erase what's written in the .env file itself, so these tests silently
failed not because the code was wrong, but because the credential
genuinely WAS configured (correctly, for real usage) and the test's
"unconfigured" assumption no longer held.

A SECOND, separately-discovered layer of the same problem (from a real
user report running via `uv run`, which auto-loads .env into the
process environment BEFORE Python even starts): merely disabling
Settings' own env_file reading (below) does nothing if the credential
is already sitting in os.environ itself, since pydantic-settings reads
real environment variables with higher priority than the .env file
regardless. Real Stripe/EasyPost/Groq credentials in os.environ meant
dozens of tests that assumed a fake gateway (using made-up IDs like
"pi_test_123") instead got the REAL gateway, which correctly rejected
those fake IDs as genuine 404s - a real, confusing failure mode that
looked like a code bug but was actually a test-isolation gap.

This is a serious test-design flaw, not just a Windows quirk: any
developer with a real, filled-in .env sees the ENTIRE test suite start
silently depending on whichever cloud credentials happen to be present.
Fixed here, comprehensively, for the whole session: neither the real
.env file NOR any pre-existing real environment variable is visible to
the test suite - every test's settings come only from explicit
os.environ values (or defaults) set within that test itself, or from
monkeypatch.setattr on settings/get_settings where a specific test
wants to simulate a credential being present.
"""
import os
import pytest

from app.core.config import Settings

# Every credential/connection-string field that would otherwise select
# a REAL backend over this project's local-first/fake substitutes -
# checked directly against app/core/config.py's actual field list, not
# assumed. AUTH_DATABASE_URL and DATABASE_URL are deliberately NOT
# included here - tests that need Postgres (test_jwt_auth.py, and any
# fixture that explicitly sets AUTH_ENABLED=true) set it themselves,
# and clearing it here would break that real, deliberate usage.
_CLOUD_CREDENTIAL_ENV_VARS = [
    "GROQ_API_KEY", "MISTRAL_API_KEY", "GOOGLE_API_KEY", "COHERE_API_KEY",
    "QDRANT_API_KEY", "QDRANT_URL",
    "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
    "EASYPOST_API_KEY", "SHIPPO_API_KEY", "STRIPE_API_KEY",
    "NEO4J_URI", "NEO4J_PASSWORD",
    "REDIS_URL",
    # Not credentials, but same leakage risk: a real .env customizing
    # which model to use (e.g. after a model was deprecated/retired,
    # or to match a Graphiti-specific choice) leaks into tests the
    # exact same way via `uv run`'s auto-loading of .env into the
    # process environment, breaking any test that hardcodes an
    # expected request shape against this project's own DEFAULT model
    # name - found directly from a real user report showing
    # test_groq_client_sends_correct_request_shape failing with the
    # user's own customized ROUTER_MODEL instead of the code's actual
    # default, which the test correctly expects.
    "ROUTER_MODEL", "GRAPHITI_ROUTER_MODEL",
]


_SAVED_CREDENTIAL_ENV_VARS: dict | None = None


def pytest_configure(config):
    """Runs before pytest COLLECTS a single test file - genuinely the
    earliest possible hook, earlier than any fixture (even session-
    scoped ones only run before the FIRST TEST, which is AFTER
    collection has already imported every test module).

    Found necessary directly from a real user report: a session-scoped
    FIXTURE version of this clearing still wasn't early enough. Several
    test files construct a real Settings object at MODULE IMPORT TIME
    (e.g. test_deepeval_integration.py legitimately needs to know at
    collection time whether GROQ_API_KEY exists, to decide whether to
    skip) - and module imports happen during COLLECTION, before ANY
    fixture, session-scoped or otherwise, has run. A real QDRANT_URL
    present at collection time got captured into a module-level
    singleton (get_qdrant_client()'s cache) before the fixture-based
    clearing ever had a chance to run, sending real ingested data into
    a real cloud Qdrant collection with a different vector dimension
    than the local embedder produces - "Vector dimension error:
    expected dim: 1024, got 768" - and later tests, now looking at the
    (correctly cleared) local embedded Qdrant instead, found it empty.

    pytest_configure runs before all of that, closing the gap for good.
    """
    global _SAVED_CREDENTIAL_ENV_VARS
    _SAVED_CREDENTIAL_ENV_VARS = {var: os.environ.pop(var, None) for var in _CLOUD_CREDENTIAL_ENV_VARS}

    # graphiti_core calls python-dotenv's load_dotenv() at IMPORT time
    # (graphiti_core/graphiti.py and driver/driver.py). The first test to
    # import it - anything touching episodic memory - copied the real .env
    # back into os.environ mid-test, and that test then talked to the REAL
    # Stripe account (observed: a golden-set scenario's payment lookup hit
    # api.stripe.com and got "No such payment_intent: 'pi_golden_mc'").
    # Neutralized for the whole session before anything can import it.
    import dotenv
    import dotenv.main
    dotenv.load_dotenv = lambda *args, **kwargs: False
    dotenv.main.load_dotenv = dotenv.load_dotenv
    Settings.model_config["env_file"] = None
    from app.core.config import get_settings
    get_settings.cache_clear()

    # ALSO stashed under a dedicated, differently-named env var - found
    # necessary directly from a real user report: `from tests.conftest
    # import ...` in test_deepeval_integration.py imports conftest.py via
    # a REGULAR Python import path, which is not guaranteed to be the
    # same module OBJECT pytest's own hook-loading machinery uses
    # internally to call pytest_configure/pytest_unconfigure - meaning
    # a module-level Python variable set here was NOT reliably visible
    # through that second, independent import, silently keeping
    # _SAVED_CREDENTIAL_ENV_VARS at its unset None default from that
    # import's perspective even after this function had genuinely run.
    # os.environ is a real, single, process-wide dict, immune to this
    # class of Python module-identity mismatch - reading/writing through
    # it instead of a plain module attribute is what actually survives
    # being accessed via two different import paths.
    if _SAVED_CREDENTIAL_ENV_VARS.get("GROQ_API_KEY"):
        os.environ["_OEA_TEST_SAVED_GROQ_API_KEY"] = _SAVED_CREDENTIAL_ENV_VARS["GROQ_API_KEY"]
    # Same stash for ROUTER_MODEL - found necessary directly from a
    # real user report: with GROQ_API_KEY correctly restored,
    # test_deepeval_integration.py's tests started genuinely attempting
    # a real Groq call (the skip-detection fix above worked) - but then
    # failed with a real 404 Not Found from Groq's own API. GroqDeepEvalModel
    # reads settings.router_model, which - with ROUTER_MODEL cleared
    # session-wide by this same function - falls back to config.py's
    # hardcoded default ("llama-3.3-70b-versatile"). The user's own real
    # .env explicitly overrides ROUTER_MODEL to a different model
    # ("openai/gpt-oss-120b") - strong evidence the hardcoded default has
    # since been retired on Groq's end, which is exactly what a 404 on an
    # otherwise-valid, authenticated chat-completions request looks like.
    # This test file genuinely needs a real, currently-working model
    # name to make its real call succeed, not this project's possibly-
    # stale hardcoded default - so it gets the user's own configured
    # model back, the same way it gets its own real API key back.
    if _SAVED_CREDENTIAL_ENV_VARS.get("ROUTER_MODEL"):
        os.environ["_OEA_TEST_SAVED_ROUTER_MODEL"] = _SAVED_CREDENTIAL_ENV_VARS["ROUTER_MODEL"]


def pytest_unconfigure(config):
    """Restores the real environment after the ENTIRE test session
    ends - not after each test, since individual tests that need a fake
    credential set their own via a dedicated fixture (e.g. groq_settings
    in test_groq_client.py) and clean up after themselves."""
    if _SAVED_CREDENTIAL_ENV_VARS is not None:
        for var, value in _SAVED_CREDENTIAL_ENV_VARS.items():
            if value is not None:
                os.environ[var] = value
            else:
                os.environ.pop(var, None)
    os.environ.pop("_OEA_TEST_SAVED_GROQ_API_KEY", None)
    os.environ.pop("_OEA_TEST_SAVED_ROUTER_MODEL", None)


@pytest.fixture(autouse=True)
def _reenforce_credential_isolation_per_test():
    """Belt-and-suspenders on top of pytest_configure's session-level
    clearing above: re-clears the same credential env vars before EVERY
    single test, not just once at session start. Genuinely redundant if
    pytest_configure is doing its job correctly - added as a second,
    independent layer specifically because a real user report showed
    these credentials still leaking through even with the session-level
    hook in place.

    Deliberately does NOT restore anything after each test (only clears)
    - found and fixed a real, self-inflicted regression directly: an
    earlier version of this fixture DID restore the real credential
    after every test, which reintroduced the exact bug pytest_configure
    was built to fix. A module-scoped fixture in a LATER test file (e.g.
    test_phase3_rag.py's ingested_corpus) runs its own setup BEFORE this
    function-scoped fixture gets a chance to clear anything for that
    file's first test - so if the PRECEDING test's teardown had just
    restored the real QDRANT_URL, that module-scoped setup ingested real
    data into a real cloud collection again, and the actual test body
    (which THIS fixture then correctly cleared for) found the local
    embedded Qdrant empty - the "Collection not found" bug, reintroduced
    by the very fixture meant to prevent it. Real restoration only needs
    to happen once, at the very end of the whole session -
    pytest_unconfigure above already does that."""
    for var in _CLOUD_CREDENTIAL_ENV_VARS:
        os.environ.pop(var, None)
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield


@pytest.fixture(autouse=True)
def _bypass_auth_for_tests():
    """Real authentication was added to every protected endpoint after
    a direct audit found NONE existed at all. Rather than update every
    one of this suite's 240+ existing tests to construct and pass a
    real API key (which would also make most of them about testing
    auth plumbing rather than the actual behavior each test exists to
    verify), this sets settings.auth_enabled=False for the whole
    session via an environment variable.

    This does NOT mean auth is untested — see tests/test_auth.py, which
    explicitly sets AUTH_ENABLED=true and clears the settings cache to
    prove auth is genuinely enforced when not bypassed. This fixture
    bypasses auth for tests whose actual subject is something else
    entirely (diagnosis logic, RAG retrieval, idempotency, etc.), which
    is the large majority of this suite.

    Deliberately an environment variable + settings flag, NOT
    FastAPI's app.dependency_overrides — found directly that the
    latter doesn't survive several existing test fixtures throughout
    this project that call importlib.reload(app.main) for isolation,
    which creates a BRAND NEW app object with an empty
    dependency_overrides dict each time, silently discarding any
    override applied to a previous instance. A setting checked fresh
    inside get_current_api_key() itself has no such problem, since
    it's never tied to a specific app object at all.
    """
    import os
    os.environ["AUTH_ENABLED"] = "false"
    from app.core.config import get_settings
    get_settings.cache_clear()
    yield
    os.environ.pop("AUTH_ENABLED", None)
    get_settings.cache_clear()
