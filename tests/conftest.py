"""
Session-wide test isolation from a real .env file.

Found running this project's real test suite against a real,
fully-configured .env (Groq/Mistral/EasyPost/Shippo/Neo4j credentials
all filled in for actual cloud usage): every test written as "confirm
this raises/falls back to the local substitute when the credential is
NOT configured" was popping the *process environment variable*
(os.environ.pop("GROQ_API_KEY", None)) to simulate an absent credential
- but pydantic-settings' Settings class reads a real .env FILE directly
on every construction, independent of the current process environment.
Popping an os.environ var does not erase what's written in the .env file
itself, so these tests silently failed not because the code was wrong,
but because the credential genuinely WAS configured (correctly, for real
usage) and the test's "unconfigured" assumption no longer held.

This is a serious test-design flaw, not just a Windows quirk: any
developer with a real, filled-in .env sees the ENTIRE test suite start
silently depending on whichever cloud credentials happen to be present -
tests that should be fast, deterministic, and offline instead start
making (or attempting) real network calls to Groq/Mistral/Neo4j/etc.
depending on what's in a file that has nothing to do with the test suite
itself. Fixed here, once, for the whole session: the real .env file is
never read during `pytest tests/` - every test's settings come only from
explicit os.environ values (or defaults) set within that test itself.
"""
import pytest

from app.core.config import Settings


@pytest.fixture(autouse=True, scope="session")
def _isolate_tests_from_real_env_file():
    Settings.model_config["env_file"] = None
    yield
