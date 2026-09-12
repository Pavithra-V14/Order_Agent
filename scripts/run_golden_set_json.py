"""
Runs the golden set and prints results as JSON to stdout. Deliberately
a SEPARATE, standalone script rather than something callable in-process
from the live FastAPI app: tests/golden_set.py's scenarios each call
importlib.reload(app.core.db), swapping the database connection to a
throwaway temp SQLite file. Calling these functions directly from
within a live application would leave the app's OWN database
connection corrupted (pointed at a scenario's now-deleted temp file)
for the rest of the process's lifetime.

Running this as a subprocess (a fresh Python process) contains all of
that reloading/swapping inside a process that exits immediately after,
never touching the live app's actual database connection at all.

Usage:
    python3 scripts/run_golden_set_json.py                  # runs all 8
    python3 scripts/run_golden_set_json.py scenario_a scenario_b   # runs only the named ones
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# CRITICAL: disable Settings' real .env file reading BEFORE any other
# import — found necessary as the actual root cause of a real production
# crash: stripping cloud-credential env vars from this subprocess's
# inherited environment (a fix applied earlier, in
# app/api/v1/testing.py) does NOT stop pydantic-settings from
# independently reading the real .env FILE FROM DISK, since
# Settings.model_config specifies env_file=".env" — a file path, not an
# environment variable, so it's read completely independent of what
# was or wasn't inherited via os.environ. This subprocess sits in the
# same project directory as the real .env file regardless of how it's
# invoked (via the API's subprocess.run(), or a person running it
# directly from a terminal), so it would otherwise pick up real
# GROQ_API_KEY/NEO4J_URI/etc. on its own and make real, slow cloud
# calls — exactly what caused a real golden-set scenario to crash with
# a Graphiti timeout even after the env-stripping fix. This mirrors
# tests/conftest.py's identical fix for pytest, applied here because
# this script runs as its own separate process that conftest.py's
# session-scoped fixture never touches.
from app.core.config import Settings
Settings.model_config["env_file"] = None

# THIRD, more direct layer of protection, added after the above two
# still weren't enough in a real, reported case (a real Windows
# environment where a golden-set run somehow still selected a real
# StripeGateway, crashing an unrelated scenario that called a
# fake-only method). Rather than keep reasoning about WHY inherited
# env vars or .env file reading might still leak through on some
# specific OS/environment combination this hasn't been tested against,
# this forcibly clears every cloud-credential field on the constructed
# Settings object directly, regardless of what path it might otherwise
# have been populated from — a "trust nothing, verify directly"
# guarantee rather than one more assumption about mechanism.
import app.core.config as config_module
_original_cached_get_settings = config_module.get_settings  # the lru_cache-wrapped function itself
_original_get_settings = _original_cached_get_settings.__wrapped__


def _isolated_get_settings():
    settings = _original_cached_get_settings()
    for field in ("stripe_api_key", "easypost_api_key", "shippo_api_key",
                  "groq_api_key", "mistral_api_key", "qdrant_url", "qdrant_api_key",
                  "neo4j_uri", "neo4j_password", "redis_url",
                  "langfuse_public_key", "langfuse_secret_key", "tracing_enabled"):
        if hasattr(settings, field):
            object.__setattr__(settings, field, None if field != "tracing_enabled" else False)
    return settings


# Preserve .cache_clear() — other code (including golden_set.py's own
# scenarios) calls get_settings.cache_clear() expecting the normal
# lru_cache interface. Found directly: replacing get_settings with a
# plain function that lacks this method broke a DIFFERENT scenario
# with "'function' object has no attribute 'cache_clear'" the moment
# this fix's first version ran for real.
_isolated_get_settings.cache_clear = _original_cached_get_settings.cache_clear

config_module.get_settings = _isolated_get_settings

from tests.golden_set import ALL_SCENARIOS


def main():
    requested_names = sys.argv[1:]
    if requested_names:
        scenarios = [s for s in ALL_SCENARIOS if s.__name__ in requested_names]
    else:
        scenarios = ALL_SCENARIOS

    results = []
    for scenario in scenarios:
        try:
            r = scenario()
            results.append({"name": r.name, "passed": r.passed, "detail": r.detail})
        except Exception as e:
            results.append({"name": scenario.__name__, "passed": False, "detail": f"CRASHED: {e}"})

    print(json.dumps(results))


if __name__ == "__main__":
    main()
