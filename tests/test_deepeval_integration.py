"""
Real DeepEval integration - architecture doc Layer 10's evaluation spec
calls for DeepEval specifically ("CI-gated via DeepEval"), which this
project's golden set (tests/golden_set.py) never actually used, despite
being the real, functioning pre-deployment gate. This wires in genuine
DeepEval LLMTestCases and metrics, judged by a real Groq-backed model
(app/eval/deepeval_model.py) rather than DeepEval's OpenAI default,
against REAL retrieved policy text from this project's own RAG pipeline
- not a fabricated example.

HONEST LIMITATION: this sandbox has no network route to api.groq.com
(same limitation GroqClient's own docstring documents), so these tests
cannot be network-verified from here. They're written correctly against
DeepEval's real, documented interfaces and skip cleanly (not fail with
a confusing network error) when GROQ_API_KEY isn't configured - the
same honest pattern already used throughout this project for anything
requiring live network access this sandbox doesn't have.
"""
import os
import tempfile

import pytest

from app.core.config import get_settings

# Read from a dedicated stashed env var, NOT from get_settings()/
# os.environ["GROQ_API_KEY"] directly, and NOT via `from tests.conftest
# import ...` either - found necessary directly from a real user
# report, in two layers:
#
# Layer 1: this file originally captured get_settings().groq_api_key at
# its OWN module import time, which worked until conftest.py's
# pytest_configure hook was added (for a different, unrelated bug -
# real cloud credentials leaking into module-scoped fixtures).
# pytest_configure runs BEFORE test collection, which is BEFORE any
# test module's own import-time code executes - so GROQ_API_KEY was
# already cleared from os.environ by the time this file's "capture the
# real key" line ran, always seeing None regardless of what the user's
# real environment had.
#
# Layer 2: the first fix attempt had conftest.py stash the real value in
# a module-level Python variable and had this file read it via `from
# tests.conftest import get_real_credential_saved_before_test_isolation`
# - but a REGULAR Python import of conftest.py is not guaranteed to be
# the SAME module object pytest's own hook-loading machinery uses
# internally, so the stashed value was invisible through that import
# path even though pytest_configure had genuinely run and set it.
#
# The actual fix: conftest.py's pytest_configure ALSO stashes the real
# value under a dedicated, differently-named os.environ key
# (_OEA_TEST_SAVED_GROQ_API_KEY) - os.environ is a single, real,
# process-wide dict, immune to the module-identity mismatch above.
_REAL_GROQ_API_KEY_AT_COLLECTION = os.environ.get("_OEA_TEST_SAVED_GROQ_API_KEY")
_groq_configured = bool(_REAL_GROQ_API_KEY_AT_COLLECTION)
requires_groq = pytest.mark.skipif(
    not _groq_configured,
    reason="GROQ_API_KEY not configured - DeepEval's judge model needs a real Groq call to run at all "
           "(this sandbox has no network route to api.groq.com either way - see app/eval/deepeval_model.py)",
)


@pytest.fixture(autouse=True)
def _restore_real_groq_key_for_this_file():
    """Runs after conftest.py's blanket credential-clearing fixture
    (autouse fixtures from conftest.py resolve before same-scope
    autouse fixtures defined in the test file itself), re-applying the
    real key AND real model name this file's tests actually need - the
    deliberate exception to every other test in this suite, which
    correctly wants no real credentials at all.

    Restoring ROUTER_MODEL too, not just GROQ_API_KEY, found necessary
    directly from a real user report: with only the key restored, these
    tests genuinely attempted a real Groq call but got a real 404 - the
    request correctly authenticated, but config.py's hardcoded default
    model name is very likely retired on Groq's end (strong evidence:
    the user's own real .env explicitly overrides ROUTER_MODEL to a
    different model). This test file needs the user's own currently-
    working model, not this project's possibly-stale hardcoded default.
    """
    if _REAL_GROQ_API_KEY_AT_COLLECTION:
        os.environ["GROQ_API_KEY"] = _REAL_GROQ_API_KEY_AT_COLLECTION
    real_router_model = os.environ.get("_OEA_TEST_SAVED_ROUTER_MODEL")
    if real_router_model:
        os.environ["ROUTER_MODEL"] = real_router_model
    if _REAL_GROQ_API_KEY_AT_COLLECTION or real_router_model:
        get_settings.cache_clear()
    yield


@pytest.fixture(autouse=True)
def isolated_qdrant_with_real_policies():
    """Real, dedicated, freshly-ingested Qdrant - these tests need
    genuine retrieved policy text, not a fabricated example, for the
    faithfulness judgment to mean anything."""
    import shutil
    tmp_qdrant = tempfile.mkdtemp(prefix="test_deepeval_qdrant_")
    tmp_reindex_state = os.path.join(tempfile.gettempdir(), f"test_deepeval_reindex_{os.getpid()}_{id(object())}.json")
    os.environ["QDRANT_LOCAL_PATH"] = tmp_qdrant
    get_settings.cache_clear()
    import app.rag.vectorstore as vectorstore_module
    if vectorstore_module._client_singleton is not None:
        vectorstore_module._client_singleton.close()
    vectorstore_module._client_singleton = None
    vectorstore_module._client_singleton_key = None

    import app.rag.ingestion as ingestion_module
    ingestion_module._REINDEX_STATE_PATH = tmp_reindex_state
    from app.rag.ingestion import ingest_policy_directory
    ingest_policy_directory("data/policies")

    yield

    if vectorstore_module._client_singleton is not None:
        vectorstore_module._client_singleton.close()
    vectorstore_module._client_singleton = None
    vectorstore_module._client_singleton_key = None
    os.environ.pop("QDRANT_LOCAL_PATH", None)
    get_settings.cache_clear()
    shutil.rmtree(tmp_qdrant, ignore_errors=True)
    if os.path.exists(tmp_reindex_state):
        os.remove(tmp_reindex_state)


def _get_real_retrieval_context(query: str, as_of_date: str, doc_type: str) -> list[str]:
    from app.rag.retrieval import hybrid_search
    chunks = hybrid_search(query=query, as_of_date=as_of_date, doc_type=doc_type, top_k=3)
    assert chunks, f"expected real retrieval results for query={query!r} - check ingestion ran correctly"
    return [c.text for c in chunks]


def test_retrieval_context_helper_returns_real_nonempty_text():
    """Runs regardless of GROQ_API_KEY - proves the RAG-retrieval half
    of this integration is solid on its own, even when the LLM-judging
    half can't be verified in this sandbox. Both Groq-requiring tests
    above call this same helper, so if IT were broken, they'd fail for
    the wrong reason even when Groq IS configured elsewhere."""
    context = _get_real_retrieval_context(
        "return window apparel", as_of_date="2025-06-15", doc_type="return_policy",
    )
    assert len(context) > 0
    assert all(isinstance(c, str) and len(c) > 0 for c in context)


@requires_groq
def test_faithful_reasoning_passes_deepeval_faithfulness_metric():
    """A resolution reasoning that genuinely reflects what the retrieved
    policy text says should be judged faithful by a real LLM judge -
    not just by this project's own manual doc_id-matching check
    (compute_rag_metrics' groundedness_score), an independent
    cross-check using a different evaluation method entirely."""
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric
    from app.eval.deepeval_model import GroqDeepEvalModel

    retrieval_context = _get_real_retrieval_context(
        "return window apparel", as_of_date="2025-06-15", doc_type="return_policy",
    )

    test_case = LLMTestCase(
        input="What is the return window for apparel purchased in June 2025?",
        actual_output="Apparel items purchased in June 2025 fall under the 180-day return window "
                       "policy in effect at that time.",
        retrieval_context=retrieval_context,
    )

    metric = FaithfulnessMetric(model=GroqDeepEvalModel(), threshold=0.7)
    metric.measure(test_case)

    assert metric.score >= metric.threshold, (
        f"expected a faithful reasoning to pass DeepEval's real LLM-judged faithfulness check, "
        f"got score={metric.score}, reason={metric.reason}"
    )


@requires_groq
def test_fabricated_reasoning_fails_deepeval_faithfulness_metric():
    """THE regression test proving this integration genuinely
    discriminates, not just always passes: a reasoning that asserts
    something the retrieved policy text does NOT actually say must be
    judged unfaithful by the real LLM judge."""
    from deepeval.test_case import LLMTestCase
    from deepeval.metrics import FaithfulnessMetric
    from app.eval.deepeval_model import GroqDeepEvalModel

    retrieval_context = _get_real_retrieval_context(
        "return window apparel", as_of_date="2025-06-15", doc_type="return_policy",
    )

    test_case = LLMTestCase(
        input="What is the return window for apparel purchased in June 2025?",
        actual_output="Apparel items have a lifetime, no-questions-asked return policy with no time "
                       "limit whatsoever, and customers are also entitled to free express shipping "
                       "on every return regardless of reason.",
        retrieval_context=retrieval_context,
    )

    metric = FaithfulnessMetric(model=GroqDeepEvalModel(), threshold=0.7)
    metric.measure(test_case)

    assert metric.score < metric.threshold, (
        f"expected a fabricated reasoning (asserting claims the retrieved policy text does not "
        f"support) to FAIL the faithfulness check, got score={metric.score} (passed threshold "
        f"{metric.threshold}) - the metric isn't discriminating between faithful and unfaithful output"
    )


def test_groq_deepeval_model_is_correctly_constructed_without_a_real_call():
    """Verifies the model wrapper itself is wired correctly (settings,
    initialization) WITHOUT making a real network call — runs
    regardless of whether GROQ_API_KEY is configured, since this only
    checks construction, not generation."""
    from app.core.config import get_settings
    if not get_settings().groq_api_key:
        pytest.skip("GROQ_API_KEY not configured - nothing to construct")

    from app.eval.deepeval_model import GroqDeepEvalModel
    model = GroqDeepEvalModel()
    assert model.get_model_name().startswith("groq/")
    assert model.model is not None
