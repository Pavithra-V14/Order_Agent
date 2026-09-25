"""
Stage 2 memory upgrade tests: real LLM summarization for the Summary
Buffer (app/memory/summary_buffer.py), DB persistence
(ExceptionCase.working_memory_summary), and eviction on case
resolution.

Scope, stated honestly (same discipline as tests/test_graphiti_adapter.py
and tests/test_related_fraud_signals_wiring.py): GroqClient/LiteLLMClient's
summarize_context() is verified by CONTRACT (correct system prompt used,
correct request/response shape) via a mocked _chat_json, not against a
real Groq endpoint - this sandbox has no route to api.groq.com. What's
verified for real: FakeLLMClient's deterministic fold (genuine, run
directly, no mocking), SummaryBuffer.add()'s llm-argument wiring and its
fallback-on-failure behavior, and the persist/load/evict round trip
against a real (if temporary) SQLite database.
"""
import os
import tempfile
from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    """Same non-reload pattern as tests/test_summary_buffer_wiring.py and
    (after Stage 1's fix) tests/test_related_fraud_signals_wiring.py -
    rebinds SessionLocal/engine on the SAME app.core.db module rather
    than importlib.reload()-ing it, to avoid the declarative-registry
    collision documented in that file's fixture docstring."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_summary_buffer_stage2_{os.getpid()}_{id(object())}.db")
    import app.core.db as db_module
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    fresh_engine = create_engine(f"sqlite:///{tmp_path}", connect_args={"check_same_thread": False})
    db_module.Base.metadata.create_all(bind=fresh_engine)
    db_module.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=fresh_engine)
    db_module.engine = fresh_engine

    yield db_module

    fresh_engine.dispose()
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except PermissionError:
        pass


def _seed_case(db_module, case_id, order_id="ORD-SB-STAGE2", customer_id="CUST-SB-STAGE2"):
    from app.core.db import ExceptionCase, CaseState
    db = db_module.SessionLocal()
    case = ExceptionCase(id=case_id, order_id=order_id, customer_id=customer_id,
                          channel="direct", exception_type="return", state=CaseState.DIAGNOSING)
    db.add(case)
    db.commit()
    return db


# --- FakeLLMClient.summarize_context -----------------------------------

def test_fake_llm_client_summarize_context_matches_old_stub_behavior():
    """Regression guard: FakeLLMClient's summarize_context must produce
    EXACTLY the same output the old private _summarize stub did, so
    every caller relying on the deterministic fallback (no llm given, or
    a real call failing) sees identical behavior to before Stage 2."""
    from app.agents.llm_client import FakeLLMClient
    llm = FakeLLMClient()

    first = llm.summarize_context("", {"agent": "diagnosis", "summary": "step 0"})
    assert first == "[diagnosis] step 0"

    second = llm.summarize_context(first, {"agent": "diagnosis", "summary": "step 1"})
    assert second == "[diagnosis] step 0; [diagnosis] step 1"


# --- LiteLLMClient/GroqClient.summarize_context (mocked contract) ------

def test_litellm_client_summarize_context_sends_correct_prompt_and_parses_response(monkeypatch):
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    captured = {}

    def fake_chat_json(self, system_prompt, user_prompt, purpose=None):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return {"summary": "folded summary text"}

    from app.agents.llm_client import LiteLLMClient
    monkeypatch.setattr(LiteLLMClient, "_chat_json", fake_chat_json)

    client = LiteLLMClient()
    result = client.summarize_context("existing summary", {"agent": "diagnosis", "summary": "step 3"})

    assert result == "folded summary text"
    assert "running summary" in captured["system_prompt"].lower()
    assert "existing summary" in captured["user_prompt"]
    assert "step 3" in captured["user_prompt"]

    os.environ.pop("GROQ_API_KEY", None)
    get_settings.cache_clear()


def test_litellm_client_summarize_context_falls_back_to_existing_summary_on_malformed_response(monkeypatch):
    """A real LLM response missing the expected "summary" key must not
    crash - result.get("summary", existing_summary) degrades to
    returning the existing summary unchanged, same non-fatal discipline
    used throughout this project's real-LLM call sites."""
    os.environ["GROQ_API_KEY"] = "gsk_fake_test_key"
    from app.core.config import get_settings
    get_settings.cache_clear()

    from app.agents.llm_client import LiteLLMClient
    monkeypatch.setattr(LiteLLMClient, "_chat_json", lambda self, s, u, purpose=None: {"unexpected_key": "oops"})

    client = LiteLLMClient()
    result = client.summarize_context("existing summary", {"agent": "diagnosis", "summary": "step 3"})
    assert result == "existing summary"

    os.environ.pop("GROQ_API_KEY", None)
    get_settings.cache_clear()


# --- SummaryBuffer.add() with llm wiring --------------------------------

def test_summary_buffer_add_without_llm_uses_deterministic_fold_unchanged():
    """Backward compatibility: existing direct callers
    (tests/test_phase5_memory.py) that never pass llm must see IDENTICAL
    behavior to before Stage 2."""
    from app.memory.summary_buffer import SummaryBuffer
    buf = SummaryBuffer(case_id="case-no-llm", max_verbatim_items=2)
    for i in range(4):
        buf.add({"agent": "diagnosis", "summary": f"step {i}"})
    ctx = buf.get_context()
    assert "step 0" in ctx["running_summary"]
    assert "step 1" in ctx["running_summary"]


def test_summary_buffer_add_with_llm_uses_real_summarize_context():
    from app.memory.summary_buffer import SummaryBuffer

    class StubLLM:
        def summarize_context(self, existing_summary, new_item):
            return f"STUBBED[{existing_summary}|{new_item['summary']}]"

    buf = SummaryBuffer(case_id="case-with-llm", max_verbatim_items=1)
    buf.add({"agent": "diagnosis", "summary": "step 0"}, llm=StubLLM())
    buf.add({"agent": "diagnosis", "summary": "step 1"}, llm=StubLLM())

    assert buf.running_summary == "STUBBED[|step 0]"


def test_summary_buffer_add_falls_back_when_llm_summarize_context_raises():
    """A broken/unreachable real LLM must not lose the diagnosis step or
    raise - falls back to the deterministic fold, same as llm=None."""
    from app.memory.summary_buffer import SummaryBuffer

    class BrokenLLM:
        def summarize_context(self, existing_summary, new_item):
            raise RuntimeError("Groq unreachable")

    buf = SummaryBuffer(case_id="case-broken-llm", max_verbatim_items=1)
    buf.add({"agent": "diagnosis", "summary": "step 0"}, llm=BrokenLLM())
    buf.add({"agent": "diagnosis", "summary": "step 1"}, llm=BrokenLLM())

    assert "step 0" in buf.running_summary  # deterministic fallback still ran


# --- persist_buffer_state / load_buffer_state ---------------------------

def test_persist_and_load_buffer_state_round_trip(isolated_db):
    db = _seed_case(isolated_db, "case-persist-test")

    from app.memory.summary_buffer import SummaryBuffer, persist_buffer_state, load_buffer_state
    buf = SummaryBuffer(case_id="case-persist-test")
    buf.add({"agent": "diagnosis", "summary": "step 0"})
    buf.add({"agent": "diagnosis", "summary": "step 1"})
    persist_buffer_state(db, "case-persist-test", buf)

    loaded = load_buffer_state(db, "case-persist-test")
    assert loaded.case_id == "case-persist-test"
    assert loaded.running_summary == buf.running_summary
    assert loaded.verbatim_items == buf.verbatim_items


def test_persist_buffer_state_is_a_noop_without_a_real_case_row(isolated_db):
    """A case_id with no matching ExceptionCase row (common when
    run_diagnosis() is called directly in tests without first creating a
    case) must not raise - documented no-op."""
    db = isolated_db.SessionLocal()
    from app.memory.summary_buffer import SummaryBuffer, persist_buffer_state
    buf = SummaryBuffer(case_id="case-does-not-exist")
    buf.add({"agent": "diagnosis", "summary": "step 0"})
    persist_buffer_state(db, "case-does-not-exist", buf)  # must not raise


def test_load_buffer_state_returns_empty_buffer_when_nothing_saved_yet(isolated_db):
    db = _seed_case(isolated_db, "case-nothing-saved")
    from app.memory.summary_buffer import load_buffer_state
    loaded = load_buffer_state(db, "case-nothing-saved")
    assert loaded.running_summary == ""
    assert loaded.verbatim_items == []


# --- diagnosis_agent integration: real persistence during a real run ---

def test_run_diagnosis_persists_buffer_state_to_the_case_row(isolated_db):
    """THE end-to-end proof: a real run_diagnosis() call against a real
    ExceptionCase row must leave working_memory_summary genuinely
    populated on that row - not just returned in DiagnosisResult, but
    actually durable in the database."""
    from app.tools.oms import create_order
    from app.tools.wms import seed_stock
    from app.agents.diagnosis_agent import run_diagnosis
    from app.agents.llm_client import FakeLLMClient
    from app.core.db import ExceptionCase

    db = _seed_case(isolated_db, "case-persist-integration", order_id="ORD-PERSIST-INTEGRATION")
    create_order(
        db, order_id="ORD-PERSIST-INTEGRATION", customer_id="CUST-SB-STAGE2", channel="direct",
        status="paid", total_amount_usd=45.0, purchase_date=datetime.now(timezone.utc),
        line_items=[{"sku": "SKU-PERSIST", "category": "apparel", "qty": 1, "price": 45.0}],
    )
    seed_stock(db, sku="SKU-PERSIST", warehouse="WH-A", on_hand_qty=5, sellable_qty=5)

    run_diagnosis(db, FakeLLMClient(), order_id="ORD-PERSIST-INTEGRATION", case_id="case-persist-integration")

    case = db.get(ExceptionCase, "case-persist-integration")
    assert case.working_memory_summary is not None
    assert case.working_memory_summary["case_id"] == "case-persist-integration"
    assert len(case.working_memory_summary["recent_items"]) > 0


# --- eviction on resolution ----------------------------------------------

def test_evict_buffer_removes_in_process_entry():
    from app.memory.summary_buffer import get_or_create_buffer, evict_buffer, _buffers, reset_buffers
    reset_buffers()
    get_or_create_buffer("case-to-evict")
    assert "case-to-evict" in _buffers
    evict_buffer("case-to-evict")
    assert "case-to-evict" not in _buffers
    reset_buffers()


def test_evict_buffer_is_a_noop_for_an_unbuffered_case():
    from app.memory.summary_buffer import evict_buffer
    evict_buffer("case-never-buffered")  # must not raise


# --- True TTL eviction (follow-up: evict_buffer() alone is event- ------
# --- triggered on resolution only, never fires for an abandoned case) --

def test_evict_stale_buffers_removes_only_buffers_older_than_max_age():
    from app.memory.summary_buffer import get_or_create_buffer, evict_stale_buffers, reset_buffers
    from datetime import datetime, timedelta, timezone
    reset_buffers()

    fresh = get_or_create_buffer("case-fresh")
    fresh.add({"agent": "diagnosis", "summary": "just happened"})

    stale = get_or_create_buffer("case-stale")
    stale.add({"agent": "diagnosis", "summary": "long ago"})
    stale.last_touched_at = datetime.now(timezone.utc) - timedelta(hours=48)

    evicted_count = evict_stale_buffers(max_age_seconds=3600)

    assert evicted_count == 1
    from app.memory.summary_buffer import _buffers
    assert "case-fresh" in _buffers
    assert "case-stale" not in _buffers
    reset_buffers()


def test_evict_stale_buffers_is_a_noop_when_nothing_is_stale():
    from app.memory.summary_buffer import get_or_create_buffer, evict_stale_buffers, reset_buffers
    reset_buffers()
    get_or_create_buffer("case-fresh-1").add({"agent": "diagnosis", "summary": "recent"})
    assert evict_stale_buffers(max_age_seconds=3600) == 0
    reset_buffers()


def test_add_updates_last_touched_at():
    from app.memory.summary_buffer import SummaryBuffer
    from datetime import datetime, timedelta, timezone
    buf = SummaryBuffer(case_id="case-touch-test")
    old_touch = datetime.now(timezone.utc) - timedelta(hours=1)
    buf.last_touched_at = old_touch
    buf.add({"agent": "diagnosis", "summary": "new activity"})
    assert buf.last_touched_at > old_touch


def test_evict_stale_buffers_endpoint_actually_evicts(isolated_db):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.memory.summary_buffer import get_or_create_buffer, reset_buffers, _buffers
    from datetime import datetime, timedelta, timezone

    reset_buffers()
    stale = get_or_create_buffer("case-endpoint-stale")
    stale.add({"agent": "diagnosis", "summary": "old"})
    stale.last_touched_at = datetime.now(timezone.utc) - timedelta(hours=48)

    with TestClient(app) as client:
        resp = client.post("/api/v1/admin/evict-stale-buffers?max_age_seconds=3600")
        assert resp.status_code == 200
        assert resp.json()["evicted_count"] == 1

    assert "case-endpoint-stale" not in _buffers
    reset_buffers()
