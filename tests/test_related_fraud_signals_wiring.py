"""
Stage 1 memory upgrade — end-to-end wiring tests for the cross-customer
fraud signal, from run_fraud_risk_agent (app/agents/workflow_agents.py)
down through FakeLLMClient's rule-based scoring (app/agents/llm_client.py).

Scope, stated honestly (same discipline as tests/test_graphiti_adapter.py):
find_related_fraud_signals itself is mocked here — its own real behavior
(Cypher query shape, param correctness, error handling) is covered
separately in tests/test_graphith_adapter.py. What THIS file verifies is
the wiring: does run_fraud_risk_agent actually call it with the right
arguments, does a match actually change the risk assessment, does a
lookup failure degrade gracefully, and is the order_id parameter fully
backward-compatible (existing callers that don't pass it are unaffected).
"""
import os
import tempfile
from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    """Rebinds SessionLocal/engine on the SAME app.core.db module object,
    rather than importlib.reload()-ing it - matches
    tests/test_summary_buffer_wiring.py's pattern, not
    tests/test_phase6_agents.py's. Deliberate: this file's tests
    construct ExceptionCase/AuditLogEntry directly AND call into
    app.agents.orchestrator, which imports those same classes at ITS
    own module-level import time. A reload() creates a SECOND,
    independent declarative registry generation - if orchestrator.py
    (or any other already-imported module) was first imported before
    the reload, it keeps holding the OLD generation's classes while a
    freshly-reloaded db_module hands out the NEW generation, and mixing
    the two within one SQLAlchemy mapper configuration pass raises
    InvalidRequestError. Confirmed as the actual mechanism by reproducing
    it directly with the reload-based pattern before switching to this
    one - not assumed."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_related_fraud_{os.getpid()}_{id(object())}.db")
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


def _seed_order(db_module, order_id, customer_id, payment_fingerprint=None):
    from app.tools.oms import create_order
    db = db_module.SessionLocal()
    create_order(
        db, order_id=order_id, customer_id=customer_id, channel="direct",
        status="paid", total_amount_usd=100.0,
        purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
        line_items=[{"sku": "SKU-1", "category": "electronics", "qty": 1, "price": 100.0}],
        payment_fingerprint=payment_fingerprint,
    )
    return db


# --- compute_payment_fingerprint -------------------------------------------

def test_compute_payment_fingerprint_is_deterministic():
    from app.tools.oms import compute_payment_fingerprint
    a = compute_payment_fingerprint("Visa", "4242")
    b = compute_payment_fingerprint("visa", "4242")  # case-insensitive brand
    assert a == b
    assert a is not None
    different = compute_payment_fingerprint("Mastercard", "4242")
    assert different != a


def test_compute_payment_fingerprint_none_when_incomplete():
    from app.tools.oms import compute_payment_fingerprint
    assert compute_payment_fingerprint(None, "4242") is None
    assert compute_payment_fingerprint("Visa", None) is None
    assert compute_payment_fingerprint(None, None) is None


def test_compute_payment_fingerprint_never_contains_raw_card_number():
    """A real, non-trivial guarantee this signal must hold: the output
    must not simply be the input concatenated - it's a real hash, not an
    encoding, so a stored fingerprint can never be reversed to recover
    the last4 it was built from."""
    from app.tools.oms import compute_payment_fingerprint
    fp = compute_payment_fingerprint("Visa", "4242")
    assert "4242" not in fp
    assert "visa" not in fp.lower()


# --- run_fraud_risk_agent wiring --------------------------------------------

def test_fraud_agent_unaffected_when_order_id_not_given(isolated_db):
    """Backward compatibility: existing callers (tests/golden_set.py)
    never pass order_id at all - must behave exactly as before."""
    from app.agents.llm_client import FakeLLMClient
    from app.agents.workflow_agents import run_fraud_risk_agent

    db = isolated_db.SessionLocal()
    result = run_fraud_risk_agent(db, FakeLLMClient(), customer_id="CUST-NO-ORDER")

    assert result["related_fraud_signals"] == []
    assert result["flag"] is False


def test_fraud_agent_unaffected_when_order_has_no_fingerprint(isolated_db):
    db = _seed_order(isolated_db, "ORD-NO-FP", "CUST-NO-FP", payment_fingerprint=None)

    from app.agents.llm_client import FakeLLMClient
    from app.agents.workflow_agents import run_fraud_risk_agent
    result = run_fraud_risk_agent(db, FakeLLMClient(), customer_id="CUST-NO-FP", order_id="ORD-NO-FP")

    assert result["related_fraud_signals"] == []
    assert result["flag"] is False


def test_fraud_agent_calls_find_related_fraud_signals_with_correct_args(isolated_db, monkeypatch):
    db = _seed_order(isolated_db, "ORD-WITH-FP", "CUST-UNDER-REVIEW", payment_fingerprint="fp_shared_123")

    captured = {}

    def fake_find_related_fraud_signals(payment_fingerprint, exclude_customer_id):
        captured["payment_fingerprint"] = payment_fingerprint
        captured["exclude_customer_id"] = exclude_customer_id
        return []

    monkeypatch.setattr(
        "app.memory.graphiti_adapter.find_related_fraud_signals", fake_find_related_fraud_signals,
    )

    from app.agents.llm_client import FakeLLMClient
    from app.agents.workflow_agents import run_fraud_risk_agent
    run_fraud_risk_agent(db, FakeLLMClient(), customer_id="CUST-UNDER-REVIEW", order_id="ORD-WITH-FP")

    assert captured["payment_fingerprint"] == "fp_shared_123"
    assert captured["exclude_customer_id"] == "CUST-UNDER-REVIEW"


def test_fraud_agent_flags_when_related_fraud_signal_found(isolated_db, monkeypatch):
    """THE core Stage 1 acceptance test: a genuine cross-customer match
    must actually change the fraud agent's real output - not just be
    computed and silently discarded."""
    db = _seed_order(isolated_db, "ORD-MATCH", "CUST-UNDER-REVIEW", payment_fingerprint="fp_shared_123")

    def fake_find_related_fraud_signals(payment_fingerprint, exclude_customer_id):
        return [{"customer_id": "CUST-KNOWN-FRAUD", "fraud_episode_id": "ep-1", "flagged_at": "2025-05-01T00:00:00Z"}]

    monkeypatch.setattr(
        "app.memory.graphiti_adapter.find_related_fraud_signals", fake_find_related_fraud_signals,
    )

    from app.agents.llm_client import FakeLLMClient
    from app.agents.workflow_agents import run_fraud_risk_agent
    result = run_fraud_risk_agent(db, FakeLLMClient(), customer_id="CUST-UNDER-REVIEW", order_id="ORD-MATCH")

    assert result["flag"] is True
    assert result["risk_score"] >= 0.6
    assert len(result["related_fraud_signals"]) == 1
    assert any("CUST-KNOWN-FRAUD" in r for r in result["reasons"])


def test_fraud_agent_degrades_gracefully_when_lookup_fails(isolated_db, monkeypatch):
    """A genuinely broken/misconfigured Neo4j must never break fraud
    scoring - the agent should behave exactly as if no fingerprint
    existed, not raise."""
    db = _seed_order(isolated_db, "ORD-BROKEN-LOOKUP", "CUST-1", payment_fingerprint="fp_abc")

    def broken_find_related_fraud_signals(payment_fingerprint, exclude_customer_id):
        raise RuntimeError("Neo4j connection refused")

    monkeypatch.setattr(
        "app.memory.graphiti_adapter.find_related_fraud_signals", broken_find_related_fraud_signals,
    )

    from app.agents.llm_client import FakeLLMClient
    from app.agents.workflow_agents import run_fraud_risk_agent
    # Must not raise.
    result = run_fraud_risk_agent(db, FakeLLMClient(), customer_id="CUST-1", order_id="ORD-BROKEN-LOOKUP")
    assert result["related_fraud_signals"] == []
    assert result["flag"] is False


# --- FakeLLMClient.assess_fraud_risk rule logic -----------------------------

def test_fake_llm_client_weighs_related_fraud_signals_heavily():
    from app.agents.llm_client import FakeLLMClient
    llm = FakeLLMClient()

    without_signal = llm.assess_fraud_risk(
        {"address_changed_same_day": False, "related_fraud_signals": []},
        {"fraud_flags_raised": 0, "total_return_cases": 0},
    )
    with_signal = llm.assess_fraud_risk(
        {"address_changed_same_day": False, "related_fraud_signals": [
            {"customer_id": "CUST-X", "fraud_episode_id": "ep-1", "flagged_at": "2025-01-01"},
        ]},
        {"fraud_flags_raised": 0, "total_return_cases": 0},
    )

    assert without_signal["flag"] is False
    assert with_signal["flag"] is True
    assert with_signal["risk_score"] > without_signal["risk_score"]
    assert any("CUST-X" in r for r in with_signal["reasons"])


def test_fake_llm_client_handles_missing_related_fraud_signals_key():
    """case_context without the key at all (e.g. an older caller that
    predates this change) must not raise - defaults to no signal."""
    from app.agents.llm_client import FakeLLMClient
    llm = FakeLLMClient()
    result = llm.assess_fraud_risk(
        {"address_changed_same_day": False},
        {"fraud_flags_raised": 0, "total_return_cases": 0},
    )
    assert result["flag"] is False


# --- orchestrator: fraud_flag_raised episode logging (real gap fix) --------

def _seed_case_and_order(db_module, case_id, order_id, customer_id):
    from app.tools.oms import create_order
    from app.core.db import ExceptionCase, CaseState

    db = db_module.SessionLocal()
    create_order(
        db, order_id=order_id, customer_id=customer_id, channel="direct",
        status="paid", total_amount_usd=100.0,
        purchase_date=datetime(2025, 6, 15, tzinfo=timezone.utc),
        line_items=[{"sku": "SKU-1", "category": "electronics", "qty": 1, "price": 100.0}],
        payment_fingerprint="fp_case_test",
    )
    case = ExceptionCase(id=case_id, order_id=order_id, customer_id=customer_id,
                          channel="direct", exception_type="fraud", state=CaseState.DIAGNOSING)
    db.add(case)
    db.commit()
    return db


def test_aggregate_node_logs_fraud_flag_raised_episode_when_flagged(isolated_db):
    """THE regression test for a real, previously-existing gap found
    while building this: 'fraud_flag_raised' was referenced by
    app/memory/episodic.py's counting logic and covered by
    app/memory/eval.py's golden set, but nothing in the live pipeline
    ever actually wrote one - summarize_customer_risk_profile()'s
    fraud_flags_raised count was therefore guaranteed to read 0 for
    every real customer. This proves aggregate_node now actually writes
    it when fraud.flag is True.

    The write is now enqueued via log_episode_async() (memory-upgrade
    follow-up moving it off the fraud-decision hot path - see
    app/workers/handlers.py's handle_log_episode()), not written
    synchronously inline - so this polls briefly for the async job to
    actually land, same "ack fast, process async" pattern
    tests/test_phase12_api.py's webhook tests already use."""
    db = _seed_case_and_order(isolated_db, "CASE-FRAUD-LOG", "ORD-FRAUD-LOG", "CUST-FRAUD-LOG")

    from app.agents.orchestrator import make_aggregate_node
    aggregate_node = make_aggregate_node()
    aggregate_node({
        "case_id": "CASE-FRAUD-LOG", "order_id": "ORD-FRAUD-LOG", "customer_id": "CUST-FRAUD-LOG",
        "diagnosis_findings": {}, "diagnosis_root_causes": [], "diagnosis_terminated_reason": "concluded",
        "fraud_result": {"risk_score": 0.9, "flag": True, "reasons": ["test reason"]},
        "inventory_result": {}, "customer_context_result": {},
    })

    import time
    from app.memory.episodic import get_customer_history
    deadline = time.monotonic() + 2.0
    history = []
    while time.monotonic() < deadline:
        history = get_customer_history(db, "CUST-FRAUD-LOG", episode_type="fraud_flag_raised")
        if history:
            break
        time.sleep(0.01)

    assert len(history) == 1
    assert history[0]["content"]["payment_fingerprint"] == "fp_case_test"
    assert history[0]["content"]["risk_score"] == 0.9


def test_aggregate_node_does_not_log_episode_when_not_flagged(isolated_db):
    db = _seed_case_and_order(isolated_db, "CASE-NO-FLAG", "ORD-NO-FLAG", "CUST-NO-FLAG")

    from app.agents.orchestrator import make_aggregate_node
    aggregate_node = make_aggregate_node()
    aggregate_node({
        "case_id": "CASE-NO-FLAG", "order_id": "ORD-NO-FLAG", "customer_id": "CUST-NO-FLAG",
        "diagnosis_findings": {}, "diagnosis_root_causes": [], "diagnosis_terminated_reason": "concluded",
        "fraud_result": {"risk_score": 0.1, "flag": False, "reasons": []},
        "inventory_result": {}, "customer_context_result": {},
    })

    from app.memory.episodic import get_customer_history
    history = get_customer_history(db, "CUST-NO-FLAG", episode_type="fraud_flag_raised")
    assert len(history) == 0
