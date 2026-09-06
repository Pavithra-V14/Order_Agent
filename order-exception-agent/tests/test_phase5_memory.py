"""
Phase 5 DoD: "A test customer with 3 synthetic past cases returns all 3,
correctly time-ordered, from a single memory query."
"""
import os
import tempfile
from datetime import datetime, timezone

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), "test_phase5.db")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}"

    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)
    db_module.init_db()

    yield db_module

    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_customer_history_returns_all_episodes_time_ordered(isolated_db):
    from app.memory.episodic import log_episode, get_customer_history

    db = isolated_db.SessionLocal()
    customer_id = "CUST-999"

    # seed 3 out of chronological order, to prove ordering is real, not incidental
    log_episode(db, customer_id, "case_resolved",
                {"exception_type": "return", "outcome": "approved"},
                occurred_at=datetime(2025, 3, 1, tzinfo=timezone.utc), case_id="case-1")
    log_episode(db, customer_id, "case_resolved",
                {"exception_type": "carrier", "outcome": "reshipped"},
                occurred_at=datetime(2025, 1, 10, tzinfo=timezone.utc), case_id="case-2")
    log_episode(db, customer_id, "case_resolved",
                {"exception_type": "payment", "outcome": "resolved"},
                occurred_at=datetime(2025, 2, 15, tzinfo=timezone.utc), case_id="case-3")

    history = get_customer_history(db, customer_id)

    assert len(history) == 3
    # most-recent-first ordering
    occurred_dates = [h["occurred_at"] for h in history]
    assert occurred_dates == sorted(occurred_dates, reverse=True)
    assert history[0]["case_id"] == "case-1"   # March -- most recent
    assert history[1]["case_id"] == "case-3"   # February
    assert history[2]["case_id"] == "case-2"   # January -- oldest
    db.close()


def test_customer_history_filters_by_episode_type(isolated_db):
    from app.memory.episodic import log_episode, get_customer_history

    db = isolated_db.SessionLocal()
    customer_id = "CUST-888"
    log_episode(db, customer_id, "case_resolved", {"exception_type": "return"},
                occurred_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    log_episode(db, customer_id, "fraud_flag_raised", {"reason": "address_change"},
                occurred_at=datetime(2025, 2, 1, tzinfo=timezone.utc))

    only_fraud = get_customer_history(db, customer_id, episode_type="fraud_flag_raised")
    assert len(only_fraud) == 1
    assert only_fraud[0]["episode_type"] == "fraud_flag_raised"
    db.close()


def test_history_does_not_leak_across_customers(isolated_db):
    from app.memory.episodic import log_episode, get_customer_history

    db = isolated_db.SessionLocal()
    log_episode(db, "CUST-A", "case_resolved", {"exception_type": "return"},
                occurred_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
    log_episode(db, "CUST-B", "case_resolved", {"exception_type": "return"},
                occurred_at=datetime(2025, 1, 1, tzinfo=timezone.utc))

    history_a = get_customer_history(db, "CUST-A")
    assert len(history_a) == 1
    db.close()


def test_risk_profile_summarizes_returns_and_fraud_flags_separately(isolated_db):
    """Per architecture doc's explicit warning: return count alone must
    NOT be conflated with fraud signal — the summary keeps them as
    separate fields so the Fraud/Risk Agent (Phase 6) can weight them
    independently rather than treating high-return-count as inherently risky."""
    from app.memory.episodic import log_episode, summarize_customer_risk_profile

    db = isolated_db.SessionLocal()
    customer_id = "CUST-HIGH-LTV"
    for i in range(5):
        log_episode(db, customer_id, "case_resolved",
                    {"exception_type": "return", "outcome": "approved"},
                    occurred_at=datetime(2025, 1, i + 1, tzinfo=timezone.utc))
    log_episode(db, customer_id, "fraud_flag_raised", {"reason": "test"},
                occurred_at=datetime(2025, 2, 1, tzinfo=timezone.utc))

    profile = summarize_customer_risk_profile(db, customer_id)
    assert profile["total_return_cases"] == 5
    assert profile["fraud_flags_raised"] == 1
    db.close()


# ---------------------------------------------------------------------------
# Summary Buffer (short-term, per-case) tests
# ---------------------------------------------------------------------------
def test_summary_buffer_keeps_recent_items_verbatim():
    from app.memory.summary_buffer import SummaryBuffer

    buf = SummaryBuffer(case_id="case-x", max_verbatim_items=3)
    for i in range(3):
        buf.add({"agent": "diagnosis", "summary": f"step {i}"})

    ctx = buf.get_context()
    assert len(ctx["recent_items"]) == 3
    assert ctx["running_summary"] == ""


def test_summary_buffer_folds_oldest_items_when_overflowing():
    from app.memory.summary_buffer import SummaryBuffer

    buf = SummaryBuffer(case_id="case-y", max_verbatim_items=2)
    for i in range(5):
        buf.add({"agent": "diagnosis", "summary": f"step {i}"})

    ctx = buf.get_context()
    assert len(ctx["recent_items"]) == 2   # bounded, per max_verbatim_items
    assert "step 0" in ctx["running_summary"]  # oldest items folded, not dropped
    assert "step 2" in ctx["running_summary"]
    # most recent 2 stay verbatim
    assert ctx["recent_items"][-1]["summary"] == "step 4"


def test_get_or_create_buffer_is_stable_per_case_id():
    from app.memory.summary_buffer import get_or_create_buffer, reset_buffers

    reset_buffers()
    buf1 = get_or_create_buffer("case-z")
    buf1.add({"agent": "diagnosis", "summary": "first step"})
    buf2 = get_or_create_buffer("case-z")  # same case_id -> same buffer instance
    assert buf2.get_context()["recent_items"][0]["summary"] == "first step"
    reset_buffers()
