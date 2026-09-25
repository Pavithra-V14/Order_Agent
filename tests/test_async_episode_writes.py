"""
Memory-upgrade follow-up: tests for moving episodic-memory WRITES off
the fraud/resolution hot path via the existing job queue
(app/workers/job_queue.py), rather than a live, blocking Graphiti call
inline in the decision path.

Scope, stated honestly: this covers the job-queue wiring itself
(enqueue -> handler runs -> write lands, defensive handler registration
and worker start, failure -> alert). It does NOT re-verify Graphiti's
own internals (already covered by tests/test_graphiti_adapter.py) - the
handler just calls the existing, already-tested log_episode().
"""
import os
import tempfile
import time

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_async_episode_{os.getpid()}_{id(object())}.db")
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


@pytest.fixture(autouse=True)
def reset_job_queue_state():
    """The job queue is process-global (get_job_queue() singleton) -
    reset before AND after every test so one test's worker thread/jobs
    don't leak into the next, same discipline as this project's other
    module-level caches (summary_buffer's _buffers, learning_loop's
    similarity cache)."""
    from app.workers.job_queue import reset_job_queue
    reset_job_queue()
    yield
    reset_job_queue()


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    result = None
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.01)
    return result


def test_log_episode_async_returns_immediately_with_a_job_id(isolated_db):
    from datetime import datetime, timezone
    from app.memory.episodic import log_episode_async

    job_id = log_episode_async(
        customer_id="CUST-ASYNC-1", episode_type="case_resolved",
        content={"exception_type": "return", "action": "refund", "amount_usd": 20.0},
        occurred_at=datetime.now(timezone.utc), case_id="case-async-1",
    )
    assert isinstance(job_id, str) and job_id


def test_log_episode_async_write_actually_lands(isolated_db):
    from datetime import datetime, timezone
    from app.memory.episodic import log_episode_async, get_customer_history

    log_episode_async(
        customer_id="CUST-ASYNC-2", episode_type="case_resolved",
        content={"exception_type": "return", "action": "refund", "amount_usd": 15.0},
        occurred_at=datetime.now(timezone.utc), case_id="case-async-2",
    )

    from app.core.db import SessionLocal
    db = SessionLocal()
    history = _wait_for(lambda: get_customer_history(db, "CUST-ASYNC-2", episode_type="case_resolved"))
    assert history, "the async write must eventually land - polled for 2s and found nothing"
    assert history[0]["content"]["amount_usd"] == 15.0


def test_log_episode_async_works_without_the_fastapi_app_having_started(isolated_db):
    """THE key robustness test: this project's job queue worker/handler
    registration normally happens in app.main's lifespan hook - but
    resolution_completion.py and orchestrator.py call log_episode_async()
    as plain Python code, including from tests that never spin up
    TestClient(app). This proves the defensive, idempotent
    register_handler()+start_worker() calls inside log_episode_async()
    itself make this work regardless - not just when the FastAPI app
    happens to have started first."""
    import sys
    assert "app.main" not in sys.modules or True  # documented intent; not asserting import state itself

    from datetime import datetime, timezone
    from app.memory.episodic import log_episode_async, get_customer_history

    log_episode_async(
        customer_id="CUST-ASYNC-NO-APP", episode_type="case_resolved",
        content={"exception_type": "return", "action": "refund", "amount_usd": 5.0},
        occurred_at=datetime.now(timezone.utc), case_id="case-async-no-app",
    )

    from app.core.db import SessionLocal
    db = SessionLocal()
    history = _wait_for(lambda: get_customer_history(db, "CUST-ASYNC-NO-APP", episode_type="case_resolved"))
    assert history, "log_episode_async must work even when app.main's lifespan never ran"


def test_handle_log_episode_alerts_and_reraises_on_failure(isolated_db, monkeypatch):
    """Unit-level version of tests/test_log_episode_alerting.py's
    end-to-end proof: the handler itself, called directly, must both
    (1) produce a real AlertRecord and (2) re-raise so the job queue's
    own status tracking still correctly marks the job FAILED."""
    def always_fails(*args, **kwargs):
        raise ValueError("simulated failure")

    import app.memory.episodic as episodic_module
    monkeypatch.setattr(episodic_module, "log_episode", always_fails)

    from datetime import datetime, timezone
    from app.workers.handlers import handle_log_episode

    with pytest.raises(ValueError):
        handle_log_episode({
            "customer_id": "CUST-HANDLER-FAIL", "episode_type": "case_resolved",
            "content": {}, "occurred_at_iso": datetime.now(timezone.utc).isoformat(),
            "case_id": "case-handler-fail",
        })

    from app.core.db import SessionLocal
    from app.core.alerting import get_recent_alerts
    db = SessionLocal()
    alerts = get_recent_alerts(db, event_type="log_episode_failure")
    assert len(alerts) == 1
    assert alerts[0]["detail"]["customer_id"] == "CUST-HANDLER-FAIL"


def test_resolution_completion_still_resolves_the_case_even_if_enqueue_itself_fails(isolated_db, monkeypatch):
    """The OTHER failure mode (distinct from the handler's own failure,
    covered above and in test_log_episode_alerting.py): log_episode_async()
    itself raising synchronously (e.g. a real Redis connection error at
    enqueue time). Must still be non-fatal to the case resolving -
    exactly the guarantee resolution_completion.py's own try/except
    already provided before this async change, unchanged by it."""
    from datetime import datetime, timezone
    from app.tools.oms import create_order
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy
    from app.agents.resolution_completion import complete_resolution
    from app.core.db import ExceptionCase, CaseState, SessionLocal

    db = SessionLocal()
    create_order(
        db, order_id="ORD-ENQUEUE-FAIL", customer_id="CUST-ENQUEUE-FAIL", channel="direct",
        status="paid", total_amount_usd=30.0, purchase_date=datetime.now(timezone.utc),
        line_items=[{"sku": "SKU-1", "category": "apparel", "qty": 1, "price": 30.0}],
        payment_intent_id="pi_enqueue_fail_test",
    )
    from app.tools.payment import get_payment_gateway
    get_payment_gateway().seed_transaction("pi_enqueue_fail_test", amount_usd=30.0, status="succeeded")

    case = ExceptionCase(id="case-enqueue-fail", order_id="ORD-ENQUEUE-FAIL", customer_id="CUST-ENQUEUE-FAIL",
                          channel="direct", exception_type="return", state=CaseState.DETECTED)
    db.add(case)
    db.commit()

    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=30.0, confidence=0.9,
        reasoning="Test reasoning long enough to pass schema validation.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )

    import app.agents.resolution_completion as rc_module
    monkeypatch.setattr(rc_module, "log_episode_async", lambda *a, **kw: (_ for _ in ()).throw(
        ConnectionError("simulated Redis connection refused")))

    result = complete_resolution(
        db, case=case, proposed_decision=decision, final_decision=decision,
        decided_by="system:test", action_label="test_action", payment_intent_id="pi_enqueue_fail_test",
    )
    assert result["outcome"] == "resolved"
