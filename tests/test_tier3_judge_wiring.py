"""
Tests for Tier 3 judge wiring (app/guardrails/tier3_judge.py) - found
fully implemented during a direct audit but never actually called from
anywhere. Verifies the sampling trigger in
app/agents/resolution_completion.py and the background job handler in
app/workers/handlers.py both genuinely work, per architecture doc 8.6's
requirement that this stays async and sampled, never blocking execution.
"""
import os
import tempfile

import pytest


@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_tier3_{os.getpid()}_{id(object())}.db")
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


def test_handle_tier3_judge_sample_stores_a_passing_result(isolated_db):
    """A decision citing a genuinely known policy doc with substantive
    reasoning should pass cleanly and be recorded as such."""
    from app.workers.handlers import handle_tier3_judge_sample
    from app.core.db import Tier3JudgeResultRecord
    import app.rag.ingestion as ingestion_module

    payload = {
        "case_id": "case-tier3-pass-test",
        "decision": {
            "action": "refund", "amount_usd": 45.0, "confidence": 0.9,
            "reasoning": "Order delivered within the return window per the cited policy; approving a full refund.",
            "cited_policy": {"doc_id": "RET-POLICY-2025-A", "version": "1", "clause_summary": "return window"},
        },
    }

    # Reads and restores the module's CURRENT reindex-state path,
    # rather than assuming the literal "data/reindex_state.json" -
    # found necessary directly: several other tests in this suite
    # monkeypatch app.rag.ingestion._REINDEX_STATE_PATH to a dedicated
    # temp file for their own isolation, and this test genuinely ran
    # after one of those redirected it, silently checking a stale path
    # with no seeded content at all.
    real_path = ingestion_module._REINDEX_STATE_PATH
    original_content = None
    if os.path.exists(real_path):
        with open(real_path) as f:
            original_content = f.read()

    import json
    os.makedirs(os.path.dirname(real_path) or ".", exist_ok=True)
    with open(real_path, "w") as f:
        json.dump({"RET-POLICY-2025-A": {"hashes": {}}}, f)

    try:
        result = handle_tier3_judge_sample(payload)
        assert result["tier3_passed"] is True
        assert result["quality_score"] == 1.0

        db = isolated_db.SessionLocal()
        record = db.query(Tier3JudgeResultRecord).filter(
            Tier3JudgeResultRecord.case_id == "case-tier3-pass-test"
        ).first()
        assert record is not None
        assert record.passed is True
        db.close()
    finally:
        if original_content is not None:
            with open(real_path, "w") as f:
                f.write(original_content)
        elif os.path.exists(real_path):
            os.remove(real_path)


def test_handle_tier3_judge_sample_flags_unknown_policy_and_alerts(isolated_db):
    """THE regression test proving the actual point of Tier 3: a
    decision citing a policy doc_id that doesn't genuinely exist must
    be flagged, stored as failed, AND produce a real, queryable alert -
    not silently pass."""
    from app.workers.handlers import handle_tier3_judge_sample
    from app.core.db import Tier3JudgeResultRecord
    from app.core.alerting import get_recent_alerts
    import app.rag.ingestion as ingestion_module

    payload = {
        "case_id": "case-tier3-fail-test",
        "decision": {
            "action": "refund", "amount_usd": 45.0, "confidence": 0.9,
            "reasoning": "Approving per policy.",
            "cited_policy": {"doc_id": "MADE-UP-POLICY-DOES-NOT-EXIST", "version": "1", "clause_summary": "x"},
        },
    }

    real_path = ingestion_module._REINDEX_STATE_PATH
    original_content = None
    if os.path.exists(real_path):
        with open(real_path) as f:
            original_content = f.read()

    import json
    os.makedirs(os.path.dirname(real_path) or ".", exist_ok=True)
    with open(real_path, "w") as f:
        json.dump({"RET-POLICY-2025-A": {"hashes": {}}}, f)

    try:
        result = handle_tier3_judge_sample(payload)
        assert result["tier3_passed"] is False
        assert result["quality_score"] < 1.0
        assert any("does not match any known policy" in flag for flag in result["flags"])

        db = isolated_db.SessionLocal()
        record = db.query(Tier3JudgeResultRecord).filter(
            Tier3JudgeResultRecord.case_id == "case-tier3-fail-test"
        ).first()
        assert record is not None
        assert record.passed is False

        alerts = get_recent_alerts(db, event_type="tier3_judge_flagged")
        assert any(a["detail"].get("case_id") == "case-tier3-fail-test" for a in alerts)
        db.close()
    finally:
        if original_content is not None:
            with open(real_path, "w") as f:
                f.write(original_content)
        elif os.path.exists(real_path):
            os.remove(real_path)


def test_complete_resolution_enqueues_tier3_sample_sometimes(isolated_db, monkeypatch):
    """Proves the sampling trigger in complete_resolution() actually
    enqueues a real job — forces the 15% sample to fire deterministically
    by monkeypatching random.random(), rather than relying on chance."""
    from datetime import datetime, timezone
    from app.tools.oms import create_order
    from app.tools.payment import get_payment_gateway
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy
    from app.agents.resolution_completion import complete_resolution

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-TIER3-SAMPLE-TEST", customer_id="CUST-TIER3-SAMPLE-TEST", channel="direct",
        status="paid", total_amount_usd=45.0, purchase_date=datetime.now(timezone.utc),
        line_items=[{"sku": "SKU-TIER3-SAMPLE-TEST", "category": "apparel", "qty": 1, "price": 45.0}],
        payment_intent_id="pi_tier3_sample_test",
    )
    get_payment_gateway().seed_transaction("pi_tier3_sample_test", amount_usd=45.0, status="succeeded")

    case = isolated_db.ExceptionCase(
        id="case-tier3-sample-test", order_id="ORD-TIER3-SAMPLE-TEST", customer_id="CUST-TIER3-SAMPLE-TEST",
        channel="direct", exception_type="return", state=isolated_db.CaseState.DETECTED,
    )
    db.add(case)
    db.commit()

    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=45.0, confidence=0.9,
        reasoning="Test reasoning long enough to pass schema validation.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )

    enqueued = []

    class FakeJobQueue:
        def enqueue(self, job_type, payload):
            enqueued.append((job_type, payload))
            return "fake-job-id"

    import app.agents.resolution_completion as rc_module
    monkeypatch.setattr("app.workers.job_queue.get_job_queue", lambda: FakeJobQueue())
    monkeypatch.setattr("random.random", lambda: 0.01)  # forces the 15% sample to fire

    complete_resolution(
        db, case=case, proposed_decision=decision, final_decision=decision,
        decided_by="system:test", action_label="test_action",
        payment_intent_id="pi_tier3_sample_test",
    )

    assert len(enqueued) == 1
    assert enqueued[0][0] == "tier3_judge_sample"
    assert enqueued[0][1]["case_id"] == "case-tier3-sample-test"
    db.close()


def test_complete_resolution_does_not_enqueue_tier3_sample_outside_sample_rate(isolated_db, monkeypatch):
    """The other half: when the random draw is OUTSIDE the sample rate,
    nothing should be enqueued at all — proving this is genuinely
    sampled, not always-on."""
    from datetime import datetime, timezone
    from app.tools.oms import create_order
    from app.tools.payment import get_payment_gateway
    from app.guardrails.schema import ResolutionDecision, ResolutionAction, CitedPolicy
    from app.agents.resolution_completion import complete_resolution

    db = isolated_db.SessionLocal()
    create_order(
        db, order_id="ORD-TIER3-NOSAMPLE-TEST", customer_id="CUST-TIER3-NOSAMPLE-TEST", channel="direct",
        status="paid", total_amount_usd=45.0, purchase_date=datetime.now(timezone.utc),
        line_items=[{"sku": "SKU-TIER3-NOSAMPLE-TEST", "category": "apparel", "qty": 1, "price": 45.0}],
        payment_intent_id="pi_tier3_nosample_test",
    )
    get_payment_gateway().seed_transaction("pi_tier3_nosample_test", amount_usd=45.0, status="succeeded")

    case = isolated_db.ExceptionCase(
        id="case-tier3-nosample-test", order_id="ORD-TIER3-NOSAMPLE-TEST", customer_id="CUST-TIER3-NOSAMPLE-TEST",
        channel="direct", exception_type="return", state=isolated_db.CaseState.DETECTED,
    )
    db.add(case)
    db.commit()

    decision = ResolutionDecision(
        action=ResolutionAction.REFUND, amount_usd=45.0, confidence=0.9,
        reasoning="Test reasoning long enough to pass schema validation.",
        cited_policy=CitedPolicy(doc_id="RET-POLICY-2025-A", version="1", clause_summary="return window"),
    )

    enqueued = []

    class FakeJobQueue:
        def enqueue(self, job_type, payload):
            enqueued.append((job_type, payload))
            return "fake-job-id"

    monkeypatch.setattr("app.workers.job_queue.get_job_queue", lambda: FakeJobQueue())
    monkeypatch.setattr("random.random", lambda: 0.99)  # outside the 15% sample rate

    complete_resolution(
        db, case=case, proposed_decision=decision, final_decision=decision,
        decided_by="system:test", action_label="test_action",
        payment_intent_id="pi_tier3_nosample_test",
    )

    assert len(enqueued) == 0
    db.close()
