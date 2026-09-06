"""
Phase 9 DoD: "The proposal-not-auto-apply behavior is verified by test -
this is the single guardrail most likely to be accidentally skipped
under time pressure, so verify it explicitly."

Also: "manually seed 10 human decisions in one similarity cluster with 0%
overturn, confirm the batch job proposes raising tau for that cluster
(and does NOT apply it automatically)."
"""
import os
import tempfile

import pytest

@pytest.fixture(autouse=True)
def isolated_db():
    tmp_path = os.path.join(tempfile.gettempdir(), f"test_phase9_{os.getpid()}_{id(object())}.db")
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_path}"

    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)
    db_module.init_db()

    yield db_module

    try:

        if os.path.exists(tmp_path):

            os.remove(tmp_path)

    except PermissionError:

        pass  # Windows may still hold a brief lock from engine cleanup; harmless to leave a stray temp file

def _seed_case_and_pattern_entry(db_module, db, case_num, cluster_key, matched):
    from app.core.db import ExceptionCase, CaseState
    from app.agents.learning_loop import record_resolution_outcome

    case_id = f"case-p9-{cluster_key}-{case_num}"
    case = ExceptionCase(id=case_id, order_id=f"ORD-{case_num}", customer_id="CUST-P9",
                          channel="direct", exception_type="return", state=CaseState.RESOLVED)
    db.add(case)
    db.commit()

    proposed = {"action": "refund", "amount_usd": 25.0}
    final = dict(proposed) if matched else {"action": "refund", "amount_usd": 15.0}

    record_resolution_outcome(
        db, case_id=case_id, cluster_key=cluster_key,
        case_feature_summary=f"Low-value return request, category apparel, case {case_num}",
        agent_proposed_resolution=proposed, human_final_resolution=final,
    )

def test_ten_zero_overturn_cases_propose_raise_but_never_auto_apply(isolated_db):
    from app.agents.learning_loop import propose_threshold_adjustments, get_active_threshold
    from app.core.db import ThresholdOverrideRecord

    db = isolated_db.SessionLocal()
    for i in range(10):
        _seed_case_and_pattern_entry(isolated_db, db, i, "return_low_value", True)

    current_threshold = 0.90
    proposals = propose_threshold_adjustments(db, current_threshold=current_threshold, min_sample_size=5)

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.cluster_key == "return_low_value"
    assert proposal.sample_size == 10
    assert proposal.overturn_rate == 0.0
    assert proposal.proposed_threshold > current_threshold, "0% overturn should propose RAISING the threshold"
    assert proposal.status == "pending_review"

    override = db.get(ThresholdOverrideRecord, "return_low_value")
    assert override is None, "A proposal existing must have ZERO effect until a human explicitly accepts it"

    active = get_active_threshold(db, "return_low_value", default_threshold=current_threshold)
    assert active == current_threshold, "The effective threshold must still be the original default, unchanged"
    db.close()

def test_explicit_human_accept_is_required_to_create_an_override(isolated_db):
    from app.agents.learning_loop import propose_threshold_adjustments, accept_threshold_proposal, get_active_threshold

    db = isolated_db.SessionLocal()
    for i in range(10):
        _seed_case_and_pattern_entry(isolated_db, db, i, "return_low_value", True)

    proposals = propose_threshold_adjustments(db, current_threshold=0.90, min_sample_size=5)
    proposal_id = proposals[0].id

    override = accept_threshold_proposal(db, proposal_id, accepted_by="human:jane@company.com")
    assert override.active_threshold == proposals[0].proposed_threshold
    assert override.accepted_by == "human:jane@company.com"

    active = get_active_threshold(db, "return_low_value", default_threshold=0.90)
    assert active == proposals[0].proposed_threshold, "After explicit accept, the override IS now effective"
    db.close()

def test_accept_rejects_system_or_automated_identity(isolated_db):
    """The guardrail's other half: even the accept function itself refuses
    to be called on the system's own behalf."""
    from app.agents.learning_loop import propose_threshold_adjustments, accept_threshold_proposal

    db = isolated_db.SessionLocal()
    for i in range(10):
        _seed_case_and_pattern_entry(isolated_db, db, i, "return_low_value", True)
    proposals = propose_threshold_adjustments(db, current_threshold=0.90, min_sample_size=5)

    for bad_identity in ("system", "auto", "automated", "", None):
        with pytest.raises(ValueError):
            accept_threshold_proposal(db, proposals[0].id, accepted_by=bad_identity)
    db.close()

def test_cluster_with_overturns_proposes_lowering_threshold(isolated_db):
    """The mirror case: a cluster with real overturns should propose
    LOWERING the threshold, not raising it."""
    from app.agents.learning_loop import propose_threshold_adjustments

    db = isolated_db.SessionLocal()
    for i in range(8):
        _seed_case_and_pattern_entry(isolated_db, db, i, "return_high_risk", i < 5)

    proposals = propose_threshold_adjustments(db, current_threshold=0.90, min_sample_size=5)
    assert len(proposals) == 1
    assert proposals[0].overturn_rate > 0.0
    assert proposals[0].proposed_threshold < 0.90, "Nonzero overturn should propose LOWERING the threshold"
    db.close()

def test_clusters_below_min_sample_size_produce_no_proposal(isolated_db):
    """Statistical caution: too few samples shouldn't move the threshold
    at all, in either direction."""
    from app.agents.learning_loop import propose_threshold_adjustments

    db = isolated_db.SessionLocal()
    for i in range(3):
        _seed_case_and_pattern_entry(isolated_db, db, i, "rare_cluster", True)

    proposals = propose_threshold_adjustments(db, current_threshold=0.90, min_sample_size=5)
    assert proposals == []
    db.close()

def test_retrieve_similar_past_resolutions_ranks_by_relevance(isolated_db):
    from app.agents.learning_loop import record_resolution_outcome, retrieve_similar_past_resolutions
    from app.core.db import ExceptionCase, CaseState

    db = isolated_db.SessionLocal()
    cases = [
        ("case-a", "Apparel return request, item damaged in transit", "return_low_value"),
        ("case-b", "Electronics payment declined by bank", "payment_issue"),
        ("case-c", "Apparel return request, wrong size shipped", "return_low_value"),
    ]
    for case_id, summary, cluster in cases:
        db.add(ExceptionCase(id=case_id, order_id=f"ORD-{case_id}", customer_id="CUST-X",
                              channel="direct", exception_type="return", state=CaseState.RESOLVED))
        db.commit()
        record_resolution_outcome(
            db, case_id=case_id, cluster_key=cluster, case_feature_summary=summary,
            agent_proposed_resolution={"action": "refund", "amount_usd": 20.0},
            human_final_resolution={"action": "refund", "amount_usd": 20.0},
        )

    results = retrieve_similar_past_resolutions(db, "Apparel item damaged during shipping", k=2)
    assert len(results) == 2
    result_ids = {r.case_id for r in results}
    assert result_ids == {"case-a", "case-c"}
    db.close()

def test_retrieve_similar_past_resolutions_empty_store_returns_empty(isolated_db):
    from app.agents.learning_loop import retrieve_similar_past_resolutions

    db = isolated_db.SessionLocal()
    results = retrieve_similar_past_resolutions(db, "anything", k=3)
    assert results == []
    db.close()
