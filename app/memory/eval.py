"""
Memory evaluation - found completely missing during a direct audit: the
memory layer (app/memory/episodic.py) had zero dedicated evaluation of
any kind, unlike RAG (app/rag/eval.py) which has a real, labeled
golden set. This is the memory-layer equivalent: hand-labeled scenarios
with KNOWN-CORRECT expected outputs for summarize_customer_risk_profile()
and get_customer_history(), the two functions real diagnosis logic
(app/agents/workflow_agents.py's fraud/customer-context agents) actually
depends on.

Deliberately small and hand-curated, same principle as the RAG eval
set: a golden set should encode a human's judgment about what's
correct, not the system's own output re-asserted as ground truth.
Each scenario seeds a specific, controlled episode history for a
throwaway test customer, then asserts the EXACT expected risk-profile
values — not just "did it run without crashing."
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session


@dataclass
class MemoryEvalCase:
    name: str
    description: str
    episodes: list  # list of (episode_type, content, days_ago) tuples
    expected_total_return_cases: int
    expected_fraud_flags_raised: int
    expected_most_recent_episode_type: str | None


MEMORY_EVAL_SET = [
    MemoryEvalCase(
        name="clean_customer_no_history",
        description="A customer with zero episode history — everything should report as empty/zero, not error.",
        episodes=[],
        expected_total_return_cases=0,
        expected_fraud_flags_raised=0,
        expected_most_recent_episode_type=None,
    ),
    MemoryEvalCase(
        name="single_return_no_fraud",
        description="One resolved return case, no fraud flags — a normal, low-risk customer.",
        episodes=[
            ("case_resolved", {"exception_type": "return", "action": "refund", "amount_usd": 45.0}, 10),
        ],
        expected_total_return_cases=1,
        expected_fraud_flags_raised=0,
        expected_most_recent_episode_type="case_resolved",
    ),
    MemoryEvalCase(
        name="multiple_returns_correctly_counted",
        description="Three resolved returns must count as exactly 3, not more/fewer — a real regression "
                    "risk if filtering logic double-counts or silently drops entries.",
        episodes=[
            ("case_resolved", {"exception_type": "return", "action": "refund", "amount_usd": 20.0}, 5),
            ("case_resolved", {"exception_type": "return", "action": "refund", "amount_usd": 30.0}, 15),
            ("case_resolved", {"exception_type": "return", "action": "refund", "amount_usd": 40.0}, 25),
        ],
        expected_total_return_cases=3,
        expected_fraud_flags_raised=0,
        expected_most_recent_episode_type="case_resolved",
    ),
    MemoryEvalCase(
        name="non_return_resolutions_not_miscounted_as_returns",
        description="THE regression test for a real filtering-logic risk: a resolved PAYMENT case and a "
                    "resolved DELIVERY case must NOT be counted in total_return_cases — only genuine "
                    "exception_type='return' resolutions should count.",
        episodes=[
            ("case_resolved", {"exception_type": "payment", "action": "refund", "amount_usd": 45.0}, 5),
            ("case_resolved", {"exception_type": "delivery", "action": "reship", "amount_usd": 0.0}, 10),
        ],
        expected_total_return_cases=0,
        expected_fraud_flags_raised=0,
        expected_most_recent_episode_type="case_resolved",
    ),
    MemoryEvalCase(
        name="fraud_flag_correctly_counted",
        description="A single prior fraud flag must be counted exactly once.",
        episodes=[
            ("fraud_flag_raised", {"reason": "suspicious return pattern"}, 20),
        ],
        expected_total_return_cases=0,
        expected_fraud_flags_raised=1,
        expected_most_recent_episode_type="fraud_flag_raised",
    ),
    MemoryEvalCase(
        name="mixed_history_most_recent_by_time_not_insertion_order",
        description="THE regression test for a real temporal-ordering risk: most_recent_episode must be "
                    "whichever episode has the LATEST occurred_at timestamp, not whichever was inserted "
                    "last or listed last in this scenario's own episode list.",
        episodes=[
            ("case_resolved", {"exception_type": "return", "action": "refund", "amount_usd": 20.0}, 30),
            ("fraud_flag_raised", {"reason": "old flag"}, 60),
            # Listed LAST in this scenario definition, but genuinely the
            # MOST RECENT by actual elapsed time (2 days ago) — proves
            # ordering is by real timestamp, not list/insertion position.
            ("case_resolved", {"exception_type": "payment", "action": "deny", "amount_usd": 0.0}, 2),
        ],
        expected_total_return_cases=1,
        expected_fraud_flags_raised=1,
        expected_most_recent_episode_type="case_resolved",
    ),
]


def run_memory_eval(db: Session) -> dict:
    """Runs every scenario in MEMORY_EVAL_SET against a fresh, isolated
    customer_id per scenario (never a real customer), seeding the exact
    episode history each scenario defines, then comparing
    summarize_customer_risk_profile()'s ACTUAL output against the
    hand-labeled expected values.

    Each run uses a fresh UUID suffix on every customer_id — found
    necessary directly, from this eval's own test suite: a deterministic
    customer_id (just f"MEMORY-EVAL-{case.name}") means running the eval
    twice against the same database ACCUMULATES episodes for the same
    customer rather than starting clean, silently breaking a scenario
    like "exactly 3 returns" on the second run once 6 have piled up.
    """
    import uuid
    from app.memory.episodic import log_episode, summarize_customer_risk_profile

    results = []
    now = datetime.now(timezone.utc)
    run_suffix = uuid.uuid4().hex[:8]

    for case in MEMORY_EVAL_SET:
        customer_id = f"MEMORY-EVAL-{case.name}-{run_suffix}"

        for episode_type, content, days_ago in case.episodes:
            log_episode(db, customer_id=customer_id, episode_type=episode_type,
                        content=content, occurred_at=now - timedelta(days=days_ago))

        profile = summarize_customer_risk_profile(db, customer_id)

        actual_most_recent_type = profile["most_recent_episode"]["episode_type"] if profile["most_recent_episode"] else None

        checks = {
            "total_return_cases": (profile["total_return_cases"] == case.expected_total_return_cases,
                                    case.expected_total_return_cases, profile["total_return_cases"]),
            "fraud_flags_raised": (profile["fraud_flags_raised"] == case.expected_fraud_flags_raised,
                                    case.expected_fraud_flags_raised, profile["fraud_flags_raised"]),
            "most_recent_episode_type": (actual_most_recent_type == case.expected_most_recent_episode_type,
                                          case.expected_most_recent_episode_type, actual_most_recent_type),
        }
        passed = all(ok for ok, _, _ in checks.values())

        results.append({
            "name": case.name,
            "description": case.description,
            "passed": passed,
            "checks": {
                field_name: {"expected": expected, "actual": actual}
                for field_name, (ok, expected, actual) in checks.items() if not ok
            } if not passed else {},
        })

    return {
        "total": len(results),
        "passed": sum(1 for r in results if r["passed"]),
        "results": results,
    }
