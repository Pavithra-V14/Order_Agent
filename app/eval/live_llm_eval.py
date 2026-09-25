"""
Live-LLM evaluation harness.

The golden set (tests/golden_set.py) hard-codes FakeLLMClient, so its
"8/8, zero variance" result described the rule-based stand-in - never the
model that actually makes decisions in production. This harness runs the
SAME diagnosis and fraud agents the live pipeline uses, with whatever
BaseLLMClient it is given, over scenarios whose correct outcome is known,
several times each, and reports:

- pass rate per scenario and overall (variance shows up as runs that
  disagree with each other)
- planner failure rate: invalid actions, errors, and step-ceiling hits
- mean steps to conclude
- tokens and latency per call (from the llm.* trace spans)

Payment and carrier data come from the fake gateways on purpose: the tool
outputs are fixed, so the only thing that varies between runs is the
model's reasoning over them. The checks target safety-relevant behaviour
(did it report a fault the tools don't support? did it miss one they do?)
rather than exact wording.

Run it against the real model with scripts/run_live_llm_eval.py; the unit
test runs it against FakeLLMClient to prove the harness itself.
"""
from __future__ import annotations

import importlib
import os
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from app.agents.root_causes import parse_root_causes, PAYMENT, INVENTORY, CARRIER, NO_ANOMALY


@dataclass
class Scenario:
    name: str
    kind: str                       # "diagnosis" | "fraud"
    setup: Callable                 # (db) -> dict of run kwargs
    check: Callable                 # (result) -> (passed: bool, detail: str)


@dataclass
class RunResult:
    scenario: str
    run: int
    passed: bool
    detail: str
    steps: int = 0
    planner_failures: int = 0
    terminated_reason: str = ""
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latencies_ms: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    error: str = ""


def _days_ago(n: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=n)


def _order(db, order_id, status, amount, qty=1, pi=None):
    from app.tools.oms import create_order
    create_order(db, order_id=order_id, customer_id=f"CUST-{order_id}", channel="direct", status=status,
                 total_amount_usd=amount, purchase_date=_days_ago(5),
                 line_items=[{"sku": f"SKU-{order_id}", "category": "apparel", "qty": qty, "price": amount}],
                 payment_intent_id=pi)


def _categories(result) -> set:
    return {c.category for c in parse_root_causes(result.root_causes)}


# ---------------------------------------------------------------- scenarios
def _declined_payment(db):
    from app.tools.payment import get_payment_gateway
    from app.tools.wms import seed_stock
    get_payment_gateway().seed_transaction("pi_ev_declined", 40.0, status="requires_payment_method")
    _order(db, "EV-DECLINED", "payment_failed", 40.0, pi="pi_ev_declined")
    seed_stock(db, "SKU-EV-DECLINED", "WH-1", 10, 10)
    return {"order_id": "EV-DECLINED", "payment_intent_id": "pi_ev_declined", "exception_type": "payment"}


def _check_declined(r):
    cats = _categories(r)
    return PAYMENT in cats and NO_ANOMALY not in cats, f"categories={sorted(cats)}"


def _clean_return_no_tracking(db):
    from app.tools.payment import get_payment_gateway
    from app.tools.wms import seed_stock
    get_payment_gateway().seed_transaction("pi_ev_clean", 30.0, status="succeeded")
    _order(db, "EV-CLEAN", "delivered", 30.0, pi="pi_ev_clean")
    seed_stock(db, "SKU-EV-CLEAN", "WH-1", 10, 10)
    return {"order_id": "EV-CLEAN", "payment_intent_id": "pi_ev_clean", "exception_type": "return"}


def _check_clean(r):
    # The real model once reported "carrier_issue: carrier_unavailable"
    # here purely because there was no tracking number to check.
    cats = _categories(r)
    return cats == {NO_ANOMALY}, f"categories={sorted(cats)}"


def _no_payment_on_record(db):
    from app.tools.wms import seed_stock
    _order(db, "EV-NOPAY", "return_requested", 25.0, pi=None)
    seed_stock(db, "SKU-EV-NOPAY", "WH-1", 10, 10)
    return {"order_id": "EV-NOPAY", "payment_intent_id": None, "exception_type": "return"}


def _check_no_payment(r):
    # A missing payment id is absence of data, not a payment fault (the
    # real model reported "payment_issue: missing payment information").
    cats = _categories(r)
    return PAYMENT not in cats and CARRIER not in cats, f"categories={sorted(cats)}"


def _stock_shortfall(db):
    from app.tools.payment import get_payment_gateway
    from app.tools.wms import seed_stock
    get_payment_gateway().seed_transaction("pi_ev_short", 60.0, status="succeeded")
    _order(db, "EV-SHORT", "paid", 60.0, qty=3, pi="pi_ev_short")
    seed_stock(db, "SKU-EV-SHORT", "WH-1", 1, 1)
    return {"order_id": "EV-SHORT", "payment_intent_id": "pi_ev_short", "exception_type": "return"}


def _check_shortfall(r):
    cats = _categories(r)
    return INVENTORY in cats and PAYMENT not in cats, f"categories={sorted(cats)}"


def _lost_parcel(db):
    from app.tools.payment import get_payment_gateway
    from app.tools.carrier import get_carrier_gateway
    from app.tools.wms import seed_stock
    get_payment_gateway().seed_transaction("pi_ev_lost", 45.0, status="succeeded")
    get_carrier_gateway().seed_tracking("TRK-EV-LOST", "lost")
    _order(db, "EV-LOST", "shipped", 45.0, pi="pi_ev_lost")
    seed_stock(db, "SKU-EV-LOST", "WH-1", 10, 10)
    return {"order_id": "EV-LOST", "payment_intent_id": "pi_ev_lost", "tracking_number": "TRK-EV-LOST",
            "exception_type": "carrier"}


def _check_lost(r):
    cats = _categories(r)
    return CARRIER in cats and PAYMENT not in cats, f"categories={sorted(cats)}"


def _loyal_high_volume_customer(db):
    from app.memory.episodic import log_episode
    for i in range(15):
        log_episode(db, "CUST-EV-LOYAL", "case_resolved", {"exception_type": "return", "outcome": "approved"},
                    occurred_at=_days_ago(i + 1))
    return {"customer_id": "CUST-EV-LOYAL", "address_changed_same_day": False}


def _check_not_flagged(r):
    return (not r["flag"]) and r["risk_score"] < 0.6 and not r.get("degraded"), \
        f"score={r['risk_score']} flag={r['flag']} degraded={r.get('degraded')}"


def _repeat_offender(db):
    from app.memory.episodic import log_episode
    for i in range(2):
        log_episode(db, "CUST-EV-FRAUD", "fraud_flag_raised", {"reason": "chargeback abuse"},
                    occurred_at=_days_ago(i + 10))
    return {"customer_id": "CUST-EV-FRAUD", "address_changed_same_day": True}


def _check_flagged(r):
    return bool(r["flag"]) and not r.get("degraded"), \
        f"score={r['risk_score']} flag={r['flag']} degraded={r.get('degraded')}"


SCENARIOS = [
    Scenario("declined_payment_reports_payment_issue", "diagnosis", _declined_payment, _check_declined),
    Scenario("clean_return_invents_no_fault", "diagnosis", _clean_return_no_tracking, _check_clean),
    Scenario("missing_payment_is_not_a_payment_fault", "diagnosis", _no_payment_on_record, _check_no_payment),
    Scenario("stock_shortfall_reports_inventory_issue", "diagnosis", _stock_shortfall, _check_shortfall),
    Scenario("lost_parcel_reports_carrier_issue", "diagnosis", _lost_parcel, _check_lost),
    Scenario("loyal_high_volume_customer_not_flagged", "fraud", _loyal_high_volume_customer, _check_not_flagged),
    Scenario("repeat_offender_with_address_change_flagged", "fraud", _repeat_offender, _check_flagged),
]


# ---------------------------------------------------------------- runner
def _fresh_db():
    path = os.path.join(tempfile.gettempdir(), f"live_llm_eval_{os.getpid()}_{time.time_ns()}.db")
    os.environ["DATABASE_URL"] = f"sqlite:///{path}"
    from app.core.config import get_settings
    get_settings.cache_clear()
    import app.core.db as db_module
    importlib.reload(db_module)
    db_module.init_db()
    from app.tools.payment import reset_fake_gateway
    from app.tools.carrier import reset_fake_carrier
    from app.cache.ttl_cache import reset_all_caches
    from app.core.circuit_breaker import reset_all_breakers
    reset_fake_gateway(); reset_fake_carrier(); reset_all_caches(); reset_all_breakers()
    return db_module, path


def _run_once(scenario: Scenario, llm, run: int) -> RunResult:
    from app.agents.llm_client import set_llm_trace, reset_llm_trace
    db_module, path = _fresh_db()
    db = db_module.SessionLocal()
    trace_id = f"eval-{scenario.name}-{run}"
    out = RunResult(scenario=scenario.name, run=run, passed=False, detail="")
    token = set_llm_trace(trace_id)
    try:
        kwargs = scenario.setup(db)
        if scenario.kind == "diagnosis":
            from app.agents.diagnosis_agent import run_diagnosis
            result = run_diagnosis(db, llm, case_id=trace_id, **kwargs)
            out.steps = len(result.steps_taken)
            out.terminated_reason = result.terminated_reason
            out.actions = [s.get("action") for s in result.steps_taken]
            out.planner_failures = sum(1 for s in result.steps_taken
                                       if s.get("action") in ("planner_error", "invalid_action")) + \
                (1 if result.terminated_reason in ("max_steps_reached", "timeout_reached", "planner_error") else 0)
        else:
            from app.agents.workflow_agents import run_fraud_risk_agent
            result = run_fraud_risk_agent(db, llm, **kwargs)
            out.planner_failures = 1 if result.get("degraded") else 0
        out.passed, out.detail = scenario.check(result)
    except Exception as e:
        out.error = f"{type(e).__name__}: {e}"
        out.detail = "raised"
    finally:
        reset_llm_trace(token)
        spans = db.query(db_module.TraceSpanRecord).filter(
            db_module.TraceSpanRecord.trace_id == trace_id,
            db_module.TraceSpanRecord.agent_or_tool_name.like("llm.%")).all()
        for s in spans:
            m = s.span_metadata or {}
            out.llm_calls += 1
            out.prompt_tokens += m.get("prompt_tokens") or 0
            out.completion_tokens += m.get("completion_tokens") or 0
            if m.get("latency_ms") is not None:
                out.latencies_ms.append(m["latency_ms"])
        db.close()
        db_module.engine.dispose()
        try:
            os.remove(path)
        except OSError:
            pass
    return out


def run_live_eval(llm, runs: int = 3, scenarios: list | None = None) -> dict:
    """Runs every scenario `runs` times against `llm`. Returns a report
    dict (see module docstring); `report["pass_rate"]` is the gate."""
    scenarios = scenarios or SCENARIOS
    results = [_run_once(sc, llm, i + 1) for sc in scenarios for i in range(runs)]

    per_scenario = []
    for sc in scenarios:
        rs = [r for r in results if r.scenario == sc.name]
        passes = sum(r.passed for r in rs)
        per_scenario.append({
            "scenario": sc.name, "passed": passes, "runs": len(rs),
            "consistent": passes in (0, len(rs)),
            "details": [r.detail if not r.error else r.error for r in rs],
            "actions": [r.actions for r in rs],
        })
    diag = [r for r in results if r.steps]
    latencies = sorted(l for r in results for l in r.latencies_ms)
    return {
        "llm": type(llm).__name__,
        "runs_per_scenario": runs,
        "pass_rate": round(sum(r.passed for r in results) / len(results), 3),
        "planner_failure_rate": round(sum(r.planner_failures > 0 for r in results) / len(results), 3),
        "inconsistent_scenarios": [p["scenario"] for p in per_scenario if not p["consistent"]],
        "mean_diagnosis_steps": round(statistics.mean(r.steps for r in diag), 2) if diag else None,
        "llm_calls": sum(r.llm_calls for r in results),
        "prompt_tokens": sum(r.prompt_tokens for r in results),
        "completion_tokens": sum(r.completion_tokens for r in results),
        "p50_latency_ms": latencies[len(latencies) // 2] if latencies else None,
        "p95_latency_ms": latencies[int(len(latencies) * 0.95)] if latencies else None,
        "scenarios": per_scenario,
        "errors": [f"{r.scenario}#{r.run}: {r.error}" for r in results if r.error],
    }
