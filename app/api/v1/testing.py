"""
Testing endpoints - lets the UI trigger a labeled golden-set run (the 8
offline regression scenarios from tests/golden_set.py), with optional
scenario SELECTION rather than always running all 8, and browse past
runs. Kept deliberately separate from /metrics (real production/live
data) so the two are never confused in the UI: golden-set results are
a fixed, repeatable regression check, live metrics are whatever has
actually happened in the running system.

IMPORTANT: golden-set scenarios must run in a SEPARATE PROCESS
(scripts/run_golden_set_json.py), never in-process. Found directly:
each scenario calls importlib.reload(app.core.db), swapping the
database connection to a throwaway temp file. Calling them in-process
corrupted this live application's own database connection for the rest
of the process's lifetime the moment this endpoint was first hit.
"""
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime, timezone

from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import require_admin
from sqlalchemy.orm import Session

from app.core.db import get_db, GoldenSetRunRecord, RagEvalRunRecord

router = APIRouter(prefix="/testing", tags=["testing"])


class RunGoldenSetRequest(BaseModel):
    label: str
    scenario_names: list[str] | None = None  # None or empty = run all 8


@router.get("/golden-set/scenarios")
def list_available_scenarios(_auth=Depends(require_admin)):
    """Every available golden-set scenario, with a short description —
    lets the UI render selection checkboxes instead of only offering
    'run all 8 blindly', which was the previous, less useful default."""
    from tests.golden_set import ALL_SCENARIOS
    return [
        {"name": s.__name__, "description": (s.__doc__ or "").strip().split("\n")[0]}
        for s in ALL_SCENARIOS
    ]


@router.post("/golden-set/run")
def trigger_golden_set_run(payload: RunGoldenSetRequest, db: Session = Depends(get_db),
                            _auth=Depends(require_admin)):
    """Runs the selected golden-set scenarios (or all 8 if none
    specified) in a SEPARATE PROCESS (see module docstring for why) and
    saves the labeled result. Synchronous (not queued via the job
    system) deliberately: the whole run is meant to complete in a
    couple of seconds, and the person triggering this from the UI wants
    to see the result immediately.

    The subprocess's environment is EXPLICITLY stripped of every cloud
    credential — found necessary directly from a real production
    timeout: subprocess.run() inherits the FULL parent environment by
    default, so a real GROQ_API_KEY/QDRANT_URL/NEO4J_URI/MISTRAL_API_KEY
    configured for the live app leaks into the golden-set subprocess too,
    causing get_llm_client()/get_embedder()/get_qdrant_client()'s
    auto-selection logic to silently pick REAL cloud clients instead of
    the fast, deterministic local/fake substitutes golden-set scenarios
    are actually designed around — turning a sub-second offline
    regression check into a series of real network round-trips (Groq
    inference, Qdrant Cloud queries, Neo4j Aura connections) slow enough
    to blow past any reasonable timeout. This is the same "tests must
    never depend on whatever happens to be in a real .env" principle
    already applied to pytest (tests/conftest.py) — applied here to a
    subprocess invocation instead of an in-process settings override,
    since the mechanism has to match how isolation is actually achieved
    in each context.
    """
    import os

    cmd = [sys.executable, "scripts/run_golden_set_json.py"]
    if payload.scenario_names:
        cmd += payload.scenario_names

    isolated_env = dict(os.environ)
    for key in [
        "GROQ_API_KEY", "MISTRAL_API_KEY", "QDRANT_URL", "QDRANT_API_KEY",
        "NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD", "NEO4J_DATABASE",
        "EASYPOST_API_KEY", "SHIPPO_API_KEY", "STRIPE_API_KEY",
        "REDIS_URL", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "TRACING_ENABLED",
        "DATABASE_URL",  # golden-set scenarios build their own isolated temp SQLite files
    ]:
        isolated_env.pop(key, None)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90, env=isolated_env)
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail="Golden set subprocess exceeded 90s even with cloud credentials stripped from its "
                   "environment — this points to something else genuinely hanging, not the usual "
                   "real-cloud-credential-leak cause. Try running "
                   "`python scripts/run_golden_set_json.py` directly in a terminal to see exactly "
                   "where it stalls.",
        )
    if result.returncode != 0 and not result.stdout.strip():
        raise HTTPException(
            status_code=500,
            detail=f"Golden set subprocess failed to produce output: {result.stderr[-2000:]}",
        )
    try:
        results = json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not parse golden set subprocess output: {e}. Raw stdout: {result.stdout[-2000:]}",
        )

    if not results:
        raise HTTPException(
            status_code=400,
            detail="No scenarios matched — check the scenario_names you selected against GET /testing/golden-set/scenarios",
        )

    pass_count = sum(1 for r in results if r["passed"])
    run_id = str(uuid.uuid4())
    record = GoldenSetRunRecord(
        id=run_id, label=payload.label, run_at=datetime.now(timezone.utc),
        pass_count=pass_count, total_count=len(results), results=results, triggered_by="ui",
    )
    db.add(record)
    db.commit()

    return {
        "run_id": run_id, "label": payload.label, "pass_count": pass_count,
        "total_count": len(results), "results": results,
    }


@router.get("/golden-set/runs")
def list_golden_set_runs(db: Session = Depends(get_db), _auth=Depends(require_admin)):
    """Run history, most recent first - lets the UI show multiple
    labeled runs side by side (e.g. 'before fix' vs 'after fix')."""
    runs = db.query(GoldenSetRunRecord).order_by(GoldenSetRunRecord.run_at.desc()).all()
    return [
        {
            "run_id": r.id, "label": r.label, "run_at": r.run_at.isoformat() if r.run_at else None,
            "pass_count": r.pass_count, "total_count": r.total_count,
        }
        for r in runs
    ]


@router.get("/golden-set/runs/{run_id}")
def get_golden_set_run(run_id: str, db: Session = Depends(get_db), _auth=Depends(require_admin)):
    """Full per-scenario detail for one run."""
    record = db.get(GoldenSetRunRecord, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No such run: {run_id}")
    return {
        "run_id": record.id, "label": record.label,
        "run_at": record.run_at.isoformat() if record.run_at else None,
        "pass_count": record.pass_count, "total_count": record.total_count,
        "results": record.results, "triggered_by": record.triggered_by,
    }


@router.get("/rag-eval")
def run_rag_eval_endpoint(top_k: int = 5, _auth=Depends(require_admin)):
    """Precision/recall@k against a small, hand-labeled query -> expected
    doc_id eval set (app/rag/eval.py) — the missing eval artifact
    app/core/metrics.py's docstring documented as needed. Runs live
    (in-process) against whatever retrieval pipeline is currently
    configured (Qdrant local/Cloud, TF-IDF/Mistral embeddings) —
    deliberately NOT going through subprocess isolation like the golden
    set, since retrieval has no equivalent database-swapping side effect
    that would make in-process execution unsafe here."""
    from app.rag.eval import run_rag_eval
    return run_rag_eval(top_k=top_k)


@router.get("/memory-eval")
def run_memory_eval_endpoint(db: Session = Depends(get_db), _auth=Depends(require_admin)):
    """Hand-labeled golden set for the memory layer (app/memory/eval.py) -
    found completely missing during a direct audit: RAG had a real
    labeled eval set, memory had none at all. Runs live against
    whichever memory backend is currently active (real Graphiti/Neo4j,
    Graphiti/Kuzu, or the plain-SQL fallback) — see _use_graphiti() in
    app/memory/episodic.py."""
    from app.memory.eval import run_memory_eval
    return run_memory_eval(db)


@router.get("/reranker-lift")
def run_reranker_lift_endpoint(top_k: int = 5, _auth=Depends(require_admin)):
    """How much does reranking actually move the correct document up in
    rank, versus its position from dense+sparse fusion alone? Uses the
    same labeled eval set as /rag-eval, comparing pre- and post-rerank
    positions via hybrid_search's pre_rerank_capture hook."""
    from app.rag.eval import run_reranker_lift_eval
    return run_reranker_lift_eval(top_k=top_k)


@router.get("/index-freshness")
def get_index_freshness(_auth=Depends(require_admin)):
    """Time between a policy document's own file timestamp (source_mtime
    — when it was placed in data/policies/, whether via upload or the
    original seeding script) and when it actually became searchable
    (indexed_at — when ingest_policy_pdf() finished embedding and
    upserting it). The other half of app/core/metrics.py's documented
    'index freshness lag' gap, now computable directly from
    reindex_state.json's own recorded timestamps — no new tracking
    infrastructure needed beyond recording these two fields at the
    moment ingestion completes."""
    import json
    import os
    from datetime import datetime
    from app.rag.ingestion import _REINDEX_STATE_PATH

    if not os.path.exists(_REINDEX_STATE_PATH):
        return {"documents": [], "avg_lag_seconds": None}

    with open(_REINDEX_STATE_PATH) as f:
        state = json.load(f)

    documents = []
    lags = []
    for doc_id, entry in state.items():
        if not isinstance(entry, dict) or "indexed_at" not in entry or "source_mtime" not in entry:
            # Old-format entry (pre-dates this feature) — no timestamps
            # recorded, so freshness lag genuinely isn't knowable for
            # it. Reported as null, not silently skipped, so the UI can
            # show WHY a document has no lag figure.
            documents.append({"doc_id": doc_id, "lag_seconds": None, "note": "indexed before freshness tracking existed"})
            continue

        indexed_at = datetime.fromisoformat(entry["indexed_at"])
        source_mtime = datetime.fromisoformat(entry["source_mtime"])
        lag_seconds = (indexed_at - source_mtime).total_seconds()
        documents.append({
            "doc_id": doc_id, "lag_seconds": round(lag_seconds, 2),
            "indexed_at": entry["indexed_at"], "source_mtime": entry["source_mtime"],
        })
        lags.append(lag_seconds)

    return {
        "documents": documents,
        "avg_lag_seconds": round(sum(lags) / len(lags), 2) if lags else None,
    }


@router.post("/rag-eval-dashboard/run")
def run_live_rag_evaluation(db: Session = Depends(get_db), _auth=Depends(require_admin)):
    """Runs a genuinely LIVE RAG evaluation - combines two real things
    that previously only existed separately: the OFFLINE, labeled
    precision/recall/reranker-lift eval (run fresh against whatever
    retrieval pipeline is currently live, not stale cached numbers) and
    a snapshot of the LIVE, production faithfulness/groundedness
    metrics at this same moment (computed from real trace data of
    actual case traffic, not the labeled test set). Stores both
    together as one historical row so the dashboard can show a genuine
    trend over time, not just a single snapshot.

    Deliberately admin-only and POST (not auto-run on every page load):
    this genuinely executes real retrieval queries against your real
    configured backend (Qdrant Cloud, Mistral, etc.), which has real
    cost/latency — a person should trigger it deliberately, not have it
    fire silently every time someone opens a dashboard.
    """
    from app.rag.eval import run_rag_eval, run_reranker_lift_eval
    from app.core.metrics import compute_rag_metrics

    offline_precision_recall = run_rag_eval(top_k=5)
    offline_reranker_lift = run_reranker_lift_eval(top_k=5)
    live_metrics = compute_rag_metrics(db)

    record = RagEvalRunRecord(
        offline_avg_precision_at_k=offline_precision_recall["avg_precision_at_k"],
        offline_avg_recall_at_k=offline_precision_recall["avg_recall_at_k"],
        offline_avg_reranker_lift=offline_reranker_lift["avg_lift"],
        offline_per_case={
            "precision_recall": offline_precision_recall["per_case"],
            "reranker_lift": offline_reranker_lift["per_case"],
        },
        live_retrieval_span_count=live_metrics["retrieval_span_count"],
        live_retrieval_hit_rate=live_metrics["retrieval_hit_rate"],
        live_faithfulness_score=live_metrics["faithfulness_score"],
        live_hallucination_rate=live_metrics["hallucination_rate"],
        live_citing_decision_count=live_metrics["citing_decision_count"],
    )
    db.add(record)
    db.commit()

    return {
        "run_id": record.id, "run_at": record.run_at.isoformat(),
        "offline": {
            "avg_precision_at_k": record.offline_avg_precision_at_k,
            "avg_recall_at_k": record.offline_avg_recall_at_k,
            "avg_reranker_lift": record.offline_avg_reranker_lift,
            "per_case": record.offline_per_case,
        },
        "live": {
            "retrieval_span_count": record.live_retrieval_span_count,
            "retrieval_hit_rate": record.live_retrieval_hit_rate,
            "faithfulness_score": record.live_faithfulness_score,
            "hallucination_rate": record.live_hallucination_rate,
            "citing_decision_count": record.live_citing_decision_count,
        },
    }


@router.get("/rag-eval-dashboard/history")
def get_rag_eval_history(limit: int = 50, db: Session = Depends(get_db), _auth=Depends(require_admin)):
    """Every past run, most recent first — the actual "live over time"
    view: is retrieval quality trending better or worse across runs,
    not just what it looks like right now."""
    runs = db.query(RagEvalRunRecord).order_by(RagEvalRunRecord.run_at.desc()).limit(limit).all()
    return [
        {
            "run_id": r.id, "run_at": r.run_at.isoformat(),
            "offline_avg_precision_at_k": r.offline_avg_precision_at_k,
            "offline_avg_recall_at_k": r.offline_avg_recall_at_k,
            "offline_avg_reranker_lift": r.offline_avg_reranker_lift,
            "live_retrieval_hit_rate": r.live_retrieval_hit_rate,
            "live_faithfulness_score": r.live_faithfulness_score,
            "live_hallucination_rate": r.live_hallucination_rate,
            "live_citing_decision_count": r.live_citing_decision_count,
        }
        for r in runs
    ]


# ── Demo scenarios: one-click, realistic end-to-end cases via the UI ───────
# Each entry below maps to a real script under scripts/ that already
# seeds realistic underlying data (order, stock, payment status,
# customer history) and runs the actual pipeline - these are the exact
# same scripts documented for CLI use, exposed here so a person can
# trigger one from a browser instead of a terminal and land straight on
# the resulting case. Unlike the golden-set runner above, these
# deliberately do NOT strip real credentials from the subprocess
# environment - the whole point of a demo is to show the real pipeline
# against whatever real Stripe/Groq/Qdrant/etc is actually configured,
# same as running the script by hand would.
DEMO_SCENARIOS = {
    "full_pipeline_demo": "Payment declined, end-to-end (webhook -> diagnosis -> resolution, one call)",
    "return_pipeline_demo": "Ordinary return request within policy - nothing broken, just a decision to make",
    "delivery_pipeline_demo": "Carrier tracking shows the package lost/stuck in transit",
    "fraud_pipeline_demo": "Prior fraud flag + same-day address change - crosses the real fraud threshold",
    "inventory_pipeline_demo": "Requested quantity exceeds sellable stock (not just on-hand)",
    "refund_demo": "Direct refund tool call against the real/fake payment gateway",
    "reship_demo": "Direct reship + return-label tool call against the real/fake carrier gateway",
}

_CASE_ID_PATTERN = re.compile(r"/cases/([A-Za-z0-9_-]+)")


@router.get("/demo-scenarios")
def list_demo_scenarios(_auth=Depends(require_admin)):
    """Every demo scenario available to trigger, with a short
    description - lets the UI render a picker instead of hardcoding the
    list client-side, so adding a new scenario here is the only place
    that needs to change."""
    return [{"name": name, "description": desc} for name, desc in DEMO_SCENARIOS.items()]


class RunDemoScenarioRequest(BaseModel):
    scenario: str


@router.post("/demo-scenarios/run")
def run_demo_scenario(payload: RunDemoScenarioRequest, _auth=Depends(require_admin)):
    """Runs the selected demo script in a SEPARATE PROCESS - same
    reasoning as the golden-set runner above (these scripts freely
    create/reuse their own DB sessions and, in full_pipeline_demo's
    case, construct their own in-process TestClient(app); running that
    inside this already-running app's own request-handling process
    risks exactly the same kind of interference). Deliberately inherits
    the real environment (not stripped) - a demo's whole purpose is to
    show the real pipeline against whatever's actually configured.
    Extracts the resulting case_id by scanning stdout for a
    "/cases/<id>" reference, since that's what every script already
    prints as its own "here's where to look" pointer - no per-script
    special-casing needed here as new scenarios are added, as long as
    each new script keeps printing that same kind of pointer."""
    if payload.scenario not in DEMO_SCENARIOS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown scenario '{payload.scenario}'. Available: {list(DEMO_SCENARIOS.keys())}",
        )

    # Checked directly, not assumed: these demo scripts were built for
    # standalone CLI use and use whatever local Qdrant path
    # get_qdrant_client() defaults to - the SAME path this already-
    # running app's own process is holding open (Qdrant's embedded local
    # mode is single-process only; it raises rather than corrupting
    # data). This is fine when a real, concurrent-safe Qdrant Cloud
    # instance is configured (QDRANT_URL) - both processes talk to the
    # same remote collection with no lock conflict. When only the local
    # embedded fallback is available, this endpoint would otherwise fail
    # with a confusing "Storage folder already accessed" stack trace -
    # this check turns that into an actionable message instead.
    from app.core.config import get_settings
    if not get_settings().qdrant_url:
        raise HTTPException(
            status_code=409,
            detail=(
                "This demo needs to open its own Qdrant connection, and no QDRANT_URL (Qdrant Cloud) "
                "is configured - only the local, single-process embedded Qdrant, which this already-"
                "running server is currently holding open. Either configure QDRANT_URL for a real "
                "concurrent-safe instance, or stop this server and run the script directly: "
                f"python scripts/{payload.scenario}.py"
            ),
        )

    cmd = [sys.executable, f"scripts/{payload.scenario}.py"]
    try:
        # 240s, not 60s - found necessary directly from a real user
        # timeout. Unlike the golden-set runner above, this endpoint
        # deliberately does NOT strip real credentials, since a demo's
        # whole point is to exercise the real pipeline - which means a
        # real, possibly multi-step Groq diagnosis loop, real Qdrant
        # Cloud retrieval, and potentially a real Shippo call, each a
        # genuine network round-trip rather than an in-memory fake. 60s
        # was tuned by copying the golden-set runner's number without
        # accounting for this difference; a real LLM-driven diagnosis
        # legitimately needs more room than a fully-mocked regression
        # check does.
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    except subprocess.TimeoutExpired:
        raise HTTPException(
            status_code=504,
            detail=f"Demo scenario '{payload.scenario}' exceeded 240s - check the server's own "
                    f"terminal output for what it was doing when it stalled.",
        )

    output = result.stdout + result.stderr
    match = _CASE_ID_PATTERN.search(output)
    case_id = match.group(1) if match else None

    return {
        "scenario": payload.scenario,
        "success": result.returncode == 0,
        "case_id": case_id,
        "output": output[-4000:],  # tail only - some of these print a lot
    }

