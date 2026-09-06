"""
Phase 14: load test. Architecture doc specifies Locust/k6, which need a
live server process plus a separate load-generator process - running two
long-lived, independently-managed processes has proven unreliable in
this sandbox across tool calls. This substitutes a concurrent in-process
load test using ThreadPoolExecutor against the same FastAPI app via
TestClient - same request-handling code path, just without a second
process making real HTTP calls over a socket.

Sized against Phase 1's estimated volume: ~50 exceptions/day. This test
drives far more than that in a burst (200 concurrent requests) - the
right kind of margin for a load test.
"""
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client_and_db():
    tmp_db = os.path.join(tempfile.gettempdir(), "test_phase14_load.db")
    if os.path.exists(tmp_db):
        os.remove(tmp_db)
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp_db}"

    from app.core.config import get_settings
    get_settings.cache_clear()

    import app.core.db as db_module
    import importlib
    importlib.reload(db_module)

    from app.tools.payment import reset_fake_gateway
    from app.tools.carrier import reset_fake_carrier
    from app.cache.ttl_cache import reset_all_caches
    from app.workers.job_queue import reset_job_queue
    from app.core.circuit_breaker import reset_all_breakers
    reset_fake_gateway()
    reset_fake_carrier()
    reset_all_caches()
    reset_job_queue()
    reset_all_breakers()

    import app.main as main_module
    importlib.reload(main_module)

    with TestClient(main_module.app) as client:
        yield client, db_module

    if os.path.exists(tmp_db):
        os.remove(tmp_db)


def test_load_case_creation_and_health_check_under_burst(client_and_db):
    """200 concurrent requests across /health (read) and /cases (write) -
    ~4x Phase 1's entire estimated DAILY volume in a single burst.
    Asserts: zero errors, and p95 latency stays well under 1 second."""
    client, _ = client_and_db
    n_requests = 200
    latencies = []
    errors = []

    def make_request(i):
        start = time.monotonic()
        try:
            if i % 2 == 0:
                resp = client.get("/api/v1/health")
            else:
                resp = client.post("/api/v1/cases", json={
                    "order_id": f"ORD-LOAD-{i}", "customer_id": f"CUST-LOAD-{i}",
                    "channel": "direct", "exception_type": "return",
                })
            latency = time.monotonic() - start
            latencies.append(latency)
            if resp.status_code >= 500:
                errors.append((i, resp.status_code, resp.text[:200]))
        except Exception as e:
            errors.append((i, "exception", str(e)))

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(make_request, i) for i in range(n_requests)]
        for f in as_completed(futures):
            pass

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]

    assert len(errors) == 0, f"Expected zero 5xx errors/exceptions under burst load, got: {errors[:5]}"
    assert p95 < 1.0, f"p95 latency {p95:.3f}s exceeded 1s under burst load (p50={p50:.3f}s)"

    cases_resp = client.get("/api/v1/cases")
    created_count = sum(1 for c in cases_resp.json() if c["order_id"].startswith("ORD-LOAD-"))
    assert created_count == n_requests // 2, f"Expected {n_requests // 2} cases created, got {created_count}"


def test_load_does_not_spuriously_trip_circuit_breakers(client_and_db):
    """Confirms normal (non-failing) load doesn't accidentally trip a
    circuit breaker meant only for genuine dependency failures."""
    from app.core.circuit_breaker import get_circuit_breaker, CircuitState
    from app.tools.payment import get_payment_gateway

    client, _ = client_and_db
    gateway = get_payment_gateway()
    gateway.seed_transaction("pi_load_1", amount_usd=10.0)

    breaker = get_circuit_breaker("payment", failure_threshold=3, reset_timeout_seconds=30.0)

    def issue_refund(i):
        import app.core.db as db_module
        db = db_module.SessionLocal()
        try:
            breaker.call(lambda: gateway.issue_refund(db, "pi_load_1", 1.0, idempotency_key=f"load-key-{i}"))
        finally:
            db.close()

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(issue_refund, range(30)))

    assert breaker.state == CircuitState.CLOSED, "circuit must stay CLOSED under all-successful load"
    assert breaker.call_failures == 0
