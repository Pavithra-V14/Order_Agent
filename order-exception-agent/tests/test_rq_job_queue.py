"""
Tests for RQJobQueue against a REAL local Redis server - not mocked.
Uses RQ's own SimpleWorker in burst mode to actually dequeue and execute
jobs synchronously within the test process, which is RQ's own documented
pattern for testing.

Skipped automatically if no local Redis is reachable.
"""
import os

import pytest


def _redis_available() -> bool:
    try:
        import redis
        client = redis.from_url("redis://localhost:6379/0")
        client.ping()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _redis_available(), reason="No local Redis reachable on localhost:6379")


def _handler_success(payload: dict) -> dict:
    return {"doubled": payload["n"] * 2}


def _handler_failure(payload: dict) -> dict:
    raise ValueError(f"simulated failure for payload {payload}")


@pytest.fixture(autouse=True)
def rq_settings():
    os.environ["REDIS_URL"] = "redis://localhost:6379/0"
    from app.core.config import get_settings
    get_settings.cache_clear()
    from app.workers.job_queue import reset_job_queue
    reset_job_queue()

    import redis as redis_lib
    conn = redis_lib.from_url("redis://localhost:6379/0")
    for key in conn.scan_iter(match="rq:*"):
        conn.delete(key)

    yield

    for key in conn.scan_iter(match="rq:*"):
        conn.delete(key)
    reset_job_queue()
    os.environ.pop("REDIS_URL", None)
    get_settings.cache_clear()


def _run_worker_burst():
    import redis as redis_lib
    from rq import SimpleWorker
    from app.workers.job_queue import RQJobQueue

    conn = redis_lib.from_url("redis://localhost:6379/0")
    worker = SimpleWorker([RQJobQueue.QUEUE_NAME], connection=conn)
    worker.work(burst=True)


def test_get_job_queue_returns_rq_backed_instance_when_redis_configured(rq_settings):
    from app.workers.job_queue import get_job_queue, RQJobQueue
    q = get_job_queue()
    assert isinstance(q, RQJobQueue)


def test_get_job_queue_returns_inprocess_when_no_redis():
    os.environ.pop("REDIS_URL", None)
    from app.core.config import get_settings
    get_settings.cache_clear()
    from app.workers.job_queue import get_job_queue, InProcessJobQueue, reset_job_queue
    reset_job_queue()
    q = get_job_queue()
    assert isinstance(q, InProcessJobQueue)
    reset_job_queue()


def test_rq_job_enqueue_and_real_worker_processes_it_successfully(rq_settings):
    """THE core proof: enqueue against real Redis, run a REAL RQ worker
    to actually process it, confirm SUCCEEDED with the correct result."""
    from app.workers.job_queue import get_job_queue, JobStatus

    q = get_job_queue()
    q.register_handler("test_success", _handler_success)

    job_id = q.enqueue("test_success", {"n": 21})

    job_before = q.get_job(job_id)
    assert job_before.status == JobStatus.QUEUED

    _run_worker_burst()

    job_after = q.get_job(job_id)
    assert job_after.status == JobStatus.SUCCEEDED
    assert job_after.result == {"doubled": 42}


def test_rq_job_failure_is_captured_correctly(rq_settings):
    from app.workers.job_queue import get_job_queue, JobStatus

    q = get_job_queue()
    q.register_handler("test_failure", _handler_failure)

    job_id = q.enqueue("test_failure", {"bad": "payload"})
    _run_worker_burst()

    job_after = q.get_job(job_id)
    assert job_after.status == JobStatus.FAILED
    assert "simulated failure" in job_after.error


def test_rq_get_job_returns_none_for_nonexistent_job(rq_settings):
    from app.workers.job_queue import get_job_queue
    q = get_job_queue()
    assert q.get_job("does-not-exist") is None


def test_rq_enqueue_without_registered_handler_raises_clearly(rq_settings):
    from app.workers.job_queue import get_job_queue
    q = get_job_queue()
    with pytest.raises(ValueError, match="No handler registered"):
        q.enqueue("never_registered_job_type", {})


def test_rq_start_stop_worker_are_safe_noops(rq_settings):
    from app.workers.job_queue import get_job_queue
    q = get_job_queue()
    q.start_worker()
    q.stop_worker()
