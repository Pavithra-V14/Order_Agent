"""
Async job queue - architecture doc 8.1: background jobs run through a
queue, and webhook handlers must "ack fast, process async" rather than
processing inline.

Two backends, chosen automatically from settings.redis_url (same
settings-driven pattern as get_cache()/get_llm_client()/get_embedder()/
get_carrier_gateway()):

  1. No REDIS_URL set -> InProcessJobQueue: a background thread consuming
     from a queue.Queue, all inside this one process. No Docker, no
     external service, but jobs are lost on restart and this only works
     for a single-process deployment.

  2. REDIS_URL set -> RQJobQueue: real RQ (Redis Queue), backed by
     Upstash Redis or any standard Redis endpoint (same one the cache
     layer uses). IMPORTANT ARCHITECTURAL DIFFERENCE from the in-process
     backend: RQ requires a SEPARATE WORKER PROCESS to actually execute
     jobs — enqueue() pushes to Redis and returns immediately either way,
     but nothing runs the job unless a worker process is consuming that
     queue. No Docker needed for this — `python3 scripts/run_rq_worker.py`
     (or the `rq worker` CLI command) is just a second plain Python
     process, which can run on the same machine, a background service, a
     systemd unit, or a separate small cloud worker instance. This is
     documented here explicitly rather than silently pretending the
     in-process worker-thread model still applies once Redis is
     configured — it's a real, load-bearing difference an operator needs
     to know about.
"""
from __future__ import annotations

import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


# Lanes: short housekeeping jobs must never wait behind a case pipeline
# that can spend 30s+ in LLM calls. Before lanes, a single worker thread
# ran everything in order, so an inventory update (which invalidates the
# stock cache the next diagnosis reads) could sit behind several
# pipelines - exactly the stale-stock window it exists to close.
FAST_LANE = "fast"
SLOW_LANE = "slow"
_SLOW_JOB_TYPES = frozenset({"process_oms_webhook", "process_carrier_webhook", "tier3_judge_sample"})


def lane_for(job_type: str) -> str:
    return SLOW_LANE if job_type in _SLOW_JOB_TYPES else FAST_LANE


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class Job:
    id: str
    job_type: str
    payload: dict
    status: JobStatus = JobStatus.QUEUED
    result: dict = field(default_factory=dict)
    error: str = ""
    enqueued_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str = None
    finished_at: str = None


class InProcessJobQueue:
    """Background worker threads inside this process, one pool per lane
    (see lane_for): the fast lane keeps a dedicated worker so housekeeping
    never waits behind a case pipeline; the slow lane runs
    JOB_QUEUE_SLOW_WORKERS pipelines in parallel. Handlers are registered
    by job_type - each takes the payload dict and returns a result dict."""

    def __init__(self, slow_workers: int | None = None):
        if slow_workers is None:
            from app.core.config import get_settings
            settings = get_settings()
            slow_workers = settings.job_queue_slow_workers
            # SQLite allows one writer at a time; parallel pipelines would
            # just contend for its lock ("database is locked").
            if settings.database_url.startswith("sqlite"):
                slow_workers = 1
        self._lanes = {FAST_LANE: queue.Queue(), SLOW_LANE: queue.Queue()}
        self._lane_sizes = {FAST_LANE: 1, SLOW_LANE: max(1, int(slow_workers))}
        self._jobs = {}
        self._handlers = {}
        self._lock = threading.Lock()
        self._worker_threads: list[threading.Thread] = []
        self._stop_event = threading.Event()

    def register_handler(self, job_type, handler) -> None:
        self._handlers[job_type] = handler

    def enqueue(self, job_type, payload) -> str:
        """Ack-fast contract: returns immediately with a job_id - the
        actual handler runs on the worker thread, never inline."""
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, job_type=job_type, payload=payload)
        with self._lock:
            self._jobs[job_id] = job
        self._lanes[lane_for(job_type)].put(job_id)
        return job_id

    def get_job(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def start_worker(self) -> None:
        if any(t.is_alive() for t in self._worker_threads):
            return
        self._stop_event.clear()
        self._worker_threads = [
            threading.Thread(target=self._worker_loop, args=(lane,), daemon=True, name=f"jobs-{lane}-{i}")
            for lane, size in self._lane_sizes.items() for i in range(size)
        ]
        for t in self._worker_threads:
            t.start()

    def stop_worker(self, timeout=2.0) -> None:
        self._stop_event.set()
        for lane, size in self._lane_sizes.items():
            for _ in range(size):
                self._lanes[lane].put(None)
        for t in self._worker_threads:
            t.join(timeout=timeout)

    def _worker_loop(self, lane: str) -> None:
        q = self._lanes[lane]
        while not self._stop_event.is_set():
            job_id = q.get()
            if job_id is None:
                continue
            self._process_job(job_id)

    def _process_job(self, job_id) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            return

        handler = self._handlers.get(job.job_type)
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(timezone.utc).isoformat()

        if handler is None:
            job.status = JobStatus.FAILED
            job.error = f"No handler registered for job_type {job.job_type!r}"
            job.finished_at = datetime.now(timezone.utc).isoformat()
            return

        try:
            result = handler(job.payload)
            job.status = JobStatus.SUCCEEDED
            job.result = result or {}
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = f"{e}\n{traceback.format_exc()}"
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()

    def wait_for_job(self, job_id, timeout=5.0):
        """Test helper - polls until a job finishes or timeout elapses."""
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get_job(job_id)
            if job is not None and job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                return job
            time.sleep(0.01)
        return self.get_job(job_id)


_RQ_STATUS_MAP = {
    "queued": JobStatus.QUEUED,
    "deferred": JobStatus.QUEUED,
    "scheduled": JobStatus.QUEUED,
    "started": JobStatus.RUNNING,
    "finished": JobStatus.SUCCEEDED,
    "failed": JobStatus.FAILED,
    "stopped": JobStatus.FAILED,
    "canceled": JobStatus.FAILED,
}


class RQJobQueue:
    """Real RQ-backed queue. See module docstring for the critical
    "needs a separate worker process" note — this class's enqueue()/
    get_job() work fully standalone (they only talk to Redis), but
    nothing dequeues and runs a job without scripts/run_rq_worker.py (or
    an equivalent `rq worker <queue_name>` process) running somewhere
    reachable to the same Redis instance.
    """

    QUEUE_NAME = "order_exception_agent"                 # slow lane (pre-existing name, so running workers keep working)
    FAST_QUEUE_NAME = "order_exception_agent_fast"

    def __init__(self, redis_url: str):
        import redis as redis_lib
        from rq import Queue
        self._redis_conn = redis_lib.from_url(redis_url)
        self._queues = {SLOW_LANE: Queue(self.QUEUE_NAME, connection=self._redis_conn),
                        FAST_LANE: Queue(self.FAST_QUEUE_NAME, connection=self._redis_conn)}
        self._handlers: dict[str, callable] = {}

    def register_handler(self, job_type: str, handler) -> None:
        """RQ needs a real, module-level importable function to enqueue
        (the worker process re-imports it by reference) — this is exactly
        what app.workers.handlers already provides, so registration here
        is just recording the mapping; the function itself must already
        be a plain top-level function, not a closure or lambda."""
        self._handlers[job_type] = handler

    def enqueue(self, job_type: str, payload: dict) -> str:
        handler = self._handlers.get(job_type)
        if handler is None:
            raise ValueError(f"No handler registered for job_type {job_type!r}")
        job = self._queues[lane_for(job_type)].enqueue(handler, payload, job_timeout="5m")
        return job.id

    def get_job(self, job_id: str):
        from rq.job import Job as RQJob
        from rq.exceptions import NoSuchJobError
        try:
            rq_job = RQJob.fetch(job_id, connection=self._redis_conn)
        except NoSuchJobError:
            return None

        status = _RQ_STATUS_MAP.get(rq_job.get_status(refresh=True), JobStatus.QUEUED)
        result = rq_job.return_value(refresh=True) if rq_job.is_finished else None
        error = ""
        if rq_job.is_failed:
            latest = rq_job.latest_result()
            error = latest.exc_string if latest and latest.exc_string else "job failed (no exception detail available)"

        return Job(
            id=rq_job.id,
            job_type="",  # RQ doesn't track our job_type string separately; payload/result carry the meaning
            payload=rq_job.args[0] if rq_job.args else {},
            status=status,
            result=result if isinstance(result, dict) else ({"value": result} if result is not None else {}),
            error=error,
            enqueued_at=rq_job.enqueued_at.isoformat() if rq_job.enqueued_at else "",
            started_at=rq_job.started_at.isoformat() if rq_job.started_at else None,
            finished_at=rq_job.ended_at.isoformat() if rq_job.ended_at else None,
        )

    def start_worker(self) -> None:
        """Deliberately a no-op — see class/module docstring. The real
        worker is a separate process; this method exists only so
        app.main's lifespan hook (written against the same interface as
        InProcessJobQueue) doesn't need an if-backend branch there."""
        pass

    def stop_worker(self, timeout: float = 2.0) -> None:
        pass

    def wait_for_job(self, job_id, timeout=5.0):
        """Test helper - polls until a job finishes or timeout elapses.
        In production, callers poll get_job() the same way (e.g. the
        /webhooks/jobs/{job_id} endpoint), since a worker process
        completes jobs asynchronously either way."""
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get_job(job_id)
            if job is not None and job.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
                return job
            time.sleep(0.01)
        return self.get_job(job_id)


_queue_singleton = None


def get_job_queue():
    """Auto-selects RQJobQueue when REDIS_URL is configured, falling back
    to InProcessJobQueue otherwise. Same settings-driven pattern as every
    other cloud swap point in this project."""
    global _queue_singleton
    if _queue_singleton is None:
        from app.core.config import get_settings
        settings = get_settings()
        if settings.redis_url:
            _queue_singleton = RQJobQueue(redis_url=settings.redis_url)
        else:
            _queue_singleton = InProcessJobQueue()
    return _queue_singleton


def reset_job_queue() -> None:
    global _queue_singleton
    if _queue_singleton is not None and isinstance(_queue_singleton, InProcessJobQueue):
        _queue_singleton.stop_worker()
    _queue_singleton = None
