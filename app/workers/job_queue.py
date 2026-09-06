"""
Async job queue - architecture doc 8.1: background jobs run through a
queue, and webhook handlers must "ack fast, process async" rather than
processing inline.

No Redis in this sandbox and no Docker to run one - so this is a real,
working in-process job queue (a background thread consuming from a
queue.Queue) behind the same enqueue()/register_handler() shape a
Redis+RQ-backed queue would expose. Swap point: replace this module's
internals with rq.Queue(connection=redis.from_url(settings.redis_url))
once Redis is available - enqueue()'s call signature at every call site
doesn't need to change.
"""
from __future__ import annotations

import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum


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


class JobQueue:
    """One background worker thread processes jobs sequentially. Handlers
    are registered by job_type - each handler takes the job payload dict
    and returns a result dict."""

    def __init__(self):
        self._queue = queue.Queue()
        self._jobs = {}
        self._handlers = {}
        self._lock = threading.Lock()
        self._worker_thread = None
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
        self._queue.put(job_id)
        return job_id

    def get_job(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def start_worker(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def stop_worker(self, timeout=2.0) -> None:
        self._stop_event.set()
        self._queue.put(None)
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=timeout)

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            job_id = self._queue.get()
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


_queue_singleton = None


def get_job_queue() -> JobQueue:
    global _queue_singleton
    if _queue_singleton is None:
        _queue_singleton = JobQueue()
    return _queue_singleton


def reset_job_queue() -> None:
    global _queue_singleton
    if _queue_singleton is not None:
        _queue_singleton.stop_worker()
    _queue_singleton = None
