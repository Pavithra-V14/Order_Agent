"""
Standalone RQ worker process. Run this as a separate, plain OS process
(NOT inside the FastAPI app process) whenever REDIS_URL is configured -
this is what actually dequeues and executes jobs enqueued by the API. No
Docker needed: this is just `python3 scripts/run_rq_worker.py`, runnable
directly on your machine, in a background terminal, as a systemd/
supervisor service, or as a small separate worker instance on whatever
cloud host you deploy the API to.

Usage:
    export REDIS_URL=rediss://default:password@your-endpoint.upstash.io:6379
    python3 scripts/run_rq_worker.py

Leave it running. It processes jobs continuously until stopped (Ctrl-C).
If REDIS_URL isn't set, this exits immediately with a clear message.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.config import get_settings
from app.workers.job_queue import RQJobQueue
from app.workers import handlers


def main():
    settings = get_settings()
    if not settings.redis_url:
        print("REDIS_URL is not configured - nothing to connect to. This script is only "
              "needed when using the RQ-backed job queue. The in-process queue (the "
              "default when REDIS_URL is unset) runs its own background thread inside "
              "the API process and needs no separate worker.")
        sys.exit(1)

    import redis as redis_lib
    from rq import Worker

    redis_conn = redis_lib.from_url(settings.redis_url)

    handler_map = {
        "process_oms_webhook": handlers.handle_oms_webhook,
        "process_inventory_webhook": handlers.handle_inventory_webhook,
        "process_carrier_webhook": handlers.handle_carrier_webhook,
        "process_policy_upload": handlers.handle_policy_upload,
    }
    print(f"Starting RQ worker for queue '{RQJobQueue.QUEUE_NAME}', handlers: {list(handler_map)}")

    worker = Worker([RQJobQueue.QUEUE_NAME], connection=redis_conn)
    worker.work()


if __name__ == "__main__":
    main()
