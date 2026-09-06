from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, status

from app.workers.job_queue import get_job_queue

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


class OmsWebhookPayload(BaseModel):
    order_id: str
    new_status: str


class InventoryWebhookPayload(BaseModel):
    sku: str
    warehouse: str
    new_on_hand_qty: float
    new_sellable_qty: float


class CarrierWebhookPayload(BaseModel):
    tracking_number: str
    new_status: str
    order_id: str | None = None


class WebhookAckResponse(BaseModel):
    job_id: str
    status: str = "queued"


@router.post("/oms", response_model=WebhookAckResponse, status_code=status.HTTP_202_ACCEPTED)
def oms_webhook(payload: OmsWebhookPayload):
    """Per architecture doc 8.1: 'ack fast, process async' - validates via
    Pydantic and enqueues a job. Does NOT touch the DB or call any tool
    itself - that happens on the worker thread."""
    job_id = get_job_queue().enqueue("process_oms_webhook", payload.model_dump())
    return WebhookAckResponse(job_id=job_id)


@router.post("/inventory", response_model=WebhookAckResponse, status_code=status.HTTP_202_ACCEPTED)
def inventory_webhook(payload: InventoryWebhookPayload):
    job_id = get_job_queue().enqueue("process_inventory_webhook", payload.model_dump())
    return WebhookAckResponse(job_id=job_id)


@router.post("/carrier", response_model=WebhookAckResponse, status_code=status.HTTP_202_ACCEPTED)
def carrier_webhook(payload: CarrierWebhookPayload):
    job_id = get_job_queue().enqueue("process_carrier_webhook", payload.model_dump())
    return WebhookAckResponse(job_id=job_id)


@router.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Practical addition for testing/observing the async contract."""
    job = get_job_queue().get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No such job: {job_id}")
    return {
        "id": job.id, "job_type": job.job_type, "status": job.status.value,
        "result": job.result, "error": job.error,
        "enqueued_at": job.enqueued_at, "started_at": job.started_at, "finished_at": job.finished_at,
    }
