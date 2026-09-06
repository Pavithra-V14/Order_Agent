"""
MCP server exposing the tool layer (architecture doc Part 6 / 8.1) - this
is the actual mechanism that makes tool access standardized and swappable
across heterogeneous backing systems (mock OMS/WMS today, real systems
later), rather than bespoke per-tool glue code sprinkled through the
agent layer.

Run standalone for manual inspection: python3 -m app.tools.mcp_server
(starts an stdio MCP server) - Phase 6's orchestrator will connect to this
as an MCP client rather than importing these tool functions directly,
once the orchestrator is built.
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from app.core.db import SessionLocal
from app.tools import oms, wms, payment, carrier, notification

server = MCPServer(
    name="order-exception-agent-tools",
    version="0.1.0-phase4",
    instructions=(
        "Tools for diagnosing and resolving e-commerce order exceptions: "
        "order lookup (OMS), inventory lookup/transfer (WMS), payment "
        "status/refund (payment gateway), shipping tracking/labels "
        "(carrier), and customer notifications. Write-capable tools "
        "(refund, transfer, label) require an idempotency_key."
    ),
)


@server.tool()
def get_order(order_id: str) -> dict:
    """Look up an order by ID from the OMS. Read-only."""
    db = SessionLocal()
    try:
        result = oms.get_order(db, order_id)
        if result is None:
            return {"error": f"No such order: {order_id}"}
        return result
    finally:
        db.close()


@server.tool()
def update_order_status(order_id: str, new_status: str) -> dict:
    """Update an order's status in the OMS. Naturally idempotent - no
    idempotency_key required (see oms.py docstring)."""
    db = SessionLocal()
    try:
        return oms.update_order_status(db, order_id, new_status)
    finally:
        db.close()


@server.tool()
def get_stock(sku: str, warehouse: str | None = None) -> list[dict]:
    """Look up sellable vs. on-hand stock for a SKU, optionally scoped to
    one warehouse. Read-only. Always use sellable_qty for fulfillment
    decisions, never on_hand_qty alone."""
    db = SessionLocal()
    try:
        return wms.get_stock(db, sku, warehouse)
    finally:
        db.close()


@server.tool()
def transfer_stock(sku: str, from_warehouse: str, to_warehouse: str,
                    qty: float, idempotency_key: str) -> dict:
    """Transfer stock between warehouses. WRITE - requires idempotency_key.
    A retried call with the same key returns the original result without
    double-decrementing stock."""
    db = SessionLocal()
    try:
        return wms.transfer_stock(db, sku, from_warehouse, to_warehouse, qty, idempotency_key)
    finally:
        db.close()


@server.tool()
def get_transaction_status(payment_intent_id: str) -> dict:
    """Look up a payment transaction's status. Read-only."""
    gateway = payment.get_payment_gateway()
    return gateway.get_transaction_status(payment_intent_id)


@server.tool()
def issue_refund(payment_intent_id: str, amount_usd: float, idempotency_key: str) -> dict:
    """Issue a refund. WRITE - requires idempotency_key. A retried call
    with the same key returns the original refund result without issuing
    a second refund."""
    db = SessionLocal()
    try:
        gateway = payment.get_payment_gateway()
        return gateway.issue_refund(db, payment_intent_id, amount_usd, idempotency_key)
    finally:
        db.close()


@server.tool()
def get_tracking_status(tracking_number: str) -> dict:
    """Look up carrier tracking status. Read-only."""
    gateway = carrier.get_carrier_gateway()
    return gateway.get_tracking_status(tracking_number)


@server.tool()
def generate_return_label(order_id: str, idempotency_key: str) -> dict:
    """Generate a return shipping label. WRITE - requires idempotency_key."""
    db = SessionLocal()
    try:
        gateway = carrier.get_carrier_gateway()
        return gateway.generate_return_label(db, order_id, idempotency_key)
    finally:
        db.close()


@server.tool()
def send_customer_notification(customer_id: str, channel: str, subject: str, body: str) -> dict:
    """Send a customer notification. channel: 'email' | 'sms'. Currently
    logs and records in-memory rather than sending (see notification.py)."""
    return notification.send_notification(customer_id, channel, subject, body)


if __name__ == "__main__":
    server.run()
