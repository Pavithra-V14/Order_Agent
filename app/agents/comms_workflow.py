"""
Customer-Comms workflow - architecture doc Part 3 topology. Templated
messages (LLM-personalization is a swap point, not required for this to
be functional) triggered on major state transitions: escalated,
resolved (refund/reship/deny), pending_retry.
"""
from __future__ import annotations

from app.tools.notification import send_notification


_TEMPLATES = {
    "escalated": "We've received your request and it's being reviewed by our team. "
                 "We'll follow up within 1 business day.",
    "resolved_refund": "Good news - your refund of ${amount:.2f} has been processed and will "
                        "appear on your original payment method within 5-7 business days.",
    "resolved_partial_credit": "We've issued a partial credit of ${amount:.2f} for your order. "
                                "Details are available in your account.",
    "resolved_reship": "We're sending a replacement - track your package with tracking number {tracking_number}.",
    "resolved_deny": "After review, we're unable to approve this request. If you believe this is "
                      "in error, please reply to this message.",
    "pending_retry": "We're processing your request - this is taking a bit longer than usual, "
                      "but no action is needed from you right now.",
}

_SUBJECT_MAP = {
    "escalated": "Your request is being reviewed",
    "resolved_refund": "Your refund has been processed",
    "resolved_partial_credit": "Your credit has been issued",
    "resolved_reship": "Your replacement is on the way",
    "resolved_deny": "Update on your request",
    "pending_retry": "Your request is still being processed",
}


def send_case_notification(customer_id: str, event: str, **kwargs) -> dict:
    """event selects the template; kwargs fill in template variables
    (amount, tracking_number, etc). Swap point for LLM personalization:
    replace the plain .format() call with a generation-tier LLM call
    that rewrites the templated body in a warmer tone while keeping the
    same factual content - the template stays as the source of truth for
    WHAT is said, the LLM only changes HOW."""
    template = _TEMPLATES.get(event)
    if template is None:
        raise ValueError(f"No template for event {event!r}. Known events: {list(_TEMPLATES.keys())}")
    body = template.format(**kwargs)
    return send_notification(customer_id, channel="email", subject=_SUBJECT_MAP[event], body=body)
