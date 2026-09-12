"""
Phase 2 — generates the policy PDF corpus used to test the RAG pipeline
(Phase 3), especially the temporal-versioning edge case from the
architecture doc: an order placed under RET-POLICY-2025-A (6-month window)
must still resolve against that version after RET-POLICY-2026-A (4-month
window) supersedes it.

Run: python3 scripts/generate_policy_pdfs.py
Output: data/policies/*.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")  # no display in this sandbox
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "policies")
os.makedirs(OUT_DIR, exist_ok=True)

styles = getSampleStyleSheet()
meta_style = ParagraphStyle("meta", parent=styles["Normal"], textColor=colors.HexColor("#555555"), fontSize=9)


def _chart_path(name: str, categories: list[str], values: list[float], ylabel: str, title: str) -> str:
    """Generates a bar chart PNG to embed as a real chart element (not just
    a table dressed up) — exercises the ingestion pipeline's image/chart
    extraction + captioning path in Phase 3."""
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig, ax = plt.subplots(figsize=(5.5, 3))
    bars = ax.bar(categories, values, color="#2d5f8a")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.bar_label(bars, fmt="%.0f")
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _table(data, col_widths=None):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d5f8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


# ---------------------------------------------------------------------------
# RET-POLICY-2025-A  (effective 2025-01-01 -> 2026-01-31, 6-month window)
# ---------------------------------------------------------------------------
def build_return_policy_2025a():
    doc_path = os.path.join(OUT_DIR, "RET-POLICY-2025-A.pdf")
    doc = SimpleDocTemplate(doc_path, pagesize=letter, topMargin=0.7*inch, bottomMargin=0.7*inch)
    story = []

    story.append(Paragraph("Standard Return &amp; Refund Policy", styles["Title"]))
    story.append(Paragraph("Document ID: RET-POLICY-2025-A &nbsp;|&nbsp; Version: 1 &nbsp;|&nbsp; "
                            "Effective: 2025-01-01 to 2026-01-31 &nbsp;|&nbsp; Superseded by: RET-POLICY-2026-A",
                            meta_style))
    story.append(Paragraph(
        "Return Windows (days): apparel=180, footwear=180, electronics=30, home=180, beauty=45, all=180",
        meta_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph(
        "This policy governs returns and refunds for all orders placed between "
        "January 1, 2025 and January 31, 2026 (inclusive), across all sales channels "
        "unless a channel-specific policy states otherwise. The return window is "
        "measured from the customer's order delivery date, not the order date.",
        styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Return Windows by Category", styles["Heading2"]))
    story.append(_table([
        ["Product Category", "Return Window", "Restocking Fee", "Condition Required"],
        ["Apparel & Accessories", "180 days (6 months)", "None", "Unworn, tags attached"],
        ["Footwear", "180 days (6 months)", "None", "Unworn, original box"],
        ["Electronics", "30 days", "15% if opened", "Original packaging"],
        ["Home & Furniture", "180 days (6 months)", "10%", "Undamaged"],
        ["Beauty & Personal Care", "45 days", "None", "Unopened / unused"],
    ], col_widths=[1.7*inch, 1.5*inch, 1.3*inch, 1.7*inch]))
    story.append(Spacer(1, 14))

    chart = _chart_path(
        "chart_2025a_windows",
        ["Apparel", "Footwear", "Electronics", "Home", "Beauty"],
        [180, 180, 30, 180, 45],
        "Return window (days)",
        "Return Window by Category — Policy 2025-A",
    )
    story.append(Paragraph("Return Window Comparison", styles["Heading2"]))
    story.append(Image(chart, width=5*inch, height=2.7*inch))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Refund Processing", styles["Heading2"]))
    story.append(Paragraph(
        "Approved refunds are issued to the original payment method within 5-7 business "
        "days of the returned item being received and inspected. Refunds are issued for "
        "the exact amount originally charged; currency conversion, if applicable, uses the "
        "rate in effect at the time of the original transaction, not the refund date.",
        styles["Normal"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Version-binding note:</b> the policy version in effect at the time of purchase "
        "governs the return for the life of that order, even if this document is later "
        "superseded. See RET-POLICY-2026-A for the version effective 2026-02-01 onward.",
        styles["Normal"]))

    doc.build(story)
    return doc_path


# ---------------------------------------------------------------------------
# RET-POLICY-2026-A  (effective 2026-02-01 onward, 4-month window)
# ---------------------------------------------------------------------------
def build_return_policy_2026a():
    doc_path = os.path.join(OUT_DIR, "RET-POLICY-2026-A.pdf")
    doc = SimpleDocTemplate(doc_path, pagesize=letter, topMargin=0.7*inch, bottomMargin=0.7*inch)
    story = []

    story.append(Paragraph("Standard Return &amp; Refund Policy", styles["Title"]))
    story.append(Paragraph("Document ID: RET-POLICY-2026-A &nbsp;|&nbsp; Version: 2 &nbsp;|&nbsp; "
                            "Effective: 2026-02-01 to present &nbsp;|&nbsp; Supersedes: RET-POLICY-2025-A",
                            meta_style))
    story.append(Paragraph(
        "Return Windows (days): apparel=120, footwear=120, electronics=30, home=120, beauty=45, all=120",
        meta_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph(
        "This policy governs returns and refunds for all orders placed on or after "
        "February 1, 2026, across all sales channels unless a channel-specific policy "
        "states otherwise. <b>Orders placed before February 1, 2026 remain governed by "
        "RET-POLICY-2025-A for the life of that order</b> — this update does not "
        "retroactively shorten the return window for prior purchases.",
        styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Return Windows by Category", styles["Heading2"]))
    story.append(_table([
        ["Product Category", "Return Window", "Restocking Fee", "Condition Required"],
        ["Apparel & Accessories", "120 days (4 months)", "None", "Unworn, tags attached"],
        ["Footwear", "120 days (4 months)", "None", "Unworn, original box"],
        ["Electronics", "30 days", "15% if opened", "Original packaging"],
        ["Home & Furniture", "120 days (4 months)", "10%", "Undamaged"],
        ["Beauty & Personal Care", "45 days", "None", "Unopened / unused"],
    ], col_widths=[1.7*inch, 1.5*inch, 1.3*inch, 1.7*inch]))
    story.append(Spacer(1, 14))

    chart = _chart_path(
        "chart_2026a_windows",
        ["Apparel", "Footwear", "Electronics", "Home", "Beauty"],
        [120, 120, 30, 120, 45],
        "Return window (days)",
        "Return Window by Category — Policy 2026-A",
    )
    story.append(Paragraph("Return Window Comparison", styles["Heading2"]))
    story.append(Image(chart, width=5*inch, height=2.7*inch))
    story.append(Spacer(1, 14))

    story.append(Paragraph("Refund Processing", styles["Heading2"]))
    story.append(Paragraph(
        "Approved refunds are issued to the original payment method within 5-7 business "
        "days of the returned item being received and inspected. Refunds are issued for "
        "the exact amount originally charged.",
        styles["Normal"]))

    doc.build(story)
    return doc_path


# ---------------------------------------------------------------------------
# FRAUD-POLICY-2025-A
# ---------------------------------------------------------------------------
def build_fraud_policy():
    doc_path = os.path.join(OUT_DIR, "FRAUD-POLICY-2025-A.pdf")
    doc = SimpleDocTemplate(doc_path, pagesize=letter, topMargin=0.7*inch, bottomMargin=0.7*inch)
    story = []

    story.append(Paragraph("Fraud Prevention &amp; Risk Threshold Policy", styles["Title"]))
    story.append(Paragraph("Document ID: FRAUD-POLICY-2025-A &nbsp;|&nbsp; Version: 1 &nbsp;|&nbsp; "
                            "Effective: 2025-01-01 to present", meta_style))
    story.append(Spacer(1, 14))

    story.append(Paragraph(
        "This policy defines the risk signals and thresholds used to route order "
        "exception cases to human review versus automated resolution. Risk scoring "
        "considers behavioral patterns, not return frequency alone — a customer's "
        "lifetime value and return-reason consistency are weighted alongside raw "
        "return count to avoid penalizing legitimate high-engagement customers.",
        styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("Mandatory Human Review Triggers", styles["Heading2"]))
    story.append(_table([
        ["Signal", "Threshold", "Action"],
        ["Fraud/risk score", "Any flag present", "Always route to human review, regardless of case value"],
        ["Address changed same-day as return request", "Present", "Flag for identity-consistency review"],
        ["\"Item never arrived\" vs. carrier \"delivered\"", "Any occurrence", "Route to human review with delivery evidence attached"],
        ["Return count", "> 10 in 90 days", "Review for pattern, weighted with LTV and reason consistency"],
    ], col_widths=[2.6*inch, 1.7*inch, 2.4*inch]))
    story.append(Spacer(1, 14))

    chart = _chart_path(
        "chart_fraud_distribution",
        ["Score 0-0.3\n(low)", "Score 0.3-0.6\n(medium)", "Score 0.6-0.85\n(elevated)", "Score 0.85-1.0\n(high)"],
        [72, 19, 6, 3],
        "% of cases (illustrative)",
        "Illustrative Fraud/Risk Score Distribution",
    )
    story.append(Paragraph("Risk Score Distribution (Illustrative)", styles["Heading2"]))
    story.append(Image(chart, width=5*inch, height=2.7*inch))

    doc.build(story)
    return doc_path


if __name__ == "__main__":
    paths = [
        build_return_policy_2025a(),
        build_return_policy_2026a(),
        build_fraud_policy(),
    ]
    for p in paths:
        print("Generated:", p)
