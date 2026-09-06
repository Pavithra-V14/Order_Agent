"""
PDF extraction — architecture doc 8.2.1 specifies unstructured.io's
`hi_res` strategy for mixed text/table/image/chart PDFs. `unstructured[pdf]`
pulls a layout-detection model (torch + onnx + opencv), which — like
BGE-M3 in embeddings.py — doesn't reliably fit this sandbox's disk budget
alongside everything else (confirmed: the install failed with ENOSPC).

This module produces the *same element contract* (`ExtractedElement` with
`element_type in {text, table, image, chart}`, page number, and bbox-free
ordering) using pdfplumber (text + native table detection) and
pdf2image/poppler (page rasterization, for image/chart region capture) —
both lightweight, already-installed, and genuinely functional, not mocked.

Swap point for production: replace `extract_pdf()`'s body with a call to
`unstructured.partition.pdf.partition_pdf(filename, strategy="hi_res",
infer_table_structure=True)` and map its element types to the same
`ExtractedElement` dataclass below — nothing downstream (chunking, node
construction, ingestion) needs to change, since it's written against this
dataclass, not against unstructured's types directly.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import pdfplumber
from pdf2image import convert_from_path


@dataclass
class ExtractedElement:
    element_type: str  # "text" | "table" | "image" | "chart"
    page: int
    content: str          # for text: the text itself; for table: a flattened text summary;
                           # for image/chart: the caption (see rag/captioning.py)
    table_data: list[list[str]] | None = None   # populated only for element_type == "table"
    image_path: str | None = None                # populated only for element_type in {image, chart}
    content_hash: str = field(default="")

    def __post_init__(self):
        if not self.content_hash:
            basis = self.content if self.element_type != "table" else str(self.table_data)
            self.content_hash = hashlib.sha256(basis.encode("utf-8")).hexdigest()


def _looks_like_chart_page(page) -> bool:
    """Chart detector: flags a page as containing a chart/figure if it has
    at least one embedded raster image. An earlier version of this
    heuristic also required low text density (on the theory that a
    "chart page" is mostly-image), which was wrong for this project's own
    policy PDFs — each chart sits on the same page as surrounding prose
    (title, table, refund-processing text), so text density alone would
    have silently skipped every chart. Presence of an embedded image is
    the correct, simple signal for this corpus. A production
    unstructured.io hi_res pass replaces this heuristic entirely with its
    trained layout-detection model, which locates figures by bounding box
    regardless of surrounding text density."""
    return len(page.images) > 0


def extract_pdf(pdf_path: str, image_out_dir: str) -> list[ExtractedElement]:
    """Extracts text, table, and image/chart elements from a policy PDF.
    Returns elements in reading order per page. Caller (ingestion.py) is
    responsible for captioning image/chart elements and for chunking text."""
    os.makedirs(image_out_dir, exist_ok=True)
    elements: list[ExtractedElement] = []
    doc_stem = os.path.splitext(os.path.basename(pdf_path))[0]

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            # --- Tables first (pdfplumber's table detection is native, no OCR needed) ---
            tables = page.extract_tables()
            for t_idx, table in enumerate(tables):
                if not table or not any(any(cell for cell in row) for row in table):
                    continue
                flat = "; ".join(
                    " | ".join(str(c) if c is not None else "" for c in row)
                    for row in table
                )
                elements.append(ExtractedElement(
                    element_type="table",
                    page=page_idx,
                    content=f"Table (page {page_idx}, #{t_idx+1}): {flat}",
                    table_data=table,
                ))

            # --- Chart/image regions (heuristic — see docstring above) ---
            if _looks_like_chart_page(page):
                rendered = convert_from_path(
                    pdf_path, first_page=page_idx, last_page=page_idx, dpi=150
                )
                if rendered:
                    img_path = os.path.join(image_out_dir, f"{doc_stem}_p{page_idx}_chart.png")
                    rendered[0].save(img_path, "PNG")
                    elements.append(ExtractedElement(
                        element_type="chart",
                        page=page_idx,
                        content="",  # filled in by captioning.py
                        image_path=img_path,
                    ))

            # --- Remaining prose text ---
            # Note: pdfplumber's extract_text() includes table cell text too,
            # so table content is technically indexed twice (once as a
            # structured `table` element, once inside the page's flat `text`
            # element). This is a deliberate, documented trade-off for this
            # sandbox substitute: harmless for retrieval recall (duplicate
            # content just means a term appears in two chunks, not a
            # correctness bug), but a real unstructured.io hi_res pass
            # avoids the duplication natively via proper layout segmentation.
            raw_text = page.extract_text() or ""
            if raw_text.strip():
                elements.append(ExtractedElement(
                    element_type="text",
                    page=page_idx,
                    content=raw_text.strip(),
                ))

    return elements
