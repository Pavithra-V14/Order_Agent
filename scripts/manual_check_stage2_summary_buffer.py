"""
MANUAL CHECK - Stage 2: Summary Buffer real LLM fold + persistence.

Two things this checks, DIRECTLY - not by hoping a real diagnosis run
happens to produce more than 6 steps (the buffer's fold threshold),
which is hard to force reliably through a single realistic case:

  1. SummaryBuffer.add(llm=...) genuinely calls the LLM to fold an aged-
     out item into the running summary, when a real LLM is configured
     (GROQ_API_KEY set) - not the old deterministic string-concat stub.
  2. persist_buffer_state() / load_buffer_state() genuinely round-trip
     through ExceptionCase.working_memory_summary, surviving what a
     process restart would look like (a fresh SummaryBuffer object,
     loaded back from the DB row, not the same in-memory object).

Usage:
    python scripts/manual_check_stage2_summary_buffer.py
"""
import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.core.db import SessionLocal, ExceptionCase, CaseState, init_db
from app.memory.summary_buffer import SummaryBuffer, persist_buffer_state, load_buffer_state, reset_buffers
from app.agents.llm_client import get_llm_client


def main():
    init_db()
    db = SessionLocal()
    reset_buffers()

    case_id = "case-stage2-manual-check"
    case = db.get(ExceptionCase, case_id)
    if not case:
        case = ExceptionCase(id=case_id, order_id="ORD-STAGE2-DEMO", customer_id="CUST-STAGE2-DEMO",
                              channel="direct", exception_type="return", state=CaseState.DIAGNOSING)
        db.add(case)
        db.commit()

    llm = get_llm_client()
    print(f"Using LLM client: {type(llm).__name__}")
    if type(llm).__name__ == "FakeLLMClient":
        print("NOTE: no real LLM key configured here - summarize_context() will use")
        print("the deterministic fold (FakeLLMClient's real, tested behavior), not a")
        print("genuine LLM call. Set GROQ_API_KEY to see the real fold happen.")

    buf = SummaryBuffer(case_id=case_id, max_verbatim_items=2)  # small on purpose, to force a fold quickly
    print("\nAdding 4 items (max_verbatim_items=2, so items 1 and 2 will get folded)...")
    buf.add({"agent": "diagnosis", "summary": "Checked payment status: succeeded, no anomaly."}, llm=llm)
    buf.add({"agent": "diagnosis", "summary": "Checked inventory: sufficient stock at WH-A."}, llm=llm)
    buf.add({"agent": "diagnosis", "summary": "Checked carrier tracking: delivered on time."}, llm=llm)
    buf.add({"agent": "diagnosis", "summary": "Concluded: no anomaly detected, safe to auto-refund."}, llm=llm)

    print(f"\nRunning summary after 4 adds (2 folded in):\n  {buf.running_summary!r}")
    print(f"Verbatim items still held (should be the last 2):")
    for item in buf.verbatim_items:
        print(f"  - {item}")

    persist_buffer_state(db, case_id, buf)
    print(f"\nPersisted to ExceptionCase.working_memory_summary for case_id={case_id}")

    # Simulate a process restart: load a BRAND NEW buffer object back
    # from the DB row, not the same in-memory `buf` instance.
    reset_buffers()
    reloaded = load_buffer_state(db, case_id)
    print(f"\nReloaded (simulated restart) running_summary:\n  {reloaded.running_summary!r}")
    assert reloaded.running_summary == buf.running_summary, "persistence round-trip did not match!"
    print("\nPersistence round-trip confirmed: reloaded summary matches what was persisted.")

    print("\n" + "=" * 70)
    print(f"NOW CHECK: GET /api/v1/cases/{case_id} in Swagger UI (/docs)")
    print("Look for working_memory_summary (or case_summary) in the response -")
    print("it should contain the same folded text printed above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
