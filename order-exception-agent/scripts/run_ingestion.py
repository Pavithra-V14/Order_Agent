"""
Phase 3 convenience CLI: python3 scripts/run_ingestion.py
Ingests every PDF in data/policies/ into the local Qdrant collection.
Safe to re-run — incremental reindexing (8.2.5) means unchanged docs cost
nothing on repeat runs.
"""
import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
logging.basicConfig(level=logging.INFO, format="%(message)s")

from app.rag.ingestion import ingest_policy_directory

if __name__ == "__main__":
    summaries = ingest_policy_directory()
    print(json.dumps(summaries, indent=2))
