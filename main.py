# def main():
#     print("Hello from order-exception-agent!")


# if __name__ == "__main__":
#     main()

"""
Run: uv run python diagnose_ingestion.py <filename.pdf>

Checks, in order: (1) whether your installed code actually has the
recent fixes, (2) what reindex_state.json really contains for this
file, (3) whether Qdrant genuinely has points for it.
"""
import sys
import json
import os

sys.path.insert(0, ".")

filename = sys.argv[1] if len(sys.argv) > 1 else input("Filename: ")

print("=== Step 1: Is your installed code up to date? ===")
with open("app/api/v1/policies.py") as f:
    policies_src = f.read()
has_source_filename_fix = "source_filename" in policies_src and "_resolve_doc_id_for_filename" in policies_src
print(f"list_policies/delete_policy source_filename fix present: {has_source_filename_fix}")
if not has_source_filename_fix:
    print(">>> Your app/api/v1/policies.py does NOT have the recent fix. Re-extract the latest zip.")
    sys.exit(1)

with open("app/rag/chunking.py") as f:
    chunking_src = f.read()
has_cdc_fix = "content-defined chunking" in chunking_src.lower() or "content_defined" in chunking_src.lower()
print(f"Content-defined chunking fix present: {has_cdc_fix}")

with open("app/static/app.js") as f:
    js_src = f.read()
has_toast_fix = "chunks embedded" in js_src
print(f"UI toast chunk-count fix present: {has_toast_fix}")
if not has_toast_fix:
    print(">>> Your app/static/app.js does NOT have the toast fix - the browser toast won't show counts even if ingestion succeeds.")

print("\n=== Step 2: What does reindex_state.json actually say? ===")
from app.api.v1.policies import _resolve_doc_id_for_filename, POLICY_UPLOAD_DIR
full_path = os.path.join(POLICY_UPLOAD_DIR, filename)
if not os.path.exists(full_path):
    print(f"ERROR: {full_path} does not exist on disk at all.")
    sys.exit(1)

resolved_doc_id = _resolve_doc_id_for_filename(filename, full_path)
print(f"Resolved doc_id for '{filename}': {resolved_doc_id!r}")

reindex_path = "data/reindex_state.json"
if not os.path.exists(reindex_path):
    print(">>> data/reindex_state.json does not exist at all - ingestion never wrote anything.")
else:
    with open(reindex_path) as f:
        state = json.load(f)
    entry = state.get(resolved_doc_id)
    if entry is None:
        print(f">>> No entry for doc_id={resolved_doc_id!r} in reindex_state.json - ingestion did NOT complete successfully.")
    else:
        print(f"Entry found. source_filename recorded: {entry.get('source_filename', '(MISSING - predates the fix, or something wrote this entry without it)')}")
        print(f"  hashes count: {len(entry.get('hashes', {}))}")
        print(f"  indexed_at: {entry.get('indexed_at')}")

print("\n=== Step 3: Does Qdrant actually have points for this doc_id? ===")
from app.core.config import get_settings
from app.rag.vectorstore import get_qdrant_client
from qdrant_client.models import Filter, FieldCondition, MatchValue

settings = get_settings()
print(f"QDRANT_URL configured: {bool(settings.qdrant_url)}")
client = get_qdrant_client()
try:
    count = client.count(
        collection_name=settings.qdrant_collection,
        count_filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=resolved_doc_id))]),
    ).count
    print(f"Real points in Qdrant for doc_id={resolved_doc_id!r}: {count}")
except Exception as e:
    print(f">>> Qdrant count FAILED: {e}")