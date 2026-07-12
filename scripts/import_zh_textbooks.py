"""将 GPU 算好的中文教科书向量导入本地 Milvus"""
import json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from config import cfg
from rag.milvus_client import MilvusManager

TEXT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "textbooks", "zh_raw", "data_clean", "textbooks", "zh_paragraph", "all_books.txt")
VEC_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "textbooks", "zh_vectors.json")
BATCH = 500

# Load texts
print("Loading texts...")
with open(TEXT_PATH, "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]
print(f"Texts: {len(texts)}")

# Load vectors
print("Loading vectors...")
with open(VEC_PATH, "r", encoding="utf-8") as f:
    vectors = json.load(f)
print(f"Vectors: {len(vectors)}")

# Connect Milvus
print("Connecting to Milvus...")
m = MilvusManager(uri="http://localhost:19530")
m.connect()
kb = cfg.milvus.collections.kb
if not m.client.has_collection(kb):
    m.create_kb_collection()
    print(f"Created collection: {kb}")

# Build rows and insert
total = min(len(texts), len(vectors))
inserted = 0
t0 = time.time()
rows_buffer = []

for i in range(total):
    chunk_id = f"zh_paragraph_{i}"
    vec = vectors.get(chunk_id, {}).get("embedding")
    if vec is None:
        continue

    rows_buffer.append({
        "doc_id": chunk_id,
        "embedding": vec,
        "content": texts[i],
        "source": "Chinese_Medical_Textbooks",
        "department": "全科",
        "contraindications": "",
    })

    if len(rows_buffer) >= BATCH:
        try:
            m.insert(kb, rows_buffer)
            inserted += len(rows_buffer)
        except Exception as e:
            print(f"Insert error: {e}")
        rows_buffer = []
        elapsed = time.time() - t0
        print(f"Progress: {inserted}/{total} ({100*inserted/total:.1f}%) | {inserted/elapsed:.0f} rec/s")

# Final batch
if rows_buffer:
    try:
        m.insert(kb, rows_buffer)
        inserted += len(rows_buffer)
    except Exception as e:
        print(f"Insert error: {e}")

elapsed = time.time() - t0
print(f"\nDone! {inserted} records in {elapsed:.1f}s")

# Verify
stats = m.client.get_collection_stats(kb)
print(f"Collection stats: {stats.get('row_count', '?')} total rows")
