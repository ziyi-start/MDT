"""预计算种子知识库向量并持久化到 backend/seed_vectors.json

使用方式:
  python scripts/precompute_vectors.py

说明:
  - 加载 BGE-small-zh-v1.5 模型，对 SEED_KB 每条知识计算 512 维向量
  - 结果写入 backend/seed_vectors.json，供 seed_data.py 启动时直接读取
  - 避免启动时重复加载 embedding 模型和计算向量
"""
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from rag.embedding import _get_embedding_model, _encode
from seed_data import SEED_KB

def main():
    model = _get_embedding_model()
    dim = model.get_embedding_dimension()
    print(f"Model loaded, dim={dim}")

    data = {}
    for doc in SEED_KB:
        vec = list(_encode(doc["content"]))
        data[doc["doc_id"]] = {"embedding": vec}
        print(f"  {doc['doc_id']}: done")

    out = os.path.join(os.path.dirname(__file__), "..", "backend", "seed_vectors.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Saved {len(data)} vectors to {out}")

if __name__ == "__main__":
    main()
