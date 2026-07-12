"""将 MedQA-US 医学教科书语料库导入 Milvus Medical_KB 集合

数据来源:
  - data/textbooks/all_chunks.jsonl (125,847 条，18 本英文教科书)
  - Jin et al. 2021 MedQA-US

使用方式:
  python scripts/import_textbooks.py --reset
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from config import cfg
from rag.milvus_client import MilvusManager
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "textbooks", "all_chunks.jsonl")
ENCODE_BATCH = 500
INSERT_BATCH = 200

DEPARTMENT_MAP = {
    "Anatomy_Gray": "解剖学",
    "Biochemistry_Lippincott": "生物化学",
    "Cell_Biology_Alberts": "细胞生物学",
    "First_Aid_Step1": "全科",
    "First_Aid_Step2": "全科",
    "Gynecology_Novak": "妇产科",
    "Histology_Ross": "组织学",
    "Immunology_Janeway": "免疫学",
    "InternalMed_Harrison": "内科学",
    "Neurology_Adams": "神经病学",
    "Obstentrics_Williams": "妇产科",
    "Pathology_Robbins": "病理学",
    "Pathoma_Husain": "病理学",
    "Pediatrics_Nelson": "儿科学",
    "Pharmacology_Katzung": "药理学",
    "Physiology_Levy": "生理学",
    "Psichiatry_DSM-5": "精神病学",
    "Surgery_Schwartz": "外科学",
}


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    logger.info(f"已加载 {len(chunks)} 条文本块")
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="清空集合后重新导入")
    args = parser.parse_args()

    # 连接 Milvus
    uri = os.getenv("MDT_MILVUS_URI", cfg.milvus.uri)
    milvus = MilvusManager(uri=uri)
    milvus.connect()
    logger.info(f"Milvus: {uri}")

    # 重置/创建集合
    collection_name = cfg.milvus.collections.kb
    if args.reset and milvus.client.has_collection(collection_name):
        milvus.client.drop_collection(collection_name)
        logger.info(f"已删除集合 {collection_name}")
    if not milvus.client.has_collection(collection_name):
        milvus.create_kb_collection()
        logger.info(f"已创建集合 {collection_name}")

    # 加载模型
    logger.info("加载 BGE-small-zh-v1.5 ...")
    model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")
    logger.info("模型加载完成")

    # 加载数据
    chunks = load_chunks()
    total = len(chunks)
    inserted = 0
    t0 = time.time()

    # 先把所有文本收集好，分大 batch 编码，再分小 batch 插入
    for i in range(0, total, ENCODE_BATCH):
        encode_batch = chunks[i:i + ENCODE_BATCH]
        texts = [c["content"] for c in encode_batch]

        # 批量编码 (ONNX)
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        if isinstance(embeddings, np.ndarray):
            embeddings = embeddings.tolist()

        # 再分小批插入 Milvus
        for j in range(0, len(encode_batch), INSERT_BATCH):
            sub = encode_batch[j:j + INSERT_BATCH]
            sub_emb = embeddings[j:j + INSERT_BATCH]
            rows = []
            for c, emb in zip(sub, sub_emb):
                rows.append({
                    "doc_id": c["id"],
                    "embedding": emb,
                    "content": c["content"],
                    "source": c["title"],
                    "department": DEPARTMENT_MAP.get(c["title"], "全科"),
                    "contraindications": "",
                })
            try:
                milvus.insert(collection_name, rows)
                inserted += len(rows)
            except Exception as e:
                logger.warning(f"插入失败: {e}")

        elapsed = time.time() - t0
        logger.info(f"{inserted}/{total} ({100*inserted/total:.1f}%) | {inserted/elapsed:.1f} 条/秒 | {elapsed:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"完成! 共 {inserted} 条, 耗时 {elapsed:.0f}s ({elapsed/60:.1f} 分)")


if __name__ == "__main__":
    main()
