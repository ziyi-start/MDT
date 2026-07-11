"""Test Milvus retrieval and Cross-Encoder rerank"""
import sys, asyncio
sys.path.insert(0, r'D:\pythonProject\MDT-1\MDT\backend')
from rag.milvus_client import MilvusManager
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import MedicalReranker
from config import cfg
from rag.embedding import dummy_embed

async def test():
    milvus = MilvusManager(uri=cfg.milvus.uri)
    milvus.connect()
    retriever = HybridRetriever(milvus)
    reranker = MedicalReranker()

    query = '布洛芬的常见不良反应有哪些'

    # Test 1: Raw Milvus search
    vec = await dummy_embed(query)
    raw = milvus.search(collection_name=cfg.milvus.collections.kb, vector=vec, limit=5)
    print('=== Raw Milvus search ===')
    for r in raw:
        content = r.get('content', '')
        print(f'  id={r.get("id")} score={r.get("score"):.4f} content_len={len(content)}')
        print(f'  content[:60]: {content[:60]!r}')

    # Test 2: Hybrid retriever
    docs = await retriever.retrieve(query, top_k=5)
    print('\n=== Hybrid retriever ===')
    for d in docs:
        print(f'  doc_id={d.doc_id} score={d.score:.4f}')
        print(f'  content[:60]: {d.content[:60]!r}')

    # Test 3: Rerank
    reranked = await reranker.rerank(query, docs, top_k=5)
    print('\n=== After Rerank ===')
    for d in reranked:
        print(f'  doc_id={d.doc_id} score={d.score:.4f}')
        print(f'  content[:60]: {d.content[:60]!r}')

asyncio.run(test())
