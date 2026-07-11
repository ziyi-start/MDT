"""Re-seed Milvus with clean data"""
import sys, asyncio, json, os
sys.path.insert(0, r'D:\pythonProject\MDT-1\MDT\backend')
os.environ['MDT_USE_MILVUS'] = 'true'

from config import cfg
from rag.milvus_client import MilvusManager
from seed_data import SEED_KB

async def reseed():
    milvus = MilvusManager(uri=cfg.milvus.uri)
    milvus.connect()

    # Drop and recreate KB collection
    try:
        milvus.client.drop_collection(cfg.milvus.collections.kb)
        print(f'Dropped collection: {cfg.milvus.collections.kb}')
    except Exception as e:
        print(f'Drop failed (maybe first time): {e}')

    milvus.create_kb_collection()
    print('Created KB collection')

    # Load seed vectors
    vectors_path = r'D:\pythonProject\MDT-1\MDT\backend\seed_vectors.json'
    with open(vectors_path, 'r', encoding='utf-8') as f:
        vectors = json.load(f)

    # Insert data
    data = []
    for doc in SEED_KB:
        vec_data = vectors.get(doc["doc_id"])
        if vec_data is None:
            print(f'Missing vector: {doc["doc_id"]}')
            continue
        data.append({
            "doc_id": doc["doc_id"],
            "embedding": vec_data["embedding"],
            "content": doc["content"],
            "source": doc["source"],
            "department": doc["department"],
            "contraindications": doc["contraindications"],
        })

    milvus.insert(cfg.milvus.collections.kb, data)
    print(f'Inserted {len(data)} docs')

    # Verify by searching
    from rag.embedding import dummy_embed
    vec = await dummy_embed('布洛芬')
    results = milvus.search(collection_name=cfg.milvus.collections.kb, vector=vec, limit=3)
    print(f'\nVerification search: {len(results)} results')
    for r in results:
        content = r.get('content', '')
        print(f'  id={r.get("id")} content[:40]={content[:40]!r}')

asyncio.run(reseed())
