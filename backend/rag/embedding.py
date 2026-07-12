"""共享 Embedding 工具

使用 BGE-small-zh-v1.5 中文 Embedding 模型生成真实语义向量（512 维）。
生产环境可替换为其他 BGE 系列模型或领域微调模型。
"""
from __future__ import annotations

import asyncio
import re
import logging
from functools import lru_cache

from config import cfg

logger = logging.getLogger(__name__)

# 向量维度，从配置读取
EMBEDDING_DIM = cfg.milvus.embedding_dim

# 全局 Embedding 模型（延迟加载）
_EMBEDDING_MODEL = None


def _get_embedding_model():
    """延迟加载 BGE Embedding 模型"""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("正在加载 BGE Embedding 模型...")
            _EMBEDDING_MODEL = SentenceTransformer(
                'BAAI/bge-small-zh-v1.5',
                device=device,
            )
            dim = _EMBEDDING_MODEL.get_embedding_dimension()
            logger.info(f"BGE 模型加载成功，向量维度: {dim}, 设备: {device}")
        except Exception as e:
            logger.error(f"BGE 模型加载失败: {e}")
            raise
    return _EMBEDDING_MODEL


@lru_cache(maxsize=256)
def _encode(text: str) -> tuple[float, ...]:
    """带缓存的编码（相同文本避免重复计算）"""
    model = _get_embedding_model()
    vec = model.encode(text, normalize_embeddings=True)
    return tuple(float(x) for x in vec)


async def dummy_embed(text: str) -> list[float]:
    """BGE-small-zh-v1.5 Embedding

    生成真实语义向量，512 维，L2 归一化。
    用 named dummy_embed 保持与现有代码兼容。
    """
    return await asyncio.to_thread(lambda: list(_encode(text)))


@lru_cache(maxsize=512)
def extract_chinese_terms(text: str) -> frozenset[str]:
    """提取中文文本中的关键词 (2-4 字 n-gram + 英文词)
    用于内存模式下的关键词匹配检索。
    """
    chinese_chars = re.findall(r'[一-鿿]+', text)
    english_words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    terms = set(english_words)
    for segment in chinese_chars:
        for n in range(2, min(5, len(segment) + 1)):
            for i in range(len(segment) - n + 1):
                terms.add(segment[i:i + n])
    return terms