"""医疗 Reranker - 重排序与 CoT 安全退避判断

功能:
1. 对混合检索结果进行重排序，打压临床逻辑无关的噪声
2. 判断检索结果是否足以支撑可靠回答（CoT 退避条件）

使用 BGE-Reranker-v2-m3 跨编码器模型进行 query-document pair 相关性打分。
"""
from __future__ import annotations

import logging

from schema.models import DocumentChunk
from rag.embedding import extract_chinese_terms
from config import cfg

logger = logging.getLogger(__name__)

# 全局 Cross-Encoder 模型（延迟加载）
_CROSS_ENCODER = None


def _get_cross_encoder():
    """延迟加载 BGE-Reranker Cross-Encoder 模型"""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            logger.info("正在加载 BGE-Reranker 模型...")
            _CROSS_ENCODER = CrossEncoder(
                'BAAI/bge-reranker-v2-m3',
                device='cpu',
            )
            logger.info("BGE-Reranker 模型加载成功")
        except Exception as e:
            logger.error(f"BGE-Reranker 模型加载失败: {e}")
            raise
    return _CROSS_ENCODER


class MedicalReranker:
    """医疗重排序器

    重排策略:
    - 主方案: BGE-Reranker Cross-Encoder 对 (query, doc) pair 逐对打分
    - 后备方案: n-gram 重叠度评分

    CoT 安全退避:
    - 当最高重排得分低于极低阈值时，判定知识库无相关知识
    - 触发安全退避机制，拒绝回答
    """

    def __init__(self, low_threshold: float | None = None):
        """
        参数:
            low_threshold: CoT 退避阈值，None 时使用配置值
        """
        self.low_threshold = low_threshold if low_threshold is not None else cfg.reranker.low_threshold

    async def rerank(
        self,
        query: str,
        documents: list[DocumentChunk],
        top_k: int = 0,
    ) -> list[DocumentChunk]:
        """重排文档列表

        优先使用 BGE-Reranker Cross-Encoder，不可用时退化为 n-gram。
        """
        if not documents:
            return []
        if top_k <= 0:
            top_k = cfg.retrieval.rerank_top_k

        try:
            return await self._cross_encoder_rerank(query, documents, top_k)
        except Exception as e:
            logger.warning(f"Cross-Encoder 重排失败，退化为 n-gram 评分: {e}")

        return self._ngram_rerank(query, documents, top_k)

    async def _cross_encoder_rerank(
        self, query: str, documents: list[DocumentChunk], top_k: int
    ) -> list[DocumentChunk]:
        """使用 BGE-Reranker Cross-Encoder 对 (query, doc) 逐对打分"""
        model = _get_cross_encoder()

        pairs = [(query, doc.content) for doc in documents]
        scores = model.predict(pairs, show_progress_bar=False)

        scored_docs = []
        for doc, score in zip(documents, scores):
            float_score = float(score)
            float_score = max(0.0, min(1.0, float_score))
            scored_docs.append((float_score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        result = []
        for score, doc in scored_docs[:top_k]:
            result.append(DocumentChunk(
                doc_id=doc.doc_id,
                content=doc.content,
                source=doc.source,
                score=score,
                metadata=doc.metadata,
            ))

        logger.info(f"Cross-Encoder Rerank 完成, top-1 score={result[0].score:.3f}" if result else "Rerank 无结果")
        return result

    def _ngram_rerank(
        self, query: str, documents: list[DocumentChunk], top_k: int
    ) -> list[DocumentChunk]:
        """n-gram 重叠度评分重排（后备方案）

        评分公式: final_score = ngram_overlap * weight + original_score * weight
        ngram_overlap = 交集 n-gram 数 / 查询 n-gram 总数
        """
        query_terms = extract_chinese_terms(query)
        scored_docs = []

        for doc in documents:
            content_terms = extract_chinese_terms(doc.content)
            overlap = len(query_terms & content_terms)
            ngram_score = overlap / max(len(query_terms), 1)
            orig_score = max(0.0, min(1.0, doc.score))
            final_score = ngram_score * cfg.reranker.ngram_weight + orig_score * cfg.reranker.ngram_original_weight
            scored_docs.append((final_score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        result = []
        for score, doc in scored_docs[:top_k]:
            result.append(DocumentChunk(
                doc_id=doc.doc_id,
                content=doc.content,
                source=doc.source,
                score=score,
                metadata=doc.metadata,
            ))

        logger.info(f"n-gram Rerank 完成, top-1 score={result[0].score:.3f}" if result else "Rerank 无结果")
        return result

    def is_insufficient(self, documents: list[DocumentChunk]) -> bool:
        """判断检索结果是否不足以支撑可靠回答（CoT 退避条件）

        设计文档 3.4 节: 当 Reranker 最高分低于极低阈值（如 0.2），
        表明知识库无相关知识，应触发 CoT 安全退避。
        """
        if not documents:
            return True
        max_score = max(doc.score for doc in documents)
        if max_score < self.low_threshold:
            logger.warning(f"检索得分极低 ({max_score:.3f} < {self.low_threshold})，触发 CoT 退避")
            return True
        return False