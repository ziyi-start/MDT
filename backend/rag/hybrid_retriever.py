"""混合检索 - BM25 + Dense + RRF 融合 + 画像约束

核心机制（严格还原文档 3.3 节）:
1. 硬约束: 根据患者画像中的禁忌（疾病+过敏），动态拼接 Milvus boolean filter
2. 软约束: 在检索 Query 中拼接禁忌信息进行查询改写
3. 混合检索: 同时执行 BM25（精准匹配药名）和 Dense（语义匹配）
4. RRF 融合: 手写 Reciprocal Rank Fusion 算法合并两路检索结果
5. 重排: 调用 Medical Reranker 去噪
"""
from __future__ import annotations

import logging
from typing import Optional

from schema.models import PatientProfile, DocumentChunk
from rag.milvus_client import MilvusManager
from rag.embedding import dummy_embed, extract_chinese_terms
from config import cfg

logger = logging.getLogger(__name__)

# 内存知识库（Milvus 不可用时的备用）
_IN_MEMORY_KB: list[dict] = []


def load_in_memory_kb(docs: list[dict]):
    """加载内存知识库数据"""
    global _IN_MEMORY_KB
    _IN_MEMORY_KB = docs


class HybridRetriever:
    """BM25 + Dense 混合检索 + 手写 RRF 融合 + 画像约束"""

    def __init__(self, milvus: MilvusManager | None):
        self.milvus = milvus

    async def retrieve(
        self,
        query: str,
        profile: Optional[PatientProfile] = None,
        top_k: int = 0,
        departments: list[str] | None = None,
    ) -> list[DocumentChunk]:
        """执行混合检索（完整流程）

        流程: 软约束改写 → 硬约束过滤 → Dense检索 + BM25检索 → RRF融合 → 硬约束后过滤
        """
        if top_k <= 0:
            top_k = cfg.retrieval.top_k
        over_limit = top_k * cfg.retrieval.over_retrieval_multiplier

        # ---- 1. 画像约束: 软约束 - 查询改写 ----
        # 将疾病和过敏都作为禁忌信息拼接，扩大检索覆盖
        rewritten_query = query
        contraindications = profile.get_contraindications() if profile else []
        if contraindications:
            rewritten_query = f"{query} 禁忌: {' '.join(contraindications)}"
            logger.info(f"查询改写 (软约束): {rewritten_query}")

        # ---- 2. 检索 ----
        if self.milvus:
            results = await self._hybrid_search_milvus(rewritten_query, profile, over_limit)
        elif _IN_MEMORY_KB:
            results = self._in_memory_search(rewritten_query, profile, over_limit)
        else:
            return []

        # ---- 3. 画像约束: 硬约束后过滤 ----
        # 过滤掉文档禁忌与患者禁忌冲突的结果
        filtered = self._apply_profile_filter(results, profile)

        return filtered[:top_k]

    @staticmethod
    def _sanitize_filter_value(value: str) -> str:
        """转义 Milvus filter 表达式中的特殊字符，防止注入"""
        return value.replace('"', '').replace("'", "").replace("%", "").replace("\\", "")

    async def _hybrid_search_milvus(
        self, query: str, profile: PatientProfile | None, limit: int
    ) -> list[DocumentChunk]:
        """Milvus 混合检索: Dense + BM25 (hybrid_search) + RRF 融合"""
        # 硬约束: 构建 Milvus 过滤表达式
        filter_expr = ""
        contraindications = profile.get_contraindications() if profile else []
        if contraindications:
            # Milvus 语法: not (field like "%xxx%") 用于 VARCHAR 排除
            conditions = [
                f'not (contraindications like "%{self._sanitize_filter_value(c)}%")'
                for c in contraindications
            ]
            filter_expr = " and ".join(conditions)
            logger.info(f"Milvus 硬约束 filter: {filter_expr}")

        # Dense + BM25 混合检索（Milvus 原生 hybrid_search + RRF）
        query_vec = await dummy_embed(query)
        results = []
        try:
            results = self.milvus.hybrid_search(
                collection_name=cfg.milvus.collections.kb,
                dense_vector=query_vec,
                query_text=query,
                limit=limit,
                filter_expr=filter_expr,
            )
        except Exception as e:
            logger.warning(f"Milvus hybrid_search 失败，降级为 Dense 检索: {e}")
            try:
                dense_results = self.milvus.search(
                    collection_name=cfg.milvus.collections.kb,
                    vector=query_vec,
                    limit=limit,
                    filter_expr=filter_expr,
                )
                results = dense_results
            except Exception as e2:
                logger.warning(f"Milvus Dense 检索也失败: {e2}")

        return self._milvus_results_to_chunks(results)

    def _keyword_rerank_as_bm25(
        self, query: str, dense_results: list[dict], limit: int
    ) -> list[dict]:
        """用关键词匹配模拟 BM25 排名（BM25 不可用时的退化方案）

        对 Dense 检索结果做关键词重叠度评分，作为 BM25 一路的排名输入。
        """
        from rag.embedding import extract_chinese_terms
        query_terms = extract_chinese_terms(query)
        if not query_terms:
            return []

        scored = []
        for doc in dense_results:
            content = doc.get("content", "")
            content_terms = extract_chinese_terms(content)
            overlap = len(query_terms & content_terms)
            score = overlap / max(len(query_terms), 1)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:limit]]

    def _rrf_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        top_k: int,
        k: int = 0,
    ) -> list[DocumentChunk]:
        """Reciprocal Rank Fusion 融合算法

        公式: RRF_score(d) = Σ 1/(k + rank_i)
        其中 k=60 是标准参数，rank_i 是文档在第 i 路检索中的排名
        """
        if k <= 0:
            k = cfg.retrieval.rrf_k
        scores: dict[str, float] = {}
        doc_map: dict[str, dict] = {}

        # Dense 排名贡献
        for rank, doc in enumerate(dense_results):
            doc_id = doc.get("id", f"dense_{rank}")
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc

        # BM25 排名贡献
        for rank, doc in enumerate(bm25_results):
            doc_id = doc.get("id", f"bm25_{rank}")
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank + 1)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        # 按 RRF 分数降序排列
        sorted_ids = sorted(scores, key=scores.get, reverse=True)[:top_k]

        return [
            DocumentChunk(
                doc_id=doc_id,
                content=doc_map[doc_id].get("content", ""),
                source=doc_map[doc_id].get("source", ""),
                score=scores[doc_id],
                metadata={
                    "department": doc_map[doc_id].get("department", ""),
                    "contraindications": doc_map[doc_id].get("contraindications", ""),
                },
            )
            for doc_id in sorted_ids
        ]

    @staticmethod
    def _milvus_results_to_chunks(results: list[dict]) -> list[DocumentChunk]:
        """将 Milvus search 结果转为 DocumentChunk 列表"""
        return [
            DocumentChunk(
                doc_id=doc.get("id", ""),
                content=doc.get("content", ""),
                source=doc.get("source", ""),
                score=doc.get("score", 0),
                metadata={
                    "department": doc.get("department", ""),
                    "contraindications": doc.get("contraindications", ""),
                },
            )
            for doc in results
        ]

    def _in_memory_search(
        self, query: str, profile: PatientProfile | None, limit: int
    ) -> list[DocumentChunk]:
        """内存关键词匹配检索（Milvus 不可用时的备用方案）

        使用中文 n-gram 提取 + 关键词重叠度评分
        """
        query_terms = extract_chinese_terms(query)
        if not query_terms:
            return []
        scored = []
        for doc in _IN_MEMORY_KB:
            content_terms = extract_chinese_terms(doc.get("content", ""))
            overlap = len(query_terms & content_terms)
            if overlap > 0:
                scored.append((overlap / max(len(query_terms), 1), doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            DocumentChunk(
                doc_id=doc.get("doc_id", ""),
                content=doc.get("content", ""),
                source=doc.get("source", ""),
                score=score,
                metadata={
                    "department": doc.get("department", ""),
                    "contraindications": doc.get("contraindications", ""),
                },
            )
            for score, doc in scored[:limit]
        ]

    def _apply_profile_filter(
        self, docs: list[DocumentChunk], profile: PatientProfile | None
    ) -> list[DocumentChunk]:
        """硬约束后过滤: 排除文档禁忌与患者禁忌冲突的文档

        注意: 如果过滤后结果为空，返回原始结果（避免空结果导致无法回答）
        """
        if not profile:
            return docs
        contraindications = profile.get_contraindications()
        if not contraindications:
            return docs

        filtered = []
        for doc in docs:
            doc_contraindications = doc.metadata.get("contraindications", "")
            doc_items = {c.strip() for c in doc_contraindications.split(",") if c.strip()}
            has_conflict = bool(doc_items & set(contraindications))
            if not has_conflict:
                filtered.append(doc)
            else:
                logger.info(f"硬约束过滤: 排除文档 {doc.doc_id} (含禁忌)")

        # 全部被过滤时返回空列表，而非回退到未过滤结果
        # 医疗场景中，所有文档都与患者禁忌冲突时应触发退避，而非使用禁忌文档
        if not filtered:
            logger.warning(f"硬约束过滤: 全部 {len(docs)} 篇文档均含禁忌，返回空结果触发退避")
        return filtered