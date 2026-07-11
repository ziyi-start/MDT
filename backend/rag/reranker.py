"""医疗 Reranker - 重排序与 CoT 安全退避判断

功能:
1. 对混合检索结果进行重排序，打压临床逻辑无关的噪声
2. 判断检索结果是否足以支撑可靠回答（CoT 退避条件）

MVP: 使用 LLM 辅助重排 + n-gram 重叠度评分
生产: 使用微调的医疗 BGE-Reranker (Cross-Encoder) API
"""
from __future__ import annotations

import json
import logging

from schema.models import DocumentChunk
from llm.client import AsyncLLMClient
from rag.embedding import extract_chinese_terms
from config import cfg

logger = logging.getLogger(__name__)


class MedicalReranker:
    """医疗重排序器

    重排策略:
    - 如果配置了 LLM 客户端，使用 LLM 对文档相关性打分
    - 否则使用 n-gram 重叠度评分作为后备

    CoT 安全退避:
    - 当最高重排得分低于极低阈值时，判定知识库无相关知识
    - 触发安全退避机制，拒绝回答
    """

    def __init__(self, llm_client: AsyncLLMClient | None = None, low_threshold: float | None = None):
        """
        参数:
            llm_client: LLM 客户端（可选），用于 LLM 辅助重排
            low_threshold: CoT 退避阈值，None 时使用配置值
        """
        self.llm = llm_client
        self.low_threshold = low_threshold if low_threshold is not None else cfg.reranker.low_threshold

    async def rerank(
        self,
        query: str,
        documents: list[DocumentChunk],
        top_k: int = 0,
    ) -> list[DocumentChunk]:
        """重排文档列表

        优先使用 LLM 辅助重排，不可用时退化为 n-gram 重叠度评分。
        """
        if not documents:
            return []
        if top_k <= 0:
            top_k = cfg.retrieval.rerank_top_k

        # 尝试使用 LLM 辅助重排
        if self.llm:
            try:
                return await self._llm_rerank(query, documents, top_k)
            except Exception as e:
                logger.warning(f"LLM 重排失败，退化为 n-gram 评分: {e}")

        # 后备: n-gram 重叠度评分
        return self._ngram_rerank(query, documents, top_k)

    async def _llm_rerank(
        self, query: str, documents: list[DocumentChunk], top_k: int
    ) -> list[DocumentChunk]:
        """使用 LLM 对文档与查询的相关性打分"""
        from schema.messages import Message

        preview_len = cfg.reranker.content_preview_length
        docs_text = "\n".join(
            f"[Doc {i+1}] {doc.content[:preview_len]}" for i, doc in enumerate(documents)
        )
        prompt = f"""请对以下文档与查询的相关性打分（0.0-1.0），输出 JSON 对象：
查询：{query}

文档：
{docs_text}

输出格式：{{"scores": [{{"doc_index": 1, "score": 0.9}}, ...]}}"""

        resp = await self.llm.chat(
            messages=[Message(role="user", content=prompt)],
            response_format={"type": "json_object"},
            temperature=cfg.llm.temperatures["reranker"],
        )

        scored_docs = []
        try:
            data = json.loads(resp.content or "{}")
            score_map = {}
            # 兼容多种 LLM 输出格式: {"scores": [...]} 或 {"doc_index": ...} 或直接 [...]
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                # 尝试多种 key
                for key in ("scores", "results", "rankings"):
                    if key in data and isinstance(data[key], list):
                        items = data[key]
                        break
                if not items:
                    # 可能是 {"1": 0.9, "2": 0.7} 格式
                    for k, v in data.items():
                        if isinstance(v, (int, float)):
                            try:
                                idx = int(k) - 1
                                score_map[idx] = float(v)
                            except ValueError:
                                pass

            for item in items:
                if isinstance(item, dict):
                    raw_idx = item.get("doc_index", item.get("index"))
                    if raw_idx is None or not isinstance(raw_idx, (int, float)):
                        continue
                    idx = int(raw_idx) - 1
                    if 0 <= idx < len(documents):
                        score_map[idx] = item.get("score", item.get("relevance", 0.5))

            for i, doc in enumerate(documents):
                llm_score = score_map.get(i, 0.5)
                # 综合得分: LLM 评分 * weight + 原始检索分数 * weight
                final_score = llm_score * cfg.reranker.llm_weight + doc.score * cfg.reranker.original_weight
                scored_docs.append((final_score, doc))
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM 重排结果解析失败: {e}")
            return self._ngram_rerank(query, documents, top_k)

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

        logger.info(f"LLM Rerank 完成, top-1 score={result[0].score:.3f}" if result else "Rerank 无结果")
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
            # 综合得分: n-gram 重叠 * weight + 原始检索分数 * weight
            final_score = ngram_score * cfg.reranker.ngram_weight + doc.score * cfg.reranker.ngram_original_weight
            scored_docs.append((final_score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        result = []
        for score, doc in scored_docs[:top_k]:
            # 创建新对象避免修改原始文档
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