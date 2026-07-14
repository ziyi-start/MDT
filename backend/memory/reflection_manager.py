"""归因式反思三元组管理

设计文档 3.4 节: 归因式反思沉淀
- 打回时强制 LLM 输出结构化三元组: <意图, 归因, 避坑动作>
- 向量化后存入 Milvus Reflection_Mem 集合
- 下次遇到相似 Query 时检索反思记忆，高相似度命中则强插 Hint 预警

Agent 自进化增强:
- 每次检索命中时递增 usage_count（"从教训中学习"的效果可衡量）
- provenance 追踪每个反思的来源

CoT 安全退避机制:
- Reranker 最高分低于极低阈值（0.2）时，触发退避
- 此异常应在 MDT 流程中被检查并抛出

与 SkillManager 的职责分离:
- ReflectionManager: 从失败中提取"不要做什么"（避坑）
- SkillManager:    从成功中提取"应该怎么做"（最佳实践）
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Optional

from schema.models import ReflectionTriple
from llm.client import AsyncLLMClient
from llm.prompt_templates import REFLECTION_PROMPT
from rag.milvus_client import MilvusManager
from rag.embedding import dummy_embed, extract_chinese_terms
from schema.messages import Message
from config import cfg

logger = logging.getLogger(__name__)

# 内存反思存储（Milvus 不可用时的备用）
_IN_MEMORY_REFLECTIONS: list[dict] = []


class InsufficientInformationException(Exception):
    """信息不足异常 - 触发 CoT 安全退避

    设计文档 3.4 节: 当 Reranker 得分全部低于极低阈值，
    表明知识库无相关知识，应强制中断推理，走保守策略。
    """
    pass


class ReflectionManager:
    """归因式反思沉淀与检索

    核心机制:
    1. 生成: 打回时强制 LLM 输出 ReflectionTriple 三元组
    2. 存储: 向量化后存入 Milvus Reflection_Mem（或内存备用）
    3. 检索: 遇到相似 Query 时检索反思记忆，返回避坑 Hint
    """

    def __init__(self, llm: AsyncLLMClient, milvus: MilvusManager | None):
        self.llm = llm
        self.milvus = milvus
        self._max_in_memory = cfg.reflection.max_in_memory

    async def generate_and_store(self, query: str, reason: str) -> ReflectionTriple:
        """生成归因反思三元组并存储

        流程: LLM 归因 Prompt → 强制结构化输出 → 向量化 → 存入 Milvus/内存
        """
        # 归因 Prompt 强制输出结构化三元组 JSON
        resp = await self.llm.chat(
            messages=[
                Message(role="user", content=REFLECTION_PROMPT.format(query=query, reason=reason)),
            ],
            response_format={"type": "json_object"},
            temperature=cfg.llm.temperatures["reflection"],
        )

        try:
            data = json.loads(resp.content or "{}")
            triple = ReflectionTriple(
                intent=data.get("intent", ""),
                cause=data.get("cause", ""),
                avoid_action=data.get("avoid_action", ""),
            )
            # 校验: 反思三元组各字段不应为空
            if not triple.intent:
                triple.intent = query
            if not triple.cause:
                triple.cause = reason
            if not triple.avoid_action:
                triple.avoid_action = "需要更谨慎地评估此场景"
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"反思三元组解析失败: {e}")
            triple = ReflectionTriple(
                intent=query,
                cause=reason,
                avoid_action="需要更谨慎地评估此场景",
            )

        # 向量化存储
        if self.milvus:
            try:
                text_repr = f"{triple.intent} {triple.cause} {triple.avoid_action}"
                vec = await dummy_embed(text_repr)
                triple_id = str(uuid.uuid4())
                self.milvus.upsert(
                    collection_name=cfg.milvus.collections.reflection,
                    data=[{
                        "triple_id": triple_id,
                        "embedding": vec,
                        "intent": triple.intent,
                        "cause": triple.cause,
                        "avoid_action": triple.avoid_action,
                    }],
                )
            except Exception as e:
                logger.warning(f"反思 Milvus 写入失败: {e}")
                self._store_in_memory(triple)
        else:
            self._store_in_memory(triple)

        logger.info(f"反思存储: intent={triple.intent}, action={triple.avoid_action}")
        return triple

    def _store_in_memory(self, triple: ReflectionTriple):
        """存入内存备用存储（带容量上限）"""
        global _IN_MEMORY_REFLECTIONS
        if len(_IN_MEMORY_REFLECTIONS) >= self._max_in_memory:
            _IN_MEMORY_REFLECTIONS = _IN_MEMORY_REFLECTIONS[-self._max_in_memory // 2:]
        _IN_MEMORY_REFLECTIONS.append(triple.model_dump())

    def _increment_usage(self, triple_id: str):
        """反思被命中时，递增使用计数"""
        if not self.milvus or not triple_id:
            return
        try:
            results = self.milvus.client.query(
                collection_name=cfg.milvus.collections.reflection,
                filter=f'triple_id == "{triple_id}"',
                output_fields=["usage_count"],
            )
            if results:
                current = results[0].get("usage_count", 0)
                self.milvus.client.upsert(
                    collection_name=cfg.milvus.collections.reflection,
                    data=[{"triple_id": triple_id, "usage_count": current + 1}],
                )
        except Exception:
            pass

    async def stats(self) -> dict:
        """反思管理器统计"""
        return {
            "in_memory_count": len(_IN_MEMORY_REFLECTIONS),
            "milvus_available": self.milvus is not None,
        }

    async def search_reflection(self, query: str, threshold: float | None = None) -> Optional[str]:
        """检索反思记忆，返回避坑动作 Hint

        高相似度命中时返回格式: "⚠️历史教训：{避坑动作}"
        未命中返回 None
        """
        if threshold is None:
            threshold = cfg.reflection.search_threshold

        if self.milvus:
            try:
                vec = await dummy_embed(query)
                results = self.milvus.search(
                    collection_name=cfg.milvus.collections.reflection,
                    vector=vec,
                    limit=1,
                    output_fields=["intent", "cause", "avoid_action"],
                )
                if results and results[0].get("score", 0) >= threshold:
                    hit = results[0]
                    self._increment_usage(hit.get("triple_id", ""))
                    hint = f"⚠️历史教训：{hit.get('avoid_action', '')}"
                    logger.info(f"反思命中 (Milvus): {hint}")
                    return hint
            except Exception as e:
                logger.warning(f"反思 Milvus 检索失败: {e}")

        # 内存模式: 基于关键词的模糊匹配
        if _IN_MEMORY_REFLECTIONS:
            query_terms = extract_chinese_terms(query)
            best_match = None
            best_overlap = 0
            for ref in _IN_MEMORY_REFLECTIONS:
                intent_terms = extract_chinese_terms(ref.get("intent", ""))
                overlap = len(query_terms & intent_terms)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = ref

            # 关键词重叠度 >= 阈值 视为命中
            min_overlap = max(
                int(len(query_terms) * cfg.reflection.in_memory_overlap_ratio),
                cfg.reflection.in_memory_min_terms,
            )
            if best_match and best_overlap >= min_overlap:
                hint = f"⚠️历史教训：{best_match.get('avoid_action', '')}"
                logger.info(f"反思命中 (内存): {hint}")
                return hint

        return None