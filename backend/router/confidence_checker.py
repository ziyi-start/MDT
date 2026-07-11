"""置信度评估器 - 文档一致性 + 生成自验证

设计文档 3.2 节: 置信度评估机制
- 文档一致性校验: Top-1 与 Top-2 分数差 < 阈值且结论相悖 → 判定"检索冲突"
- 生成自验证: 要求 LLM 输出引用来源 [Source: Doc X]，核心结论无引用 → 判定"低置信度"
- 携因打回: 触发低置信/冲突时，将失败原因作为新 Context 注入
"""
from __future__ import annotations

import json
import logging

from schema.models import DocumentChunk
from llm.client import AsyncLLMClient
from llm.prompt_templates import CONFIDENCE_CHECK_PROMPT
from schema.messages import Message
from config import cfg

logger = logging.getLogger(__name__)


class RouteEscalationException(Exception):
    """路由升级异常 - 携因打回至 MDT

    携带失败原因，由顶层编排器捕获并强制转入 MDT 模式。
    设计文档: "将失败原因作为新 Context 注入 Recruiter，强制路由至 MDT 模式"
    """
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


class ConfidenceChecker:
    """置信度评估器

    双管齐下判断生成质量:
    1. 文档级校验: 检索结果是否存在冲突
    2. 生成级自验证: 回答是否基于检索文献
    """

    def __init__(self, llm: AsyncLLMClient, score_gap_threshold: float | None = None):
        """
        参数:
            llm: LLM 客户端，用于生成自验证
            score_gap_threshold: 文档一致性校验的分数差距阈值，None 时使用配置值
        """
        self.llm = llm
        self.score_gap_threshold = score_gap_threshold if score_gap_threshold is not None else cfg.confidence.score_gap_threshold

    def check_document_consistency(self, documents: list[DocumentChunk]) -> tuple[bool, str]:
        """文档一致性校验

        设计文档: "若 Reranker Top-1 与 Top-2 分数差 < 阈值且结论相悖，判定为检索冲突"

        策略: 分数差距小 + 两篇文档对同一药物/治疗的结论相悖（一篇推荐一篇禁忌）
        """
        if len(documents) < 2:
            return True, ""

        top1, top2 = documents[0], documents[1]
        score_gap = abs(top1.score - top2.score)

        if score_gap < self.score_gap_threshold:
            # 使用 LLM 判断两篇文档是否存在结论冲突（比简单否定词匹配更准确）
            conflict = self._detect_conclusion_conflict(top1.content, top2.content)
            if conflict:
                reason = f"检索指南冲突: 文档得分差距仅 {score_gap:.3f} 但结论相悖"
                logger.warning(reason)
                return False, reason

        return True, ""

    def _detect_conclusion_conflict(self, content1: str, content2: str) -> bool:
        """检测两篇文档是否存在结论冲突

        启发式检测: 一篇推荐某药物/方案，另一篇明确禁忌/不推荐同一药物/方案
        """
        # 禁忌/不推荐关键词
        negation_phrases = ["禁用", "禁忌", "不推荐", "避免使用", "不可使用", "不建议", "不应使用"]
        # 推荐/可用关键词
        positive_phrases = ["推荐", "首选", "可用", "安全", "可以使用", "一线用药"]

        has_negation_1 = any(p in content1 for p in negation_phrases)
        has_negation_2 = any(p in content2 for p in negation_phrases)
        has_positive_1 = any(p in content1 for p in positive_phrases)
        has_positive_2 = any(p in content2 for p in positive_phrases)

        # 一篇推荐，一篇禁忌 → 冲突
        if (has_positive_1 and has_negation_2) or (has_positive_2 and has_negation_1):
            return True

        return False

    async def check_generation_confidence(self, answer: str, documents: list[DocumentChunk]) -> tuple[bool, str]:
        """生成自验证

        设计文档: "要求 LLM 生成答案时输出引用来源 [Source: Doc 1]。
        若核心结论无对应文档引用，判定为低置信度。"

        策略:
        1. 检查回答是否包含引用标记
        2. 使用 LLM 验证核心结论是否有文献支撑
        """
        has_citation = "[Source:" in answer or "[Source " in answer

        # 即使有引用标记，仍需 LLM 验证核心结论是否有强支撑
        docs_text = "\n".join(f"[Doc {i+1}] {d.content[:cfg.reranker.content_preview_length]}" for i, d in enumerate(documents))
        resp = await self.llm.chat(
            messages=[
                Message(role="user", content=CONFIDENCE_CHECK_PROMPT.format(
                    answer=answer, documents=docs_text,
                )),
            ],
            response_format={"type": "json_object"},
            temperature=cfg.llm.temperatures["confidence_check"],
        )
        try:
            data = json.loads(resp.content or "{}")
            confidence = data.get("confidence", 0.5)
            conflict = data.get("conflict_detected", False)

            if confidence < cfg.confidence.min_confidence or conflict:
                reason = data.get("reason", "置信度评估未通过")
                return False, reason
        except json.JSONDecodeError:
            # LLM 验证失败时，降级为简单引用检查
            if not has_citation:
                return False, "回答缺乏文献引用支撑"

        return True, ""

    async def evaluate(self, answer: str, documents: list[DocumentChunk]) -> None:
        """综合评估，不通过则抛出 RouteEscalationException

        调用链: SimpleRAG 末尾 → evaluate → 异常 → 顶层编排器捕获 → 升级到 MDT
        """
        # 1. 文档一致性校验
        consistent, reason = self.check_document_consistency(documents)
        if not consistent:
            raise RouteEscalationException(reason)

        # 2. 生成自验证
        confident, reason = await self.check_generation_confidence(answer, documents)
        if not confident:
            raise RouteEscalationException(reason)