"""上下文组装器 — 参考 Anthropic System Prompt Design + OpenAI Structured Prompts

设计理念:
  "上下文组装不是简单的字符串拼接，而是结构化的信息编排"
  "好的上下文结构 = 角色指令 + 约束条件 + 记忆注入 + 工具接口 + 查询上下文"

参考架构:
  - Anthropic System Prompt Guidelines: 结构化角色/约束/工具/示例
  - OpenAI Structured Output: response_format + tool_choice
  - DSPy Prompt Optimization: 模块化 prompt 结构
  - LangChain PromptTemplate: 变量插值 + 部分格式化

组装策略（按 LLM 注意力分布）:
  ┌─────────────────────────────────────────────┐
  │ 高注意力区 (开头)                             │
  │   1. 角色定义 (System Role)                  │
  │   2. 核心约束 (Safety Rules)                  │
  │   3. 任务指令 (Task Instructions)             │
  ├─────────────────────────────────────────────┤
  │ 中注意力区 (中间)                             │
  │   4. 长期记忆注入 (Skills + Reflections)     │
  │   5. 上下文穿透项 (Persistent Constraints)   │
  │   6. 对话历史 (Conversation History)          │
  ├─────────────────────────────────────────────┤
  │ 低注意力区 (末尾)                             │
  │   7. 参考文献 (Retrieved Documents)          │
  │   8. 工具结果 (Tool Results)                 │
  └─────────────────────────────────────────────┘
  │ 最高注意力区 (最后)                           │
  │   9. 用户查询 (User Query)                    │
  └─────────────────────────────────────────────┘

原则:
  - 最重要信息放开头（角色+约束）和结尾（用户查询）
  - 参考文献放中间靠后（需要时查阅，不需要时快速扫过）
  - 工具结果紧跟其引用消息
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from context.memory_hierarchy import MemoryEntry, MemoryTier, MessageRole, TokenEstimator

logger = logging.getLogger(__name__)


@dataclass
class AssembleConfig:
    """组装配置"""
    max_system_tokens: int = 2000
    max_memory_tokens: int = 1500
    max_history_tokens: int = 3000
    max_context_tokens: int = 5000
    max_total_tokens: int = 14000

    include_skills: bool = True
    include_reflections: bool = True
    include_profile: bool = True
    include_tool_results: bool = True


@dataclass
class AssembledContext:
    """组装完成的上下文"""
    system_prompt: str
    user_prompt: str
    messages: list[dict]
    total_tokens: int
    budget_usage: dict


class ContextAssembler:
    """上下文组装器 — 将分散的记忆和信息源组装为结构化的 LLM 输入

    支持多种组装策略:
    - simple_rag: 简单 RAG 查询组装
    - mdt_expert: 多专家 ReAct 查询组装
    - consensus: 共识提炼组装
    - verification: 证据验证组装
    - decision: 决策评估组装

    用法:
        assembler = ContextAssembler()
        ctx = assembler.assemble_for_rag(
            role="你是一个医疗助手",
            constraints=["只基于文献回答"],
            query="高血压怎么治?",
            documents=retrieved_docs,
            skills=["技能1"],
            reflections=["⚠️注意: 避免..."]
        )
    """

    def __init__(self, config: Optional[AssembleConfig] = None):
        self.config = config or AssembleConfig()

    def assemble_for_rag(
        self,
        role: str,
        query: str,
        documents: Optional[list] = None,
        constraints: Optional[list[str]] = None,
        skills: Optional[list[str]] = None,
        reflections: Optional[list[str]] = None,
        profile: Optional[str] = None,
        history: Optional[str] = None,
    ) -> AssembledContext:
        """组装 Simple RAG 上下文"""
        system_parts = []

        system_parts.append(role)

        if constraints:
            system_parts.append("约束规则:\n" + "\n".join(f"- {c}" for c in constraints))

        if self.config.include_reflections and reflections:
            system_parts.append("--- 历史教训 ---")
            system_parts.extend(reflections)

        if self.config.include_skills and skills:
            system_parts.append("--- 参考经验 ---")
            system_parts.extend(skills)

        system_text = "\n\n".join(system_parts)
        system_text = TokenEstimator.truncate(system_text, self.config.max_system_tokens)

        user_parts = []
        if history:
            user_parts.append(f"[对话历史]\n{history}")

        if documents:
            context_text = self._format_documents(documents)
            context_text = TokenEstimator.truncate(context_text, self.config.max_context_tokens)
            user_parts.append(f"参考文献:\n{context_text}")

        if profile:
            user_parts.append(f"患者信息: {profile}")

        user_parts.append(f"问题: {query}")

        user_text = "\n\n".join(user_parts)

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]
        total_tokens = TokenEstimator.count_tokens_in_messages(messages)

        return AssembledContext(
            system_prompt=system_text,
            user_prompt=user_text,
            messages=messages,
            total_tokens=total_tokens,
            budget_usage={
                "system_tokens": TokenEstimator.estimate(system_text),
                "context_tokens": TokenEstimator.estimate(user_text),
                "total": total_tokens,
            },
        )

    def assemble_for_expert(
        self,
        department: str,
        query: str,
        profile: Optional[str] = None,
        reflection_hint: str = "",
        skill_hints: str = "",
        escalation_reason: str = "",
    ) -> AssembledContext:
        """组装 MDT 专家 ReAct 上下文"""
        system_parts = [
            f"你是{department}的资深专家医生。你需要基于检索到的文献证据，回答患者的医疗问题。",
            "",
            "重要规则:",
            "1. 你必须根据患者的禁忌和病史，主动构思专业的检索词去调用文献检索工具",
            "2. 所有结论必须引用文献来源，格式为 [Source: 文档编号]",
            "3. 如果检索结果不足以支撑结论，请明确说明",
            "4. 禁止编造不存在的药物或治疗方案",
        ]

        if reflection_hint:
            system_parts.append(f"\n{reflection_hint}")

        if skill_hints:
            system_parts.append(f"\n{skill_hints}")

        if profile:
            system_parts.append(f"\n患者画像: {profile}")

        if escalation_reason:
            system_parts.append(f"\n⚠️注意: 此问题此前经简单检索未能给出可靠回答，原因: {escalation_reason}。请特别注意此问题。")

        system_text = "\n".join(system_parts)

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": query},
        ]

        return AssembledContext(
            system_prompt=system_text,
            user_prompt=query,
            messages=messages,
            total_tokens=TokenEstimator.count_tokens_in_messages(messages),
            budget_usage={"system_tokens": TokenEstimator.estimate(system_text)},
        )

    def assemble_for_consensus(
        self,
        expert_opinions: list[str],
        profile: Optional[str] = None,
    ) -> AssembledContext:
        """组装共识提炼上下文"""
        opinions_text = "\n\n".join(expert_opinions)
        profile_text = profile or "无"

        system_text = (
            "你是一位资深主任医师，负责综合多位专家的会诊意见。\n"
            "请根据以下专家意见，提炼出一份统一的会诊报告:\n"
            "1. 识别专家间的共识与分歧\n"
            "2. 对分歧给出权衡建议\n"
            "3. 确保最终建议考虑了患者的所有禁忌\n"
            "4. 输出结构化报告"
        )

        user_text = f"专家意见:\n{opinions_text}\n\n患者画像: {profile_text}"

        messages = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ]

        return AssembledContext(
            system_prompt=system_text,
            user_prompt=user_text,
            messages=messages,
            total_tokens=TokenEstimator.count_tokens_in_messages(messages),
            budget_usage={},
        )

    def _format_documents(self, documents: list) -> str:
        """格式化文献列表"""
        parts = []
        for i, doc in enumerate(documents):
            content = getattr(doc, "content", "") if hasattr(doc, "content") else str(doc)
            source = getattr(doc, "source", "") if hasattr(doc, "source") else ""
            score = getattr(doc, "score", 0.0) if hasattr(doc, "score") else 0.0
            label = f"[Source: Doc {i + 1}]"
            if source:
                label += f" ({source})"
            if score:
                label += f" [score: {score:.2f}]"
            parts.append(f"{label}\n{content}")
        return "\n\n".join(parts)

    def optimize_budget(self, assembled: AssembledContext, max_tokens: int) -> AssembledContext:
        """预算优化 — 按优先级裁剪"""
        if assembled.total_tokens <= max_tokens:
            return assembled
        ratio = max_tokens / max(assembled.total_tokens, 1)
        new_system = TokenEstimator.truncate(assembled.system_prompt, int(len(assembled.system_prompt) * ratio))
        new_user = TokenEstimator.truncate(assembled.user_prompt, int(len(assembled.user_prompt) * ratio))
        new_messages = [
            {"role": "system", "content": new_system},
            {"role": "user", "content": new_user},
        ]
        return AssembledContext(
            system_prompt=new_system,
            user_prompt=new_user,
            messages=new_messages,
            total_tokens=TokenEstimator.estimate(new_system) + TokenEstimator.estimate(new_user),
            budget_usage=assembled.budget_usage,
        )
