"""统一上下文管理器 — 聚合四层记忆 + 多轮对话 + 上下文窗口 + 组装器

设计理念:
  "ContextManager 是 Agent 的记忆操作系统内核"
  "它协调四层记忆的存取、对话状态的维护、上下文窗口的调度、以及最终的上下文组装"

参考架构:
  - MemGPT OS: 内存管理单元 (MMU) 协调 Core/Archival/Recall 三层
  - Harness ContextBudget: 分层预算 + 渲染 + 摘要
  - LangChain BaseMemory: 统一的 load_memory_variables / save_context 接口
  - LlamaIndex ChatMemoryBuffer: 从 token_limit 管理消息窗口

统一接口:
  上下文生命周期:
    1. begin_session()     — 开始会话，初始化四层记忆
    2. add_conversation()  — 记录一轮对话
    3. prepare_context()   — 组装 LLM 上下文
    4. end_session()       — 结束会话，持久化长期记忆

  记忆操作:
    - remember()   — 存入记忆（自动分级）
    - recall()     — 检索记忆（跨层查询）
    - forget()     — 删除记忆
    - summarize()  — 压缩记忆

  预算管理:
    - gauge()      — 预算仪表盘
    - check()      — 预算状态检查
    - budget()     — 分配预算给各层
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from context.memory_hierarchy import (
    MemoryHierarchy, MemoryEntry, MemoryHierarchyConfig,
    MemoryTier, MessageRole, TokenEstimator,
)
from context.conversation_memory import ConversationMemory, ConversationConfig
from context.context_window import ContextWindow, ContextWindowConfig, BudgetReport, BudgetStatus
from context.context_assembler import ContextAssembler, AssembleConfig, AssembledContext
from context.compaction import CompactionConfig, ContentAwareCompactor

logger = logging.getLogger(__name__)


class ContextManager:
    """统一上下文管理器 — Agent 的记忆操作系统内核

    整合四个子系统:
    1. MemoryHierarchy  — 四层记忆存取 (Working / Short-Term / Long-Term / External)
    2. ConversationMemory — 多轮对话管理 (Turns / Topics / Phases)
    3. ContextWindow — 上下文窗口调度 (Budget / Pruning / Dedup)
    4. ContextAssembler — 上下文组装 (RAG / Expert / Consensus / Verification)

    用法:
        ctx = ContextManager(user_id="user_001")

        # 开始会话
        ctx.begin_session()

        # 第一轮: 设置角色和约束
        ctx.remember("你是医疗知识助手", tier=MemoryTier.WORKING, role=MessageRole.SYSTEM)
        ctx.remember("只基于文献回答", tier=MemoryTier.WORKING, role=MessageRole.SYSTEM)

        # 注入长期记忆 (技能/反思)
        ctx.remember("[技能] 高血压合并痛风: 避免NSAIDs", tier=MemoryTier.LONG_TERM)
        ctx.remember("[教训] 未检查肾功能即推荐布洛芬", tier=MemoryTier.LONG_TERM)

        # 加载检索结果
        ctx.load_documents(retrieved_docs)

        # 组装上下文
        result = ctx.prepare_context(
            query="高血压患者痛风怎么办?",
            strategy="simple_rag",
            role="你是医疗助手",
        )

        # 记录本轮对话
        ctx.add_conversation(
            query="高血压患者痛风怎么办?",
            response=llm_response,
            route_path="simple_rag",
            confidence=0.85,
        )

        # 查看预算
        print(ctx.gauge())
    """

    def __init__(
        self,
        user_id: str = "default",
        session_id: str = "",
        hierarchy_config: Optional[MemoryHierarchyConfig] = None,
        conversation_config: Optional[ConversationConfig] = None,
        window_config: Optional[ContextWindowConfig] = None,
        assemble_config: Optional[AssembleConfig] = None,
        compaction_config: Optional[CompactionConfig] = None,
    ):
        self.user_id = user_id
        self.session_id = session_id or f"sess_{user_id}_{int(time.time())}"

        self.hierarchy = MemoryHierarchy(hierarchy_config)
        self.conversation = ConversationMemory(
            user_id=user_id,
            session_id=self.session_id,
            config=conversation_config,
        )
        self.window = ContextWindow(window_config)
        self.assembler = ContextAssembler(assemble_config)
        self.compactor = ContentAwareCompactor(compaction_config)

        self._session_started = False
        self._query_count = 0

    def begin_session(self):
        """开始会话"""
        self._session_started = True
        self._query_count = 0
        logger.info(f"Context session started: {self.session_id}")

    def end_session(self):
        """结束会话"""
        self._session_started = False
        logger.info(f"Context session ended: {self.session_id} (queries={self._query_count})")

    # ============================================================
    # 记忆操作 — 统一的存取接口
    # ============================================================

    def remember(
        self,
        content: str,
        tier: MemoryTier = MemoryTier.SHORT_TERM,
        role: MessageRole = MessageRole.USER,
        importance: float = 0.5,
        metadata: Optional[dict] = None,
    ) -> MemoryEntry:
        """存入记忆 — 自动路由到正确的层级"""
        if tier == MemoryTier.WORKING:
            return self.hierarchy.add_working(content, role, importance, metadata)
        elif tier == MemoryTier.SHORT_TERM:
            return self.hierarchy.add_short_term(content, role, importance, metadata)
        elif tier == MemoryTier.LONG_TERM:
            return self.hierarchy.add_long_term(content, role, importance, metadata)
        elif tier == MemoryTier.EXTERNAL:
            return self.hierarchy.add_external(content, role, importance, metadata)

    def set_system_prompt(self, content: str, importance: float = 1.0):
        """设置系统提示 (永久层)"""
        self.hierarchy.set_working(content, MessageRole.SYSTEM, importance)
        self.window.set_permanent(content, importance)

    def append_system_prompt(self, content: str, importance: float = 0.5):
        """追加系统提示"""
        self.hierarchy.add_working(content, MessageRole.SYSTEM, importance)
        self.window.append_permanent(content, importance)

    def inject_skill(self, skill_text: str):
        """注入技能记忆到长期记忆层"""
        self.hierarchy.add_long_term(
            f"[技能] {skill_text}",
            role=MessageRole.MEMORY,
            importance=0.7,
            metadata={"type": "skill"},
        )

    def inject_reflection(self, reflection_text: str):
        """注入反思记忆到长期记忆层"""
        self.hierarchy.add_long_term(
            f"⚠️ {reflection_text}",
            role=MessageRole.MEMORY,
            importance=0.8,
            metadata={"type": "reflection"},
        )

    def inject_profile(self, profile_text: str):
        """注入患者画像到穿透项"""
        self.conversation.add_penetration(f"患者信息: {profile_text}")

    def load_documents(self, documents: list, max_tokens: int = 0):
        """加载检索结果到外部记忆层和上下文窗口"""
        self.hierarchy.load_external(
            documents,
            format_fn=lambda doc: (
                f"[{getattr(doc, 'source', '')}] {getattr(doc, 'content', str(doc))}"
            ),
        )
        self.window.load_deep(documents, max_tokens)

    def add_conversation(
        self,
        query: str,
        response: str,
        route_path: str = "",
        confidence: float = 0.0,
        departments: Optional[list[str]] = None,
    ):
        """记录一轮完整对话"""
        self._query_count += 1

        self.hierarchy.add_short_term(
            f"用户: {query}",
            role=MessageRole.USER,
            importance=0.6,
            metadata={"turn": self._query_count},
        )
        self.hierarchy.add_short_term(
            f"助手: {response[:800]}",
            role=MessageRole.ASSISTANT,
            importance=0.5,
            metadata={"turn": self._query_count, "route": route_path},
        )

        self.window.add_working(
            f"用户: {query}",
            importance=0.6,
            meta={"turn": self._query_count},
        )
        self.window.add_working(
            f"助手: {response[:500]}",
            importance=0.5,
            meta={"turn": self._query_count},
        )

        self.conversation.add_turn(
            query=query,
            response=response,
            route_path=route_path,
            confidence=confidence,
            departments=departments,
        )

    # ============================================================
    # 上下文组装
    # ============================================================

    def prepare_context(
        self,
        query: str,
        strategy: str = "simple_rag",
        role: str = "",
        constraints: Optional[list[str]] = None,
        documents: Optional[list] = None,
        profile: Optional[str] = None,
    ) -> str:
        """准备 LLM 消费上下文 — 一站式组装

        strategy:
            - "simple_rag": 简单 RAG
            - "mdt_expert": 多专家 ReAct
            - "consensus": 共识提炼
        """
        if documents:
            self.load_documents(documents)

        history = self.conversation.get_history_context()
        skills = self._collect_skills()
        reflections = self._collect_reflections()

        if strategy == "simple_rag":
            assembled = self.assembler.assemble_for_rag(
                role=role or "你是一个医疗知识助手",
                query=query,
                documents=documents,
                constraints=constraints,
                skills=skills,
                reflections=reflections,
                profile=profile,
                history=history,
            )
            return assembled.messages

        elif strategy == "mdt_expert":
            assembled = self.assembler.assemble_for_expert(
                department=role,
                query=query,
                profile=profile,
                reflection_hint="\n".join(reflections) if reflections else "",
                skill_hints="\n".join(skills) if skills else "",
            )
            return assembled.messages

        elif strategy == "consensus":
            assembled = self.assembler.assemble_for_consensus(
                expert_opinions=[query],
                profile=profile,
            )
            return assembled.messages

        return [{"role": "user", "content": query}]

    def render_context(self, query: Optional[str] = None) -> str:
        """渲染拼接后的完整上下文文本 (调试/日志用)"""
        return self.hierarchy.assemble_context(query)

    # ============================================================
    # 预算管理
    # ============================================================

    def gauge(self) -> str:
        """预算仪表盘"""
        return self.window.gauge()

    def check(self) -> BudgetReport:
        """检查预算状态"""
        return self.window.check_budget()

    def is_over_budget(self) -> bool:
        """是否超预算"""
        return self.check().status == BudgetStatus.OVERFLOW

    def budget_summary(self) -> dict:
        """预算摘要"""
        return self.hierarchy.usage_report()

    # ============================================================
    # 清理
    # ============================================================

    def clear_conversation(self):
        """清空对话层（保留系统提示和长期记忆）"""
        self.hierarchy.clear_tier(MemoryTier.SHORT_TERM)
        self.window.clear_working()

    def clear_all(self):
        """清空所有记忆"""
        for tier in MemoryTier:
            self.hierarchy.clear_tier(tier)
        self.window.clear()
        self.conversation.reset_session()

    # ============================================================
    # 状态
    # ============================================================

    def snapshot(self) -> dict:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "query_count": self._query_count,
            "memories": self.hierarchy.snapshot(),
            "conversation": self.conversation.stats(),
            "budget": self.check().__dict__,
        }

    # ============================================================
    # 内部辅助
    # ============================================================

    def _collect_skills(self) -> list[str]:
        """从长期记忆收集技能提示"""
        skills = []
        for entry in self.hierarchy._long_term:
            if entry.metadata.get("type") == "skill" or "[技能]" in entry.content:
                skills.append(entry.content)
        return skills[-3:]

    def _collect_reflections(self) -> list[str]:
        """从长期记忆收集反思提示"""
        reflections = []
        for entry in self.hierarchy._long_term:
            if entry.metadata.get("type") == "reflection" or "⚠️" in entry.content:
                reflections.append(entry.content)
        return reflections[-3:]


ContextBudget = MemoryHierarchyConfig
