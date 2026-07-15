"""多轮对话记忆管理 — 参考 LangChain ConversationMemory + Mem0 Session Memory

设计理念:
  "Agent 的每一次推理都不应孤立发生，而是处于对话的连续流中"
  "对话记忆不仅仅是消息列表，还包括: 上下文继承、话题追踪、意图链、状态机"

参考架构:
  - LangChain ConversationSummaryBufferMemory: 滑动窗口 + 摘要
  - Mem0 Session Memory: 用户-level + 会话-level 双维度
  - AutoGen ConversableAgent: 消息历史 + context_window + max_consecutive_auto_reply
  - Zep Long-term Memory Store: 结构化事实 + 对话摘要 + 消息搜索

核心能力:
  1. 多轮上下文继承: 每轮将上一轮压缩后的上下文传递给下一轮
  2. 话题追踪: 识别对话中的话题切换，管理上下文边界
  3. 状态机: 跟踪对话阶段 (initial / fact_gathering / consultation / followup)
  4. 上下文穿透: 关键约束(如患者画像)在多轮中始终可见
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from context.memory_hierarchy import MemoryEntry, MemoryHierarchy, MemoryHierarchyConfig, MemoryTier, MessageRole

logger = logging.getLogger(__name__)


class ConversationPhase(Enum):
    """对话阶段状态机"""
    INITIAL = auto()
    FACT_GATHERING = auto()
    CONSULTATION = auto()
    FOLLOWUP = auto()
    CLOSED = auto()


@dataclass
class TurnRecord:
    """单轮对话记录"""
    turn_id: str
    user_query: str
    assistant_response: str
    route_path: str = ""
    confidence: float = 0.0
    departments: list[str] = field(default_factory=list)
    topic_labels: list[str] = field(default_factory=list)
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class ConversationConfig:
    """对话记忆配置"""
    max_turns: int = 20
    max_history_tokens: int = 4000
    summary_threshold_turns: int = 6
    topic_drift_threshold: float = 0.3
    context_penetration_ratio: float = 0.1


class ConversationMemory:
    """多轮对话记忆管理器

    参考 Mem0 Session Memory 的 session-level 管理 + LangChain 的 SummaryBuffer 模式

    用法:
        conv = ConversationMemory(user_id="user_001", session_id="sess_001")
        conv.add_turn("高血压怎么治?", "建议控制饮食...", route_path="simple_rag")
        history = conv.get_history_context(max_tokens=3000)
        conv.add_turn("那痛风呢?", "...", route_path="mdt")

    设计要点:
    - 每轮记录包含 query/response/route/confidence 元信息
    - 自动话题追踪，检测话题漂移
    - 上下文穿透: 关键信息（如患者画像约束）在多轮中自动注入
    - 摘要缓冲: 超过阈值轮数时，老轮次自动压缩为摘要
    """

    def __init__(
        self,
        user_id: str,
        session_id: str = "",
        config: Optional[ConversationConfig] = None,
    ):
        self.user_id = user_id
        self.session_id = session_id or f"sess_{user_id}_{int(time.time())}"
        self.config = config or ConversationConfig()
        self.turns: list[TurnRecord] = []
        self.phase = ConversationPhase.INITIAL
        self._summaries: list[str] = []
        self._penetration_items: list[str] = []
        self._topic_stack: list[str] = []
        self._turn_counter = 0

    def add_turn(
        self,
        query: str,
        response: str,
        route_path: str = "",
        confidence: float = 0.0,
        departments: Optional[list[str]] = None,
        topic: Optional[str] = None,
    ) -> TurnRecord:
        """记录一轮对话"""
        self._turn_counter += 1
        turn = TurnRecord(
            turn_id=f"{self.session_id}_t{self._turn_counter}",
            user_query=query,
            assistant_response=response,
            route_path=route_path,
            confidence=confidence,
            departments=departments or [],
            topic_labels=[topic] if topic else self._detect_topics(query),
        )
        self.turns.append(turn)
        self._update_phase(turn)
        self._maybe_summarize_old_turns()
        return turn

    def set_penetration(self, items: list[str]):
        """设置上下文穿透项 — 这些信息在每轮中始终可见

        典型用途: 患者画像约束、系统安全约束、当前上下文状态
        """
        self._penetration_items = items

    def add_penetration(self, item: str):
        """追加穿透项"""
        if item not in self._penetration_items:
            self._penetration_items.append(item)

    def get_history_context(self, max_tokens: int = 0) -> str:
        """获取格式化的对话历史上下文，供 LLM 消费

        包含:
        1. 上下文穿透项 (关键约束)
        2. 最近 N 轮完整对话 (摘要化老轮次)
        3. 当前话题提示
        """
        if max_tokens <= 0:
            max_tokens = self.config.max_history_tokens

        parts = []

        if self._penetration_items:
            parts.append("--- 持续约束 ---")
            for item in self._penetration_items:
                parts.append(f"  {item}")

        if self._summaries:
            parts.append("--- 历史摘要 ---")
            for s in self._summaries[-2:]:
                parts.append(s)

        recent_turns = self.turns[-self.config.max_turns:]
        if recent_turns:
            parts.append("--- 最近对话 ---")
            for turn in recent_turns:
                parts.append(f"用户: {turn.user_query}")
                resp_short = turn.assistant_response[:300]
                if len(turn.assistant_response) > 300:
                    resp_short += "..."
                parts.append(f"助手: {resp_short}")

        context = "\n\n".join(parts)
        from context.memory_hierarchy import TokenEstimator
        if TokenEstimator.estimate(context) > max_tokens:
            context = TokenEstimator.truncate(context, max_tokens)
        return context

    def get_recent_messages(self, n: int = 5) -> list[dict]:
        """获取最近 N 轮消息对"""
        messages = []
        for turn in self.turns[-n:]:
            messages.append({"role": "user", "content": turn.user_query})
            messages.append({"role": "assistant", "content": turn.assistant_response[:500]})
        return messages

    def get_topic_chain(self) -> list[str]:
        """获取话题链"""
        return [t for turn in self.turns for t in turn.topic_labels]

    def is_topic_drift(self, new_query: str) -> bool:
        """检测是否发生了话题漂移"""
        if len(self.turns) < 2:
            return False
        recent_topics = self.get_topic_chain()[-3:]
        new_topics = self._detect_topics(new_query)
        overlap = len(set(recent_topics) & set(new_topics))
        if recent_topics:
            ratio = overlap / max(len(set(recent_topics)), 1)
            return ratio < self.config.topic_drift_threshold
        return False

    def reset_session(self):
        """重置会话（开始新对话）"""
        self.turns.clear()
        self._summaries.clear()
        self._topic_stack.clear()
        self.phase = ConversationPhase.INITIAL
        self.session_id = f"sess_{self.user_id}_{int(time.time())}"

    def _update_phase(self, turn: TurnRecord):
        """更新对话阶段状态机"""
        if len(self.turns) <= 1:
            self.phase = ConversationPhase.FACT_GATHERING
        elif turn.route_path == "mdt":
            self.phase = ConversationPhase.CONSULTATION
        elif len(self.turns) >= 3 and turn.route_path in ("simple_rag", "safe_fallback"):
            self.phase = ConversationPhase.FOLLOWUP

    def _maybe_summarize_old_turns(self):
        """老轮次超过阈值时，压缩为摘要"""
        if len(self.turns) > self.config.summary_threshold_turns:
            excess = len(self.turns) - self.config.summary_threshold_turns
            if excess > 0:
                old = self.turns[:excess]
                summary = self._generate_turn_summary(old)
                if summary:
                    self._summaries.append(summary)
                self.turns = self.turns[excess:]
                logger.info(f"Conversation: summarized {excess} old turns")

    def _generate_turn_summary(self, turns: list[TurnRecord]) -> str:
        """简单的话题+结论摘要"""
        if not turns:
            return ""
        topics = list(set(t for turn in turns for t in turn.topic_labels))
        queries = [t.user_query[:60] for t in turns]
        return f"[摘要] 讨论了 {len(turns)} 轮, 话题: {', '.join(topics[:5])}, 涉及: {'; '.join(queries[:3])}"

    def _detect_topics(self, query: str) -> list[str]:
        """简单的话题检测（基于关键词）"""
        topic_keywords = {
            "心血管": ["高血压", "血压", "心脏", "心衰", "冠心病", "心梗"],
            "消化": ["胃", "胃溃疡", "腹泻", "便秘", "肝", "肠"],
            "风湿免疫": ["痛风", "风湿", "关节炎", "免疫", "红斑狼疮"],
            "药物": ["用药", "药物", "处方", "禁忌", "相互作用", "剂量"],
            "内分泌": ["糖尿病", "血糖", "甲状腺", "激素"],
            "肾": ["肾功能", "肾病", "透析", "肌酐"],
            "神经": ["头痛", "眩晕", "失眠", "中风", "帕金森"],
            "呼吸": ["咳嗽", "哮喘", "肺炎", "呼吸"],
        }
        detected = []
        query_lower = query.lower()
        for topic, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected.append(topic)
        return detected or ["通用"]

    def stats(self) -> dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "total_turns": len(self.turns),
            "phase": self.phase.name,
            "summary_count": len(self._summaries),
            "penetration_count": len(self._penetration_items),
            "topic_chain": self.get_topic_chain()[-5:],
        }
