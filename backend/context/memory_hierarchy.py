"""多层级记忆架构 — 参考 MemGPT + LangChain + Harness 设计模式

设计理念:
  "Agent的记忆不是一个扁平的上下文窗口，而是一个分层存储系统"
  "就像操作系统管理内存页：热数据在缓存，温数据在内存，冷数据在磁盘"

参考架构:
  - MemGPT (Letta): OS-inspired memory hierarchy (Core/Archival/Recall)
  - LangChain: ConversationBufferMemory + SummaryMemory + VectorStoreMemory
  - CrewAI: Short-term + Long-term + Entity + User memories
  - Harness: Permanent / Working / Deep 三层上下文预算
  - Mem0: User/Session/Agent 三维度记忆

四层记忆架构:
  ┌─────────────────────────────────────────────────────────────┐
  │  L0: Working Memory (工作记忆) — 当前推理上下文              │
  │      System Prompt + Task Goal + Tool Defs + Active Context │
  │      容量: ~2000 tokens, 始终在上下文窗口内                    │
  ├─────────────────────────────────────────────────────────────┤
  │  L1: Short-Term Memory (短期记忆) — 当前会话窗口             │
  │      Conversation History (滑动窗口) + Tool Results         │
  │      容量: ~4000 tokens, 按时间/重要性动态管理                 │
  ├─────────────────────────────────────────────────────────────┤
  │  L2: Long-Term Memory (长期记忆) — 跨会话持久化              │
  │      Episodic Summaries + Semantic Facts + Procedural Skills│
  │      容量: ~6000 tokens, 按需检索注入                          │
  ├─────────────────────────────────────────────────────────────┤
  │  L3: External Memory (外部记忆) — 知识库                     │
  │      Medical KB + Drug DB + Guidelines + External APIs      │
  │      容量: ~8000 tokens, 按需检索注入                          │
  └─────────────────────────────────────────────────────────────┘

记忆逐出策略 (Eviction Policy):
  - LRU: 最近最少使用的内容优先逐出
  - Importance: 高重要性(message.importance >= 0.7)的内容优先保留
  - Recency: 最近的内容优先保留 (指数衰减权重)
  - Relevance: 与当前查询语义相关的内容优先保留
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class MemoryTier(Enum):
    """记忆层级"""
    WORKING = auto()
    SHORT_TERM = auto()
    LONG_TERM = auto()
    EXTERNAL = auto()


class MessageRole(Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    MEMORY = "memory"


class MessageImportance(Enum):
    """消息重要性等级"""
    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.5
    LOW = 0.3
    TRIVIAL = 0.1


@dataclass
class MemoryEntry:
    """记忆条目 — 统一表示各层的记忆单元"""
    tier: MemoryTier
    role: MessageRole
    content: str
    importance: float = 0.5
    timestamp: float = 0.0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)
    entry_id: str = ""

    _auto_id_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        import time
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.entry_id:
            MemoryEntry._auto_id_counter += 1
            self.entry_id = f"mem_{MemoryEntry._auto_id_counter}_{int(self.timestamp)}"


@dataclass
class MemoryHierarchyConfig:
    """记忆层级配置 — 参考 Harness + MemGPT 参数设计"""
    working_capacity: int = 2000
    short_term_capacity: int = 4000
    long_term_capacity: int = 6000
    external_capacity: int = 8000
    total_capacity: int = 14000

    short_term_window_size: int = 20
    importance_threshold: float = 0.5
    recency_decay_rate: float = 0.1

    summarization_trigger_ratio: float = 0.8
    max_summary_length: int = 500
    summary_compression_ratio: float = 0.3


class MemoryHierarchy:
    """四层记忆层级管理器

    参考 MemGPT 的 OS-inspired memory 设计:
    - Working memory (L0): 像 CPU 缓存，始终在线，容量极小
    - Short-term memory (L1): 像 RAM，当前会话数据，有容量限制
    - Long-term memory (L2): 像 SSD，持久化存储，按需加载
    - External memory (L3): 像网络存储，外部知识库，按需检索

    用法:
        hierarchy = MemoryHierarchy()
        hierarchy.add_working("你是医疗专家...", role=MessageRole.SYSTEM)
        hierarchy.add_short_term("用户: 高血压如何处理?", role=MessageRole.USER)
        hierarchy.add_long_term(episodic_summary, role=MessageRole.MEMORY)
        assembled = hierarchy.assemble_context(query="高血压用药")
    """

    def __init__(self, config: Optional[MemoryHierarchyConfig] = None):
        self.config = config or MemoryHierarchyConfig()

        self._working: list[MemoryEntry] = []
        self._short_term: list[MemoryEntry] = []
        self._long_term: list[MemoryEntry] = []
        self._external: list[MemoryEntry] = []

        self._token_counter = TokenEstimator()

    def add_working(self, content: str, role: MessageRole = MessageRole.SYSTEM,
                    importance: float = 1.0, metadata: Optional[dict] = None):
        entry = MemoryEntry(
            tier=MemoryTier.WORKING,
            role=role,
            content=content,
            importance=importance,
            token_count=self._token_counter.estimate(content),
            metadata=metadata or {},
        )
        self._working.append(entry)
        self._evict_if_needed(MemoryTier.WORKING)
        return entry

    def add_short_term(self, content: str, role: MessageRole = MessageRole.USER,
                       importance: float = 0.5, metadata: Optional[dict] = None):
        entry = MemoryEntry(
            tier=MemoryTier.SHORT_TERM,
            role=role,
            content=content,
            importance=importance,
            token_count=self._token_counter.estimate(content),
            metadata=metadata or {},
        )
        self._short_term.append(entry)
        self._evict_if_needed(MemoryTier.SHORT_TERM)
        self._maybe_summarize_short_term()
        self._enforce_window_limit()
        return entry

    def add_long_term(self, content: str, role: MessageRole = MessageRole.MEMORY,
                      importance: float = 0.7, metadata: Optional[dict] = None):
        entry = MemoryEntry(
            tier=MemoryTier.LONG_TERM,
            role=role,
            content=content,
            importance=importance,
            token_count=self._token_counter.estimate(content),
            metadata=metadata or {},
        )
        self._long_term.append(entry)
        self._evict_if_needed(MemoryTier.LONG_TERM)
        return entry

    def add_external(self, content: str, role: MessageRole = MessageRole.MEMORY,
                     importance: float = 0.6, metadata: Optional[dict] = None):
        entry = MemoryEntry(
            tier=MemoryTier.EXTERNAL,
            role=role,
            content=content,
            importance=importance,
            token_count=self._token_counter.estimate(content),
            metadata=metadata or {},
        )
        self._external.append(entry)
        self._evict_if_needed(MemoryTier.EXTERNAL)
        return entry

    def set_working(self, content: str, role: MessageRole = MessageRole.SYSTEM,
                    importance: float = 1.0, metadata: Optional[dict] = None):
        self._working.clear()
        return self.add_working(content, role, importance, metadata)

    def clear_tier(self, tier: MemoryTier):
        if tier == MemoryTier.WORKING:
            self._working.clear()
        elif tier == MemoryTier.SHORT_TERM:
            self._short_term.clear()
        elif tier == MemoryTier.LONG_TERM:
            self._long_term.clear()
        elif tier == MemoryTier.EXTERNAL:
            self._external.clear()

    def load_external(self, documents: list, format_fn: Optional[Callable] = None):
        self._external.clear()
        budget = self.config.external_capacity
        for doc in documents:
            content = format_fn(doc) if format_fn else str(doc)
            tokens = self._token_counter.estimate(content)
            if tokens > budget:
                content = self._token_counter.truncate(content, budget)
                tokens = self._token_counter.estimate(content)
            if tokens <= budget:
                self.add_external(content, role=MessageRole.MEMORY, importance=0.6)
                budget -= tokens
                if budget <= 0:
                    break

    def _score_entry(self, entry: MemoryEntry, query: Optional[str] = None) -> float:
        """综合评分: 重要性 + 时间衰减 + 相关性"""
        import time
        now = time.time()
        age_hours = max((now - entry.timestamp) / 3600, 0)
        recency_score = max(0, 1.0 - self.config.recency_decay_rate * age_hours)

        relevance_score = 0.5
        if query:
            relevance_score = self._compute_relevance(entry.content, query)

        return (
            0.4 * entry.importance +
            0.3 * recency_score +
            0.3 * relevance_score
        )

    def _compute_relevance(self, content: str, query: str) -> float:
        """简单的关键词重叠相关性计算"""
        if not query:
            return 0.5
        content_lower = content.lower()
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.5
        matched = sum(1 for t in query_terms if t in content_lower)
        return min(matched / len(query_terms), 1.0)

    def _evict_if_needed(self, tier: MemoryTier):
        """按容量逐出低分条目"""
        entries = self._get_tier_list(tier)
        capacity = self._get_tier_capacity(tier)
        total_tokens = sum(e.token_count for e in entries)

        while total_tokens > capacity and entries:
            scored = [(self._score_entry(e), e) for e in entries]
            scored.sort(key=lambda x: x[0])
            removed = entries.pop(entries.index(scored[0][1]))
            total_tokens -= removed.token_count
            logger.debug(f"Memory evict [{tier.name}]: score={scored[0][0]:.2f}, tokens={removed.token_count}")

    def _maybe_summarize_short_term(self):
        """短期记忆过多时触发摘要压缩"""
        total = sum(e.token_count for e in self._short_term)
        threshold = int(self.config.short_term_capacity * self.config.summarization_trigger_ratio)
        if total > threshold and len(self._short_term) > 4:
            self._compress_short_term_to_summary()

    def _compress_short_term_to_summary(self):
        """将短期记忆最老的条目压缩为摘要，存入长期记忆"""
        import time
        mid = max(1, len(self._short_term) // 2)
        old_entries = self._short_term[:mid]

        summary_parts = []
        for e in old_entries:
            summary_parts.append(f"[{e.role.value}] {e.content[:200]}")

        summary = "[对话摘要]\n" + "\n".join(summary_parts)
        summary = self._token_counter.truncate(summary, self.config.max_summary_length)

        self.add_long_term(
            content=summary,
            role=MessageRole.MEMORY,
            importance=0.6,
            metadata={"type": "summary", "source_count": len(old_entries), "timestamp": time.time()},
        )

        self._short_term = self._short_term[mid:]
        logger.info(f"Short-term memory compressed: {len(old_entries)} entries -> summary ({len(summary)} chars)")

    def _enforce_window_limit(self):
        """强制执行消息条数上限: 超过 short_term_window_size 则丢弃最老的"""
        limit = self.config.short_term_window_size
        if len(self._short_term) > limit:
            excess = len(self._short_term) - limit
            self._short_term = self._short_term[excess:]
            logger.debug(f"Short-term window enforced: dropped {excess} oldest entries (limit={limit})")

    def assemble_context(self, query: Optional[str] = None, max_tokens: int = 0) -> str:
        """组装完整上下文 — 四层拼接为 LLM 可消费的文本"""
        if max_tokens <= 0:
            max_tokens = self.config.total_capacity

        layers = []

        if self._working:
            layers.append("\n".join(e.content for e in self._working))

        if self._short_term:
            scored = [(self._score_entry(e, query), e) for e in self._short_term]
            scored.sort(key=lambda x: x[0], reverse=True)
            window = scored[:self.config.short_term_window_size]
            layers.append("--- 对话历史 ---")
            layers.extend(e.content for _, e in window)

        if self._long_term:
            scored = [(self._score_entry(e, query), e) for e in self._long_term]
            scored.sort(key=lambda x: x[0], reverse=True)
            layers.append("--- 长期记忆 ---")
            layers.extend(e.content for _, e in scored)

        if self._external:
            layers.append("--- 参考文献 ---")
            for i, e in enumerate(self._external):
                layers.append(f"[{i + 1}] {e.content}")

        assembled = "\n\n".join(layers)
        tokens = self._token_counter.estimate(assembled)
        if tokens > max_tokens:
            assembled = self._token_counter.truncate(assembled, max_tokens)
        return assembled

    def usage_report(self) -> dict:
        """各层使用情况报告"""
        def tier_usage(entries, capacity):
            used = sum(e.token_count for e in entries)
            return {"used": used, "capacity": capacity, "ratio": round(used / max(capacity, 1), 2)}

        return {
            "working": tier_usage(self._working, self.config.working_capacity),
            "short_term": tier_usage(self._short_term, self.config.short_term_capacity),
            "long_term": tier_usage(self._long_term, self.config.long_term_capacity),
            "external": tier_usage(self._external, self.config.external_capacity),
            "total": {
                "used": sum(e.token_count for t in [self._working, self._short_term, self._long_term, self._external] for e in t),
                "capacity": self.config.total_capacity,
            },
        }

    def _get_tier_list(self, tier: MemoryTier) -> list[MemoryEntry]:
        return {
            MemoryTier.WORKING: self._working,
            MemoryTier.SHORT_TERM: self._short_term,
            MemoryTier.LONG_TERM: self._long_term,
            MemoryTier.EXTERNAL: self._external,
        }[tier]

    def _get_tier_capacity(self, tier: MemoryTier) -> int:
        return {
            MemoryTier.WORKING: self.config.working_capacity,
            MemoryTier.SHORT_TERM: self.config.short_term_capacity,
            MemoryTier.LONG_TERM: self.config.long_term_capacity,
            MemoryTier.EXTERNAL: self.config.external_capacity,
        }[tier]

    def snapshot(self) -> dict:
        return {
            "working_count": len(self._working),
            "short_term_count": len(self._short_term),
            "long_term_count": len(self._long_term),
            "external_count": len(self._external),
            "usage": self.usage_report(),
        }


class TokenEstimator:
    """Token 估算器 — 支持中英文混合估算

    中文: ~1.5 token/字符
    英文: ~1 token/4 字符
    数字/符号: ~1 token/字符
    """

    @staticmethod
    def estimate(text: str) -> int:
        if not text:
            return 0
        import re
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - chinese_chars - sum(len(w) for w in re.findall(r'[a-zA-Z]+', text))
        return int(chinese_chars * 1.5 + english_words * 0.8 + other_chars * 0.5)

    @staticmethod
    def truncate(text: str, max_tokens: int) -> str:
        estimated = TokenEstimator.estimate(text)
        if estimated <= max_tokens:
            return text
        ratio = max_tokens / max(estimated, 1)
        keep_chars = int(len(text) * ratio * 0.9)
        return text[:keep_chars] + "\n... [truncated]"

    @staticmethod
    def count_tokens_in_messages(messages: list) -> int:
        """统计消息列表的总 token 数"""
        total = 0
        for msg in messages:
            content = getattr(msg, "content", "") or str(msg)
            total += TokenEstimator.estimate(content)
        return total
