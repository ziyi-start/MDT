"""事件驱动的情节记忆 — 参考 EM-LLM (ICLR 2025) + 人类情节记忆机制

设计理念:
  "人类记忆不是均匀的时间序列，而是由'事件'分割的离散片段"
  "Agent 的推理流也应该按认知事件分割：症状分析→检索→推理→结论"

参考架构:
  - EM-LLM (Fountas et al. 2024/ICLR 2025):
    Bayesian surprise + graph-theoretic boundary refinement 自动检测事件边界
    两阶段检索: similarity-based + temporally contiguous
    处理 10M+ tokens，超越 full-context 模型
  - CoALA episodic memory: 存储 Agent 过去行动的序列
  - 人类情节记忆: 边界检测 (boundary detection) + 事件编码 (event encoding)

核心机制:
  1. 自动事件分割: 基于 token 信息密度突变 (surprise) 自动检测事件边界
  2. 事件存储: 每个事件独立存储，携带时间戳、主题、参与者
  3. 两阶段检索: 语义相似召回 + 时间邻接扩展
  4. 事件链: 追踪事件之间的因果/时序关系

医学 MDT 场景价值:
  - 一次会诊分割为: [症状分析] → [文献检索] → [科室推理] → [共识形成] → [方案输出]
  - 下次查询时可精确定位到"上次讨论布洛芬禁忌时的推理片段"
  - 跨会诊的事件关联: "和三天前的那次痛风讨论类似..."
"""

from __future__ import annotations

import logging
import time
import math
import hashlib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from context.memory_hierarchy import TokenEstimator

logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型枚举"""
    SYMPTOM_ANALYSIS = auto()
    LITERATURE_SEARCH = auto()
    DEPARTMENT_REASONING = auto()
    TOOL_CALL = auto()
    CONSENSUS_FORMATION = auto()
    TREATMENT_PLAN = auto()
    SAFETY_CHECK = auto()
    DECISION = auto()
    REFLECTION = auto()
    UNKNOWN = auto()


@dataclass
class EpisodicEvent:
    """情节事件 — 离散的认知片段"""
    event_id: str
    event_type: EventType
    content: str
    summary: str = ""
    timestamp: float = 0.0
    duration_ms: float = 0.0
    token_count: int = 0
    surprise_score: float = 0.0
    topic_labels: list[str] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    prev_event_id: str = ""
    next_event_id: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.token_count:
            self.token_count = TokenEstimator.estimate(self.content)
        if not self.summary:
            self.summary = self.content[:150]


@dataclass
class EventMemoryConfig:
    """事件记忆配置"""
    surprise_threshold: float = 1.5
    window_size: int = 50
    max_events_per_session: int = 100
    temporal_retrieval_window: int = 3
    similarity_weight: float = 0.7
    temporal_weight: float = 0.3


class EventSegmenter:
    """事件分割器 — 基于 Bayesian surprise 的事件边界检测

    参考 EM-LLM 论文:
      Surprise = -log P(tokens | prior context)
      当新 token 序列与先前上下文的信息分布差异超过阈值时，标记事件边界。

    简化实现:
      用 token 分布的 KL-divergence 近似，当相邻 token 块的词表分布
      差异超过 surprise_threshold 时，标记边界。
    """

    def __init__(self, config: Optional[EventMemoryConfig] = None):
        self.config = config or EventMemoryConfig()
        self._token_buffer: list[str] = []
        self._prior_dist: dict[str, float] = {}

    def should_split(self, new_content: str) -> tuple[bool, float]:
        """判断是否应该在当前内容处分割事件

        返回: (是否分割, surprise_score)
        """
        tokens = self._tokenize(new_content)
        self._token_buffer.extend(tokens)

        if len(self._token_buffer) < self.config.window_size:
            return False, 0.0

        current_dist = self._compute_distribution(self._token_buffer[-self.config.window_size:])
        surprise = self._kl_divergence(current_dist, self._prior_dist) if self._prior_dist else 0.0

        if surprise > self.config.surprise_threshold or not self._prior_dist:
            self._prior_dist = current_dist
            self._token_buffer = self._token_buffer[-self.config.window_size:]
            return True, surprise

        self._prior_dist = self._smooth_merge(self._prior_dist, current_dist, alpha=0.3)
        return False, surprise

    def reset(self):
        self._token_buffer.clear()
        self._prior_dist.clear()

    def _tokenize(self, text: str) -> list[str]:
        import re
        tokens = []
        for word in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]+|\d+', text):
            if len(word) <= 2:
                tokens.append(word)
            else:
                for i in range(0, len(word), 2):
                    tokens.append(word[i:i+2])
        return tokens or [text[:5]]

    def _compute_distribution(self, tokens: list[str]) -> dict[str, float]:
        dist = {}
        for t in tokens:
            dist[t] = dist.get(t, 0) + 1
        total = len(tokens) or 1
        return {k: v / total for k, v in dist.items()}

    def _kl_divergence(self, p: dict[str, float], q: dict[str, float]) -> float:
        epsilon = 1e-10
        kl = 0.0
        all_keys = set(p.keys()) | set(q.keys())
        for k in all_keys:
            pk = p.get(k, epsilon)
            qk = q.get(k, epsilon)
            kl += pk * math.log(pk / qk)
        return kl

    def _smooth_merge(self, prior: dict, current: dict, alpha: float) -> dict:
        merged = prior.copy()
        for k, v in current.items():
            merged[k] = alpha * v + (1 - alpha) * merged.get(k, 0)
        return merged


class EventMemory:
    """事件驱动的情节记忆管理器

    核心流程:
      1. segment: 自动检测事件边界，分割对话流
      2. store: 每个事件独立存储
      3. retrieve: 语义相似 + 时间邻接两阶段检索
      4. chain: 构建事件因果/时序链

    用法:
        em = EventMemory()
        em.ingest("患者高血压3年，近期痛风发作...")  # 自动分割为事件
        em.ingest("检索结果：NSAIDs对胃溃疡患者禁忌...")
        events = em.retrieve("痛风的止痛方案", top_k=3)
    """

    def __init__(self, config: Optional[EventMemoryConfig] = None):
        self.config = config or EventMemoryConfig()
        self.segmenter = EventSegmenter(config)
        self.events: list[EpisodicEvent] = []
        self._event_counter = 0
        self._current_event_type = EventType.UNKNOWN
        self._current_event_start = time.time()

    def ingest(self, content: str, event_type: Optional[EventType] = None,
               metadata: Optional[dict] = None) -> Optional[EpisodicEvent]:
        """摄取新内容，自动检测事件边界

        如果检测到事件边界，结束当前事件并开始新事件。
        """
        should_split, surprise = self.segmenter.should_split(content)

        if should_split and self.events:
            last = self.events[-1]
            last.duration_ms = (time.time() - self._current_event_start) * 1000
            self._current_event_start = time.time()

        if should_split or not self.events:
            event = self._create_event(content, event_type, surprise, metadata)
            self.events.append(event)
            if len(self.events) > self.config.max_events_per_session:
                self.events = self.events[-self.config.max_events_per_session:]
            return event

        if self.events:
            self.events[-1].content += "\n" + content
            self.events[-1].token_count = TokenEstimator.estimate(self.events[-1].content)
            self.events[-1].summary = self.events[-1].content[:150]

        return None

    def mark_event_type(self, event_type: EventType):
        """标记当前事件类型"""
        self._current_event_type = event_type
        if self.events:
            self.events[-1].event_type = event_type

    def retrieve(self, query: str, top_k: int = 3, expand_temporal: bool = True) -> list[EpisodicEvent]:
        """两阶段检索: 语义相似 + 时间邻接

        Stage 1: 按语义相似度召回 top_k 个事件
        Stage 2: 对每个召回事件，扩展其前后 temporal_retrieval_window 个邻接事件
        """
        if not self.events:
            return []

        scored = []
        for i, event in enumerate(self.events):
            sim = self._compute_similarity(query, event.content)
            scored.append((sim, i, event))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected_indices = set()
        result = []

        for sim, idx, event in scored[:top_k]:
            if idx not in selected_indices:
                selected_indices.add(idx)
                result.append(event)

            if expand_temporal:
                window = self.config.temporal_retrieval_window
                for offset in range(-window, window + 1):
                    neighbor_idx = idx + offset
                    if 0 <= neighbor_idx < len(self.events) and neighbor_idx not in selected_indices:
                        selected_indices.add(neighbor_idx)
                        result.append(self.events[neighbor_idx])

        result.sort(key=lambda e: e.timestamp)
        return result[:top_k + 2 * self.config.temporal_retrieval_window]

    def get_event_chain(self, event_id: str, max_depth: int = 5) -> list[EpisodicEvent]:
        """获取事件链 — 前后遍历因果/时序关系"""
        chain = []
        for event in self.events:
            if event.event_id == event_id:
                chain.append(event)
                current = event
                depth = 0
                while current.next_event_id and depth < max_depth:
                    for e in self.events:
                        if e.event_id == current.next_event_id:
                            chain.append(e)
                            current = e
                            break
                    depth += 1
                    if depth >= max_depth:
                        break
                break
        return chain

    def get_session_summary(self) -> str:
        """获取当前会话的事件摘要"""
        if not self.events:
            return ""
        lines = ["[事件时间线]"]
        for i, event in enumerate(self.events):
            type_label = event.event_type.name.replace("_", " ").title()
            lines.append(
                f"  E{i}: [{type_label}] {event.summary[:100]} "
                f"(surprise={event.surprise_score:.2f}, tokens={event.token_count})"
            )
        return "\n".join(lines)

    def reset(self):
        self.events.clear()
        self.segmenter.reset()
        self._event_counter = 0

    def stats(self) -> dict:
        return {
            "total_events": len(self.events),
            "avg_tokens_per_event": (
                sum(e.token_count for e in self.events) / max(len(self.events), 1)
            ),
            "event_types": {
                t.name: sum(1 for e in self.events if e.event_type == t)
                for t in EventType
            },
        }

    def _create_event(self, content: str, event_type: Optional[EventType],
                      surprise: float, metadata: Optional[dict]) -> EpisodicEvent:
        self._event_counter += 1
        event = EpisodicEvent(
            event_id=f"evt_{self._event_counter}_{int(time.time())}",
            event_type=event_type or self._current_event_type or EventType.UNKNOWN,
            content=content,
            surprise_score=surprise,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        if self.events:
            self.events[-1].next_event_id = event.event_id
            event.prev_event_id = self.events[-1].event_id
        return event

    def _compute_similarity(self, query: str, content: str) -> float:
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0
        content_lower = content.lower()
        matched = sum(1 for t in query_terms if t in content_lower)
        return matched / len(query_terms)
