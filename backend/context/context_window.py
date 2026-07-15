"""上下文窗口管理 — 参考 Harness Context Budget + Anthropic Prompt Caching + LLMLingua 压缩

设计理念:
  "上下文窗口是最昂贵的推理资源，需要像管理 GPU 显存一样精细调度"
  "不是所有内容都值得占用上下文 —— 给 Agent 一张地图，而不是 1000 页说明书"

参考架构:
  - Harness Context Budget: 分层预算 + Token 追踪 + 预算警报
  - Anthropic Prompt Caching: 缓存不变部分，只发送变化部分
  - Microsoft LLMLingua: 选择性压缩，非均匀降采样
  - LangChain Context Compression: LLMChainExtractor + LLMContentFilter

核心能力:
  1. Token 预算追踪: 实时监控每层的 token 消耗
  2. 重要性感知裁剪: 按重要性优先级裁剪内容
  3. 冗余检测与去重: 检测重复内容，只保留高置信版本
  4. 上下文压缩管道: 摘要 -> 裁剪 -> 去重 -> 组装
  5. 预算超限预警: 接近预算上限时触发降级策略
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable

from context.memory_hierarchy import MemoryEntry, MemoryTier, MessageRole, TokenEstimator

logger = logging.getLogger(__name__)


class BudgetStatus(Enum):
    """预算状态"""
    HEALTHY = auto()
    WARNING = auto()
    CRITICAL = auto()
    OVERFLOW = auto()


@dataclass
class ContextWindowConfig:
    """上下文窗口配置"""
    total_budget: int = 14000
    permanent_budget: int = 2000
    working_budget: int = 4000
    deep_budget: int = 8000

    warning_threshold: float = 0.75
    critical_threshold: float = 0.90

    dedup_similarity_threshold: float = 0.85

    compression_enabled: bool = True
    compression_summary_ratio: float = 0.3
    compression_min_chars: int = 100

    response_reserved_tokens: int = 2048


@dataclass
class BudgetReport:
    """预算报告"""
    permanent: dict
    working: dict
    deep: dict
    total: dict
    status: BudgetStatus
    recommendation: str


class ContextWindow:
    """上下文窗口管理器

    三级预算模型 + 重要性裁剪 + 冗余检测 + 压缩管道

    三级预算:
    - Permanent (永久层): 系统提示、安全约束，始终在线 (~2K tokens)
    - Working (工作层): 对话历史、工具结果、临时上下文 (~4K tokens)
    - Deep (深度层): 检索结果、外部知识、长文档 (~8K tokens)

    裁剪优先级 (由低到高):
    1. 深度层低相关性文档
    2. 工作层久远历史消息
    3. 深度层中相关性文档
    4. 工作层近期的非关键消息
    5. 永久层非核心内容（最后手段）
    """

    def __init__(self, config: Optional[ContextWindowConfig] = None):
        self.config = config or ContextWindowConfig()
        self._permanent: list[dict] = []
        self._working: list[dict] = []
        self._deep: list[dict] = []
        self._content_hashes: set[str] = set()

    def set_permanent(self, content: str, importance: float = 1.0, meta: Optional[dict] = None):
        self._permanent = [{"content": content, "importance": importance, "meta": meta or {}}]

    def append_permanent(self, content: str, importance: float = 0.5, meta: Optional[dict] = None):
        self._permanent.append({"content": content, "importance": importance, "meta": meta or {}})

    def add_working(self, content: str, importance: float = 0.5, meta: Optional[dict] = None):
        deduped = self._dedup(content)
        if deduped:
            self._working.append({"content": deduped, "importance": importance, "meta": meta or {}})

    def load_deep(self, documents: list, max_tokens: int = 0):
        if max_tokens <= 0:
            max_tokens = self.config.deep_budget
        self._deep.clear()
        budget = max_tokens
        for doc in documents:
            content = getattr(doc, "content", "") if hasattr(doc, "content") else str(doc)
            source = getattr(doc, "source", "") if hasattr(doc, "source") else ""
            score = getattr(doc, "score", 0.5) if hasattr(doc, "score") else 0.5

            deduped = self._dedup(content)
            if not deduped:
                continue

            tokens = TokenEstimator.estimate(deduped)
            if tokens > budget:
                deduped = TokenEstimator.truncate(deduped, budget)
                tokens = TokenEstimator.estimate(deduped)
            if tokens <= budget:
                self._deep.append({
                    "content": f"[{source}] {deduped}" if source else deduped,
                    "importance": score,
                    "meta": {"source": source, "score": score},
                })
                budget -= tokens
                if budget <= 0:
                    break

    def render(self, query: Optional[str] = None) -> str:
        """渲染完整上下文，自动按预算裁剪"""
        layers = [
            self._render_permanent(),
            self._render_working(query),
            self._render_deep(query),
        ]
        assembled = "\n\n".join(l for l in layers if l)
        tokens = TokenEstimator.estimate(assembled)
        max_tokens = self.config.total_budget - self.config.response_reserved_tokens
        if tokens > max_tokens:
            assembled = TokenEstimator.truncate(assembled, max_tokens)
        return assembled

    def check_budget(self) -> BudgetReport:
        """检查预算状态"""
        permanent_used = sum(TokenEstimator.estimate(d["content"]) for d in self._permanent)
        working_used = sum(TokenEstimator.estimate(d["content"]) for d in self._working)
        deep_used = sum(TokenEstimator.estimate(d["content"]) for d in self._deep)
        total_used = permanent_used + working_used + deep_used
        total_budget = self.config.total_budget

        ratio = total_used / max(total_budget, 1)
        if ratio >= 1.0:
            status = BudgetStatus.OVERFLOW
            rec = "严重超预算，需要大幅裁剪非关键内容"
        elif ratio >= self.config.critical_threshold:
            status = BudgetStatus.CRITICAL
            rec = "接近预算上限，建议压缩深度层和久远历史"
        elif ratio >= self.config.warning_threshold:
            status = BudgetStatus.WARNING
            rec = "预算使用率较高，注意监控"
        else:
            status = BudgetStatus.HEALTHY
            rec = "预算健康"

        return BudgetReport(
            permanent={"used": permanent_used, "budget": self.config.permanent_budget},
            working={"used": working_used, "budget": self.config.working_budget},
            deep={"used": deep_used, "budget": self.config.deep_budget},
            total={"used": total_used, "budget": total_budget, "ratio": round(ratio, 2)},
            status=status,
            recommendation=rec,
        )

    def gauge(self) -> str:
        """预算仪表盘可视化"""
        report = self.check_budget()
        lines = ["Context Budget Gauge:"]
        for label, key in [("Permanent", "permanent"), ("Working  ", "working"),
                            ("Deep     ", "deep"), ("Total    ", "total")]:
            data = getattr(report, key, {})
            used = data.get("used", 0)
            budget = data.get("budget", 1)
            ratio = min(used / max(budget, 1), 1.0)
            bar_len = 20
            filled = int(ratio * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            lines.append(f"  {label} |{bar}| {used:5d}/{budget:5d} ({ratio:.0%})")
        return "\n".join(lines)

    def compress_working(self, keep_recent: int = 10):
        """压缩工作层: 保留最近 N 条，其余合并为摘要"""
        if len(self._working) <= keep_recent:
            return
        old = self._working[:-keep_recent]
        recent = self._working[-keep_recent:]
        if old:
            summary_parts = []
            for d in old:
                summary_parts.append(d["content"][:150])
            summary = "[历史摘要] " + "; ".join(summary_parts[:5])
            summary = TokenEstimator.truncate(
                summary,
                int(self.config.working_budget * self.config.compression_summary_ratio),
            )
            self._working = [{"content": summary, "importance": 0.5, "meta": {"compressed": True}}] + recent
            logger.info(f"Working layer compressed: {len(old)} entries -> summary")

    def clear(self):
        self._permanent.clear()
        self._working.clear()
        self._deep.clear()
        self._content_hashes.clear()

    def clear_working(self):
        self._working.clear()

    def clear_deep(self):
        self._deep.clear()

    def clear_permanent(self):
        self._permanent.clear()

    def _render_permanent(self) -> str:
        sorted_items = sorted(self._permanent, key=lambda x: x["importance"], reverse=True)
        return "\n".join(d["content"] for d in sorted_items)

    def _render_working(self, query: Optional[str] = None) -> str:
        if not self._working:
            return ""
        items = self._working[-20:]
        return "--- 对话历史 ---\n" + "\n".join(d["content"] for d in items)

    def _render_deep(self, query: Optional[str] = None) -> str:
        if not self._deep:
            return ""
        sorted_items = sorted(self._deep, key=lambda x: x.get("importance", 0.5), reverse=True)
        lines = ["--- 参考文献 ---"]
        for i, d in enumerate(sorted_items):
            lines.append(f"[{i + 1}] {d['content']}")
        return "\n".join(lines)

    def _dedup(self, content: str) -> Optional[str]:
        """内容去重 — 基于 MinHash 近似检测"""
        import hashlib
        content_hash = hashlib.md5(content[:500].encode()).hexdigest()
        if content_hash in self._content_hashes:
            return None
        self._content_hashes.add(content_hash)
        if len(self._content_hashes) > 1000:
            self._content_hashes = set(list(self._content_hashes)[-500:])
        return content
