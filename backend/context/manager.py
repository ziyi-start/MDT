"""上下文预算管理器 - 像管理预算一样管理上下文

设计理念（来自 Agent Harness Context Management）:
  "管理上下文像管理预算一样 —— 永久层压缩到 ≤8K，深度内容按需加载"
  "通过分层加载对抗上下文污染"
  "给 Agent 一张地图，而不是 1000 页说明书"

核心机制:
  1. 三层上下文:
     - Permanent (永久层): 系统提示、核心约束，始终保留
     - Working (工作层): 当前对话、临时信息，按需保留
     - Deep (深度层): 检索结果、长文档，按需加载/卸载
  2. Token 预算追踪: 监控每层的 token 消耗
  3. 压缩策略: 自动摘要/裁剪超过预算的内容
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# Token 估算器
# ============================================================

class TokenCounter:
    """简易 Token 估算器

    中文 ≈ 1 token/字符
    英文 ≈ 1 token/4 字符
    """

    @staticmethod
    def estimate(text: str) -> int:
        if not text:
            return 0
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + other_chars / 4)

    @staticmethod
    def truncate(text: str, max_tokens: int) -> str:
        """按 token 预算截断文本"""
        if TokenCounter.estimate(text) <= max_tokens:
            return text
        ratio = max_tokens / max(TokenCounter.estimate(text), 1)
        keep_chars = int(len(text) * ratio * 0.9)
        return text[:keep_chars] + "\n... [已截断]"


# ============================================================
# 上下文预算
# ============================================================

@dataclass
class ContextBudget:
    permanent_tokens: int = 2000
    working_tokens: int = 4000
    deep_tokens: int = 8000
    total_tokens: int = 14000

    @property
    def remaining(self) -> int:
        return self.total_tokens

    def layer_budget(self, layer: str) -> int:
        return {
            "permanent": self.permanent_tokens,
            "working": self.working_tokens,
            "deep": self.deep_tokens,
        }.get(layer, 0)


# ============================================================
# 上下文管理器
# ============================================================

class ContextManager:
    """分层上下文管理器

    用法:
        ctx = ContextManager()
        ctx.set_permanent("你是医疗专家...")
        ctx.add_working("用户: 我有高血压...")
        ctx.load_deep(docs)  # 按需加载
        print(ctx.render())  # 组装完整上下文
        print(ctx.summary()) # 查看预算使用
    """

    def __init__(self, budget: Optional[ContextBudget] = None):
        self.budget = budget or ContextBudget()
        self.counter = TokenCounter()

        self._permanent: list[str] = []
        self._working: list[str] = []
        self._deep: list[str] = []
        self._deep_metadata: dict[str, str] = {}

    # ---- 各层操作 ----

    def set_permanent(self, text: str):
        """设置永久层 (系统提示、核心约束)"""
        self._permanent = [text]

    def append_permanent(self, text: str):
        """追加永久层内容"""
        self._permanent.append(text)

    def add_working(self, text: str):
        """添加工作层 (对话历史)"""
        self._working.append(text)

    def clear_working(self):
        """清空工作层"""
        self._working.clear()

    def load_deep(self, documents: list, max_tokens: int = 0):
        """按需加载深度层 (检索结果)

        参数:
            documents: DocumentChunk 列表或 dict 列表
            max_tokens: 深度层 token 预算
        """
        if max_tokens <= 0:
            max_tokens = self.budget.deep_tokens

        self._deep.clear()
        self._deep_metadata.clear()
        budget = max_tokens

        for doc in documents:
            if hasattr(doc, "content"):
                content = doc.content
                source = getattr(doc, "source", "")
                doc_id = getattr(doc, "doc_id", "")
            elif isinstance(doc, dict):
                content = doc.get("content", "")
                source = doc.get("source", "")
                doc_id = doc.get("doc_id", doc.get("id", ""))
            else:
                continue

            tokens = self.counter.estimate(content)
            if tokens > budget:
                content = self.counter.truncate(content, budget)
                tokens = self.counter.estimate(content)

            if tokens <= budget:
                self._deep.append(f"[{source}] {content}")
                self._deep_metadata[doc_id] = source
                budget -= tokens
                if budget <= 0:
                    break

    def clear_deep(self):
        """卸载深度层"""
        self._deep.clear()
        self._deep_metadata.clear()

    # ---- Token 预算 ----

    def usage(self, layer: str) -> int:
        if layer == "permanent":
            return self.counter.estimate("\n".join(self._permanent))
        elif layer == "working":
            return self.counter.estimate("\n".join(self._working))
        elif layer == "deep":
            return self.counter.estimate("\n".join(self._deep))
        return 0

    @property
    def total_usage(self) -> int:
        return self.usage("permanent") + self.usage("working") + self.usage("deep")

    def is_over_budget(self) -> bool:
        return self.total_usage > self.budget.total_tokens

    def layer_gauge(self) -> dict:
        """返回各层预算使用率"""
        return {
            "permanent": {"used": self.usage("permanent"), "budget": self.budget.permanent_tokens,
                          "ratio": round(self.usage("permanent") / max(self.budget.permanent_tokens, 1), 2)},
            "working": {"used": self.usage("working"), "budget": self.budget.working_tokens,
                        "ratio": round(self.usage("working") / max(self.budget.working_tokens, 1), 2)},
            "deep": {"used": self.usage("deep"), "budget": self.budget.deep_tokens,
                     "ratio": round(self.usage("deep") / max(self.budget.deep_tokens, 1), 2)},
            "total": {"used": self.total_usage, "budget": self.budget.total_tokens,
                      "ratio": round(self.total_usage / max(self.budget.total_tokens, 1), 2)},
        }

    # ---- 渲染 ----

    def render(self) -> str:
        """组装完整上下文"""
        layers = []

        if self._permanent:
            layers.append("\n".join(self._permanent))

        if self._working:
            layers.append("--- 对话历史 ---")
            layers.extend(self._working[-20:])

        if self._deep:
            layers.append("--- 参考文献 ---")
            if self._deep_metadata:
                layers.append("参考资料列表:")
                for i, text in enumerate(self._deep):
                    layers.append(f"[{i+1}] {text}")
            else:
                layers.extend(self._deep)

        return "\n".join(layers)

    def summary(self) -> str:
        """上下文预算摘要"""
        gauge = self.layer_gauge()
        lines = ["上下文预算使用情况:"]
        for layer, data in gauge.items():
            bar = "█" * int(data["ratio"] * 20) + "░" * (20 - int(data["ratio"] * 20))
            lines.append(f"  {layer:12s} |{bar}| {data['used']:5d}/{data['budget']:5d} ({data['ratio']:.0%})")
        return "\n".join(lines)

    # ---- 快照 ----

    def snapshot(self) -> dict:
        return {
            "permanent_count": len(self._permanent),
            "working_count": len(self._working),
            "deep_count": len(self._deep),
            "usage": self.layer_gauge(),
        }
