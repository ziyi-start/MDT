"""分布式追踪系统 - TraceID + Span 传播

设计理念（来自 Agent Harness Observability）:
  "通过 trace 可视化执行路径和状态转换"
  "帮助开发者理解 Agent 在每个步骤做了什么、为什么做、花了多少时间"

功能:
  - 每个请求分配唯一 TraceID
  - 每个阶段记录 Span (名称、开始/结束时间、状态、元数据)
  - Span 父子关系 (Parent-Child)
  - 执行图生成 (DAG of operations)
  - 兼容 OpenTelemetry 语义
"""
from __future__ import annotations

import time
import uuid
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# Span 定义
# ============================================================

@dataclass
class Span:
    name: str
    span_id: str
    parent_id: Optional[str] = None
    trace_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "ok"
    metadata: dict = field(default_factory=dict)
    children: list[Span] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "trace_id": self.trace_id,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "metadata": self.metadata,
            "children": [c.to_dict() for c in self.children],
        }


# ============================================================
# Trace Context
# ============================================================

class TraceContext:
    """追踪上下文 - 管理单个请求的完整追踪信息

    用法:
        trace = TraceContext("query_processing")
        with trace.span("routing"):
            decision = await route(query)
        with trace.span("retrieval", parent="routing"):
            docs = await retrieve(query)
        print(trace.to_graph())
    """

    def __init__(self, name: str = "", trace_id: str = ""):
        self.trace_id = trace_id or uuid.uuid4().hex[:16]
        self.name = name
        self._start_time = time.time()
        self._spans: dict[str, Span] = {}
        self._roots: list[Span] = []
        self._stack: list[Span] = []

    @property
    def total_ms(self) -> float:
        return (time.time() - self._start_time) * 1000

    def generate_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def begin_span(
        self,
        name: str,
        parent_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Span:
        """开始一个新的 Span

        参数:
            name: Span 名称
            parent_id: 父 Span ID (None 则自动挂到栈顶)
            metadata: 额外元数据
        """
        span_id = self.generate_id()

        if parent_id is None and self._stack:
            parent_id = self._stack[-1].span_id

        span = Span(
            name=name,
            span_id=span_id,
            parent_id=parent_id,
            trace_id=self.trace_id,
            start_time=time.time(),
            metadata=metadata or {},
        )

        self._spans[span_id] = span

        if parent_id and parent_id in self._spans:
            self._spans[parent_id].children.append(span)
        else:
            self._roots.append(span)

        self._stack.append(span)
        return span

    def end_span(self, span_id: str, status: str = "ok", metadata: Optional[dict] = None):
        """结束一个 Span"""
        span = self._spans.get(span_id)
        if not span:
            logger.warning(f"结束不存在的 Span: {span_id}")
            return

        span.end_time = time.time()
        span.status = status
        if metadata:
            span.metadata.update(metadata)

        if self._stack and self._stack[-1].span_id == span_id:
            self._stack.pop()

    def span(self, name: str, metadata: Optional[dict] = None):
        """上下文管理器: Span 创建与自动结束"""
        return _SpanContext(self, name, metadata)

    def add_event(self, name: str, metadata: Optional[dict] = None):
        """在当前栈顶 Span 上添加事件"""
        if self._stack:
            current = self._stack[-1]
            if "events" not in current.metadata:
                current.metadata["events"] = []
            current.metadata["events"].append({
                "name": name,
                "time": time.time(),
                "metadata": metadata or {},
            })

    def get_span(self, span_id: str) -> Optional[Span]:
        return self._spans.get(span_id)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "total_ms": round(self.total_ms, 2),
            "root_spans": [s.to_dict() for s in self._roots],
        }

    def to_graph(self) -> str:
        """生成可读的执行图文本"""
        lines = [f"Trace: {self.trace_id} ({self.name})", f"Total: {self.total_ms:.0f}ms", ""]

        def _render(spans: list[Span], indent: int = 0):
            prefix = "  " * indent
            for span in spans:
                status_mark = "✓" if span.status == "ok" else "✗"
                meta_str = ""
                if span.metadata:
                    meta_items = [f"{k}={v}" for k, v in span.metadata.items() if k != "events"]
                    if meta_items:
                        meta_str = f" [{', '.join(meta_items)}]"
                lines.append(f"{prefix}{status_mark} {span.name} ({span.duration_ms:.0f}ms){meta_str}")
                if span.children:
                    _render(span.children, indent + 1)

        _render(self._roots)
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ============================================================
# Span 上下文管理器
# ============================================================

class _SpanContext:
    def __init__(self, trace: TraceContext, name: str, metadata: Optional[dict] = None):
        self.trace = trace
        self.name = name
        self.metadata = metadata
        self.span: Optional[Span] = None

    def __enter__(self):
        self.span = self.trace.begin_span(self.name, metadata=self.metadata)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "error" if exc_type else "ok"
        self.trace.end_span(self.span.span_id, status=status)


# ============================================================
# Trace 管理器 (全局单例)
# ============================================================

class TraceManager:
    """追踪管理器 - 管理所有活跃的 Trace"""

    def __init__(self):
        self._active_traces: dict[str, TraceContext] = {}
        self._completed_traces: list[TraceContext] = []
        self._max_completed = 1000

    def begin_trace(self, name: str = "", trace_id: str = "") -> TraceContext:
        """创建新的追踪"""
        trace = TraceContext(name=name, trace_id=trace_id)
        self._active_traces[trace.trace_id] = trace
        return trace

    def end_trace(self, trace_id: str):
        """完成一个追踪并归档"""
        trace = self._active_traces.pop(trace_id, None)
        if trace:
            self._completed_traces.append(trace)
            if len(self._completed_traces) > self._max_completed:
                self._completed_traces = self._completed_traces[-self._max_completed:]

    def get_active(self, trace_id: str) -> Optional[TraceContext]:
        return self._active_traces.get(trace_id)

    def get_recent(self, n: int = 10) -> list[dict]:
        return [t.to_dict() for t in self._completed_traces[-n:]]

    @property
    def stats(self) -> dict:
        return {
            "active_traces": len(self._active_traces),
            "completed_traces": len(self._completed_traces),
        }


# 全局追踪管理器
trace_manager = TraceManager()
