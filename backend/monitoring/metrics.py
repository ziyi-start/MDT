"""RAG 管道监控和指标收集

提供:
  - PipelineTimer: 管道阶段计时器
  - MetricsCollector: 请求级指标收集
  - SessionMetrics: 会话聚合指标
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class RequestMetrics:
    """单次请求的管道指标"""
    query: str = ""
    route_path: str = ""  # simple_rag / mdt / safe_fallback
    total_latency_ms: float = 0
    retrieval_ms: float = 0
    rerank_ms: float = 0
    generation_ms: float = 0
    confidence_check_ms: float = 0
    retrieved_count: int = 0
    reranked_count: int = 0
    final_confidence: float = 0
    is_safe_fallback: bool = False
    top_retrieval_scores: list[float] = field(default_factory=list)
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "route_path": self.route_path,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "retrieval_ms": round(self.retrieval_ms, 2),
            "rerank_ms": round(self.rerank_ms, 2),
            "generation_ms": round(self.generation_ms, 2),
            "confidence_check_ms": round(self.confidence_check_ms, 2),
            "retrieved_count": self.retrieved_count,
            "reranked_count": self.reranked_count,
            "final_confidence": round(self.final_confidence, 4),
            "is_safe_fallback": self.is_safe_fallback,
            "top_retrieval_scores": [round(s, 4) for s in self.top_retrieval_scores],
            "success": self.success,
            "error": self.error,
        }


class PipelineTimer:
    """管道阶段计时上下文管理器

    用法:
        timer = PipelineTimer()
        with timer.stage("retrieval"):
            results = retriever.retrieve(query)
        metrics = timer.to_metrics()
    """

    def __init__(self):
        self._start = time.perf_counter()
        self._stages: dict[str, float] = {}
        self._current = None
        self._stage_start = 0.0

    def stage(self, name: str):
        return _StageContext(self, name)

    def _enter_stage(self, name: str):
        self._current = name
        self._stage_start = time.perf_counter()

    def _exit_stage(self):
        if self._current:
            elapsed = (time.perf_counter() - self._stage_start) * 1000
            self._stages[self._current] = elapsed
            self._current = None

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def get(self, stage: str, default: float = 0) -> float:
        return self._stages.get(stage, default)

    def to_metrics(self, **kwargs) -> RequestMetrics:
        return RequestMetrics(
            total_latency_ms=self.total_ms,
            retrieval_ms=self.get("retrieval"),
            rerank_ms=self.get("rerank"),
            generation_ms=self.get("generation"),
            confidence_check_ms=self.get("confidence_check"),
            **kwargs,
        )


class _StageContext:
    def __init__(self, timer: PipelineTimer, name: str):
        self.timer = timer
        self.name = name

    def __enter__(self):
        self.timer._enter_stage(self.name)
        return self

    def __exit__(self, *args):
        self.timer._exit_stage()


@dataclass
class SessionMetrics:
    """会话聚合指标"""
    total_requests: int = 0
    success_count: int = 0
    safe_fallback_count: int = 0
    error_count: int = 0
    routes: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_latency_ms: float = 0
    total_retrieval_ms: float = 0
    total_rerank_ms: float = 0
    total_generation_ms: float = 0
    avg_confidence: float = 0

    def add(self, m: RequestMetrics):
        self.total_requests += 1
        if m.success:
            self.success_count += 1
        else:
            self.error_count += 1
        if m.is_safe_fallback:
            self.safe_fallback_count += 1
        self.routes[m.route_path] += 1
        self.total_latency_ms += m.total_latency_ms
        self.total_retrieval_ms += m.retrieval_ms
        self.total_rerank_ms += m.rerank_ms
        self.total_generation_ms += m.generation_ms
        if m.final_confidence > 0:
            self.avg_confidence = (
                (self.avg_confidence * (self.total_requests - 1) + m.final_confidence)
                / self.total_requests
            )

    def to_dict(self) -> dict:
        n = max(self.total_requests, 1)
        return {
            "total_requests": self.total_requests,
            "success_rate": round(self.success_count / n, 4),
            "safe_fallback_rate": round(self.safe_fallback_count / n, 4),
            "error_rate": round(self.error_count / n, 4),
            "route_distribution": dict(self.routes),
            "avg_total_latency_ms": round(self.total_latency_ms / n, 2),
            "avg_retrieval_ms": round(self.total_retrieval_ms / n, 2),
            "avg_rerank_ms": round(self.total_rerank_ms / n, 2),
            "avg_generation_ms": round(self.total_generation_ms / n, 2),
            "avg_confidence": round(self.avg_confidence, 4),
        }
