"""Run-level 上下文隔离 — 参考 Mem0 Run Memory + LangGraph State

设计理念:
  "一次 Agent 调用 (Run) 有着与 Session 不同的生命周期和隐私边界"
  "Session = 用户会话, Run = 单次 Agent 执行, Turn = 单轮交互"

参考架构:
  - Mem0: User/Session/Agent/Run 四维度记忆隔离
  - LangGraph: State per invocation + checkpoint
  - OpenAI Responses API: 每次请求独立的 input/output token 边界

维度定义:
  ┌──────────┬──────────────────┬──────────────────┐
  │ 维度     │ 生命周期          │ 典型内容          │
  ├──────────┼──────────────────┼──────────────────┤
  │ Session  │ 用户登录到登出    │ 患者画像, 多轮历史 │
  │ Run      │ 单次 process()   │ 工具调用结果, 中间产物│
  │ Turn     │ 单轮 query→回复  │ 本轮对话          │
  │ User     │ 跨 Session       │ 全局偏好, 长期画像 │
  └──────────┴──────────────────┴──────────────────┘

核心能力:
  1. Run 级别隔离: 每次 process() 创建独立的 RunContext
  2. 工具结果缓存: 同一 Run 内避免重复检索
  3. 中间产物管理: 存储专家原始输出、共识草稿等
  4. Run 结束后: 自动清理临时数据, 有价值内容提升到 Session 或 User 级别
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from context.memory_hierarchy import TokenEstimator

logger = logging.getLogger(__name__)


@dataclass
class RunArtifact:
    """Run 级别中间产物"""
    artifact_type: str
    content: str
    artifact_id: str = ""
    timestamp: float = 0.0
    token_count: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()
        if not self.token_count:
            self.token_count = TokenEstimator.estimate(self.content)
        if not self.artifact_id:
            self.artifact_id = str(uuid.uuid4())[:8]


@dataclass
class RunContext:
    """单次 Run 的上下文隔离容器"""
    run_id: str
    session_id: str
    user_id: str
    start_time: float
    end_time: float = 0.0

    artifacts: list[RunArtifact] = field(default_factory=list)
    tool_results_cache: dict[str, str] = field(default_factory=dict)
    retrieval_cache: dict[str, list] = field(default_factory=dict)
    expert_outputs: dict[str, str] = field(default_factory=dict)

    route_path: str = ""
    departments: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    def add_artifact(self, artifact_type: str, content: str, metadata: Optional[dict] = None):
        artifact = RunArtifact(
            artifact_type=artifact_type,
            content=content,
            metadata=metadata or {},
        )
        self.artifacts.append(artifact)
        logger.debug(f"Run artifact [{artifact_type}]: tokens={artifact.token_count}")
        return artifact

    def cache_tool_result(self, tool_name: str, args_key: str, result: str):
        cache_key = f"{tool_name}:{args_key}"
        self.tool_results_cache[cache_key] = result

    def get_cached_tool_result(self, tool_name: str, args_key: str) -> Optional[str]:
        cache_key = f"{tool_name}:{args_key}"
        return self.tool_results_cache.get(cache_key)

    def cache_retrieval(self, query: str, results: list):
        self.retrieval_cache[query[:100]] = results

    def get_cached_retrieval(self, query: str) -> Optional[list]:
        return self.retrieval_cache.get(query[:100])

    def store_expert_output(self, department: str, output: str):
        self.expert_outputs[department] = output

    def get_all_expert_outputs(self) -> dict[str, str]:
        return self.expert_outputs.copy()

    def finish(self):
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    @property
    def total_artifact_tokens(self) -> int:
        return sum(a.token_count for a in self.artifacts)

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "route_path": self.route_path,
            "departments": self.departments,
            "duration_ms": round(self.duration_ms, 2),
            "artifacts_count": len(self.artifacts),
            "total_artifact_tokens": self.total_artifact_tokens,
            "cached_tool_results": len(self.tool_results_cache),
            "expert_outputs": list(self.expert_outputs.keys()),
            "confidence": self.confidence,
        }


class RunMemoryManager:
    """Run-level 记忆管理器

    管理 Run 的生命周期:
      begin_run()   → 创建 RunContext
      add_*()       → 记录中间产物
      finish_run()  → 清理、提升有价值内容到 Session 级别
    """

    def __init__(self):
        self._active_run: Optional[RunContext] = None
        self._completed_runs: list[RunContext] = []
        self._max_completed_runs = 50

    def begin_run(self, session_id: str = "", user_id: str = "") -> RunContext:
        self._active_run = RunContext(
            run_id="",
            session_id=session_id,
            user_id=user_id,
            start_time=time.time(),
        )
        logger.debug(f"Run started: {self._active_run.run_id}")
        return self._active_run

    def finish_run(self, context_manager=None) -> Optional[RunContext]:
        """结束当前 Run，提升有价值内容到 Session 级别

        - 专家结论 → 注入到 ContextManager L2 Long-Term
        - 关键工具结果 → 可选持久化
        """
        if not self._active_run:
            return None

        run = self._active_run
        run.finish()

        if context_manager and run.expert_outputs:
            for dept, output in run.expert_outputs.items():
                context_manager.remember(
                    content=f"[{dept}] {output[:300]}",
                    tier=__import__('context.memory_hierarchy', fromlist=['MemoryTier']).MemoryTier.LONG_TERM,
                    role=__import__('context.memory_hierarchy', fromlist=['MessageRole']).MessageRole.MEMORY,
                    importance=0.7,
                    metadata={"source": f"run:{run.run_id}", "department": dept},
                )

        self._completed_runs.append(run)
        if len(self._completed_runs) > self._max_completed_runs:
            self._completed_runs = self._completed_runs[-self._max_completed_runs:]

        logger.debug(f"Run finished: {run.run_id}, duration={run.duration_ms:.0f}ms")
        self._active_run = None
        return run

    @property
    def active_run(self) -> Optional[RunContext]:
        return self._active_run

    def get_completed_runs(self, n: int = 10) -> list[dict]:
        return [r.summary() for r in self._completed_runs[-n:]]
