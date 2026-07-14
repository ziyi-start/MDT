"""工具调用安全守卫 - Pre/Post 验证 + 限流 + 成本追踪

设计理念（来自 Agent Harness Safety & Governance）:
  "工具调用必须经过验证、限流和审计"
  "权限、身份、审计闭环是 Agent 进入生产的前提"

功能:
  - Pre-hook: 工具调用前参数验证、权限检查、限流
  - Post-hook: 结果验证、异常捕获、审计日志
  - Rate Limiter: 令牌桶算法防止工具被过度调用
  - Cost Tracker: 每次工具调用的成本估算和追踪
  - Audit Logger: 完整的工具调用审计轨迹
"""
from __future__ import annotations

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable
from collections import defaultdict

from config import cfg

logger = logging.getLogger(__name__)


# ============================================================
# 速率限制器
# ============================================================

class RateLimiter:
    """令牌桶速率限制器 - 防止工具被过度调用

    每个工具独立计数:
    - max_calls: 时间窗口内最大调用次数
    - window_seconds: 时间窗口（秒）
    """

    def __init__(self, max_calls: int = 20, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def check(self, tool_name: str) -> bool:
        """检查工具调用是否允许

        返回: True 允许调用, False 被限流
        """
        now = time.time()
        window_start = now - self.window_seconds

        bucket = self._buckets[tool_name]
        bucket[:] = [t for t in bucket if t > window_start]

        if len(bucket) >= self.max_calls:
            logger.warning(f"工具 {tool_name} 被限流: {len(bucket)}/{self.max_calls} 调用 in {self.window_seconds}s")
            return False

        bucket.append(now)
        return True

    def remaining(self, tool_name: str) -> int:
        """剩余可用调用次数"""
        now = time.time()
        window_start = now - self.window_seconds
        bucket = [t for t in self._buckets.get(tool_name, []) if t > window_start]
        return max(0, self.max_calls - len(bucket))

    def reset(self, tool_name: str = ""):
        """重置限流器"""
        if tool_name:
            self._buckets[tool_name] = []
        else:
            self._buckets.clear()


# ============================================================
# 成本追踪
# ============================================================

@dataclass
class ToolCallCost:
    tool_name: str
    timestamp: float
    duration_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    success: bool = True
    error: str = ""


class CostTracker:
    """工具调用成本追踪

    估算标准 (DeepSeek-chat 定价):
      - 输入: $0.27 / 1M tokens
      - 输出: $1.10 / 1M tokens
    """

    INPUT_COST_PER_TOKEN = 0.27 / 1_000_000
    OUTPUT_COST_PER_TOKEN = 1.10 / 1_000_000

    def __init__(self):
        self._calls: list[ToolCallCost] = []
        self._session_start = time.time()

    def record(
        self,
        tool_name: str,
        duration_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        success: bool = True,
        error: str = "",
    ):
        cost = (input_tokens * self.INPUT_COST_PER_TOKEN +
                output_tokens * self.OUTPUT_COST_PER_TOKEN)
        call = ToolCallCost(
            tool_name=tool_name,
            timestamp=time.time(),
            duration_ms=duration_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(cost, 6),
            success=success,
            error=error,
        )
        self._calls.append(call)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(c.estimated_cost_usd for c in self._calls), 6)

    @property
    def total_calls(self) -> int:
        return len(self._calls)

    @property
    def avg_duration_ms(self) -> float:
        if not self._calls:
            return 0.0
        return sum(c.duration_ms for c in self._calls) / len(self._calls)

    def by_tool(self) -> dict:
        stats: dict = {}
        for c in self._calls:
            if c.tool_name not in stats:
                stats[c.tool_name] = {"calls": 0, "total_cost": 0.0, "errors": 0}
            stats[c.tool_name]["calls"] += 1
            stats[c.tool_name]["total_cost"] += c.estimated_cost_usd
            if not c.success:
                stats[c.tool_name]["errors"] += 1
        for v in stats.values():
            v["total_cost"] = round(v["total_cost"], 6)
        return stats

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "total_cost_usd": self.total_cost_usd,
            "avg_duration_ms": round(self.avg_duration_ms, 2),
            "by_tool": self.by_tool(),
            "session_duration_sec": round(time.time() - self._session_start, 1),
        }


# ============================================================
# 安全守卫
# ============================================================

class SafetyGuard:
    """工具调用安全守卫 - 三层防护

    1. Pre-hook: 参数验证 + 限流 + 权限检查
    2. Execution: 增强执行 (耗时监控)
    3. Post-hook: 结果验证 + 审计 + 成本追踪
    """

    def __init__(
        self,
        rate_limiter: Optional[RateLimiter] = None,
        cost_tracker: Optional[CostTracker] = None,
    ):
        self.rate_limiter = rate_limiter or RateLimiter()
        self.cost_tracker = cost_tracker or CostTracker()
        self._pre_hooks: list[Callable[[str, dict], Awaitable[Optional[str]]]] = []
        self._post_hooks: list[Callable[[str, dict, str, bool], Awaitable[None]]] = []

    def add_pre_hook(self, hook: Callable[[str, dict], Awaitable[Optional[str]]]):
        """添加前置验证钩子

        钩子签名: async def hook(tool_name: str, args: dict) -> Optional[str]
        返回 None 表示通过, 返回字符串表示拒绝原因
        """
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook: Callable[[str, dict, str, bool], Awaitable[None]]):
        """添加后置处理钩子

        钩子签名: async def hook(tool_name: str, args: dict, result: str, success: bool)
        """
        self._post_hooks.append(hook)

    async def guard_execute(
        self,
        tool_name: str,
        arguments: str,
        execute_fn: Callable[[str, str], Awaitable[str]],
    ) -> tuple[str, bool]:
        """安全执行工具调用

        返回: (result, success)
        """
        start = time.time()

        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except json.JSONDecodeError:
            return json.dumps({"error": "参数格式错误"}, ensure_ascii=False), False

        # ---- Pre-hook: 限流 ----
        if not self.rate_limiter.check(tool_name):
            return json.dumps(
                {"error": f"工具 {tool_name} 调用过频，请稍后重试"},
                ensure_ascii=False,
            ), False

        # ---- Pre-hook: 验证 ----
        pre_args = args if isinstance(args, dict) else {}
        for hook in self._pre_hooks:
            rejection = await hook(tool_name, pre_args)
            if rejection is not None:
                duration_ms = (time.time() - start) * 1000
                self.cost_tracker.record(tool_name, duration_ms, success=False, error=rejection)
                return json.dumps({"error": rejection}, ensure_ascii=False), False

        # ---- 执行 ----
        try:
            result = await execute_fn(tool_name, arguments)
            duration_ms = (time.time() - start) * 1000
            self.cost_tracker.record(tool_name, duration_ms, success=True)
            success = True
        except Exception as e:
            duration_ms = (time.time() - start) * 1000
            result = json.dumps({"error": str(e)}, ensure_ascii=False)
            self.cost_tracker.record(tool_name, duration_ms, success=False, error=str(e))
            success = False

        # ---- Post-hook ----
        for hook in self._post_hooks:
            try:
                await hook(tool_name, pre_args, result, success)
            except Exception as e:
                logger.warning(f"Post-hook 失败: {e}")

        return result, success


# ============================================================
# 内置验证钩子
# ============================================================

async def validate_tool_args(tool_name: str, args: dict) -> Optional[str]:
    """内置验证: 检查关键参数不为空"""
    if tool_name == "literature_search":
        query = args.get("query", "")
        if not query or len(query.strip()) < 2:
            return "检索查询不能为空或过短"
    elif tool_name == "check_drug_interaction":
        drugs = args.get("drugs", "")
        if not drugs:
            return "药物名称不能为空"
    return None
