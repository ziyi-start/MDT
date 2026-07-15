"""Agent 自主记忆工具 — 参考 Letta Memory Blocks + ChatGPT memory tool

设计理念:
  "Agent 不应该被动地接受记忆，而应该像人类一样主动决定'这个信息很重要，我要记住'"
  "Hot-path memory update: Agent 在推理过程中通过 tool calling 自主管理记忆"

参考架构:
  - Letta memory blocks: Agent 可通过 memory_replace/memory_insert 主动编辑上下文
  - ChatGPT memory tool: Agent 通过 bio tool 决定记住用户的偏好
  - CoALA decision procedure: Agent 的 propose→evaluate→select 循环中自主存储

三种记忆工具:
  1. remember(fact, importance, tier) — 主动记住一个事实
  2. forget(memory_id)              — 主动遗忘（标记 obsolete）
  3. recall(query, n)              — 主动检索记忆（显式回忆）

设计要点:
  - 这些工具注册在 global_tool_registry 中，Agent 在 ReAct 循环中可以调用
  - 与 ContextManager 深度集成，remember 直接写入 L2 Long-Term Memory
  - 自动去重：重复的 fact 不会重复存储
  - importance 由 Agent 自主打分，不依赖外部规则
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_global_context_manager = None


def set_context_manager(ctx_manager):
    """设置全局上下文管理器引用（由 orchestrator 调用）"""
    global _global_context_manager
    _global_context_manager = ctx_manager


def get_context_manager():
    return _global_context_manager


async def remember(fact: str, importance: float = 0.8, tier: str = "long_term") -> str:
    """Agent 自主记忆工具 — 主动记住一个关键事实

    Agent 在 ReAct 循环中调用此工具，将推理过程中发现的关键信息
    存入长期记忆，后续对话自动穿透。

    参数:
        fact: 要记住的事实（简洁描述）
        importance: 重要性 0.0-1.0 (0.5=中等, 0.8=重要, 1.0=关键)
        tier: 记忆层级 ("short_term" | "long_term")

    返回:
        确认消息

    医学场景示例:
        - "该患者肌酐清除率 CrCl=30mL/min，CKD 3期"
        - "氯吡格雷+NSAIDs联用显著增加消化道出血风险"
        - "患者对青霉素过敏"
    """
    ctx = get_context_manager()
    if ctx is None:
        return "[Memory Error] 上下文管理器未初始化"

    from context.memory_hierarchy import MemoryTier, MessageRole
    mem_tier = MemoryTier.LONG_TERM if tier == "long_term" else MemoryTier.SHORT_TERM

    try:
        ctx.remember(
            content=f"[Agent Self-Memory] {fact}",
            tier=mem_tier,
            role=MessageRole.MEMORY,
            importance=importance,
            metadata={"source": "agent_self", "timestamp": time.time()},
        )
        return f"[Memory OK] 已记住: {fact[:100]}"
    except Exception as e:
        logger.error(f"Agent remember failed: {e}")
        return f"[Memory Error] {e}"


async def forget(memory_id: str) -> str:
    """Agent 自主遗忘工具 — 标记过时记忆

    当 Agent 发现之前的记忆已经不再适用（如患者病情变化、用药方案调整），
    可以主动调用 forget 标记记忆为过时。

    参数:
        memory_id: 要遗忘的记忆标识符 (entry_id 或内容关键词)

    返回:
        确认消息
    """
    ctx = get_context_manager()
    if ctx is None:
        return "[Memory Error] 上下文管理器未初始化"

    from context.memory_hierarchy import MemoryTier
    try:
        for tier in [MemoryTier.LONG_TERM, MemoryTier.SHORT_TERM]:
            entries = ctx.hierarchy._get_tier_list(tier)
            for entry in entries:
                if memory_id in entry.entry_id or memory_id in entry.content:
                    entry.importance = 0.0
                    entry.metadata["obsolete"] = True
                    entry.metadata["obsolete_reason"] = "agent_forgot"
                    entry.metadata["obsolete_time"] = time.time()
        return f"[Memory OK] 已遗忘: {memory_id[:100]}"
    except Exception as e:
        logger.error(f"Agent forget failed: {e}")
        return f"[Memory Error] {e}"


async def recall(query: str, n: int = 3) -> str:
    """Agent 自主回忆工具 — 显式检索记忆

    Agent 在需要时显式调用此工具检索之前存储的记忆。
    与 ContextManager 的被动记忆注入不同，此工具让 Agent 主动搜索。

    参数:
        query: 检索查询
        n: 返回条数

    返回:
        格式化的记忆列表
    """
    ctx = get_context_manager()
    if ctx is None:
        return "[Memory Error] 上下文管理器未初始化"

    from context.memory_hierarchy import MemoryTier
    try:
        found = []
        for tier in [MemoryTier.LONG_TERM, MemoryTier.SHORT_TERM]:
            entries = ctx.hierarchy._get_tier_list(tier)
            for entry in entries:
                if entry.importance < 0.1:
                    continue
                score = ctx.hierarchy._score_entry(entry, query)
                if score > 0.3:
                    found.append((score, entry))

        found.sort(key=lambda x: x[0], reverse=True)
        if not found:
            return "[Memory] 未找到相关记忆"

        results = []
        for score, entry in found[:n]:
            results.append(f"[score={score:.2f}] {entry.content[:200]}")
        return "\n".join(results)
    except Exception as e:
        logger.error(f"Agent recall failed: {e}")
        return f"[Memory Error] {e}"


MEMORY_TOOL_DEFINITIONS = [
    {
        "name": "remember",
        "description": "记住一个关键事实到长期记忆（Agent 在发现重要信息时主动调用）。医学场景示例：患者禁忌症、关键检查结果、用药注意事项。",
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "要记住的关键事实（简洁准确）"},
                "importance": {"type": "number", "description": "重要性 0.0-1.0", "default": 0.8},
                "tier": {"type": "string", "enum": ["short_term", "long_term"], "default": "long_term"},
            },
            "required": ["fact"],
        },
    },
    {
        "name": "forget",
        "description": "遗忘过时的记忆（患者病情变化或用药方案调整时使用）",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "要遗忘的记忆标识符"},
            },
            "required": ["memory_id"],
        },
    },
    {
        "name": "recall",
        "description": "主动检索之前存储的记忆",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询"},
                "n": {"type": "integer", "description": "返回条数", "default": 3},
            },
            "required": ["query"],
        },
    },
]
