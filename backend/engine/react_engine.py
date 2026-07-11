"""手写 ReAct 引擎 - 核心循环

ReAct (Reasoning + Acting) 循环流程:
1. 调用 LLM 获取响应
2. 检查 finish_reason:
   - "stop": LLM 已给出最终回答，退出循环
   - "tool_calls": LLM 请求调用工具，执行工具后将结果注入继续循环
   - "length": 达到最大 token 限制，退出循环
3. 限制最大迭代次数 (max_iterations) 防止死循环
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from schema.messages import Message, ToolCall
from llm.client import AsyncLLMClient
from engine.tool_registry import ToolRegistry
from config import cfg

logger = logging.getLogger(__name__)


class ReactEngine:
    """手写 ReAct 循环引擎

    核心机制: LLM 推理 → 工具调用 → 结果注入 → 继续推理 → 直到 stop 或超限
    """

    def __init__(
        self,
        llm_client: AsyncLLMClient,
        tool_registry: ToolRegistry,
        system_prompt: str,
        max_iterations: int = 0,
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.system_prompt = system_prompt
        self.max_iterations = max_iterations or cfg.react.max_iterations

    async def run(self, query: str, context: str = "") -> str:
        """执行 ReAct 循环

        参数:
            query: 用户问题
            context: 预检索的参考文档（可选），注入为 system 消息

        返回:
            LLM 最终生成的文本回答
        """
        # 初始化消息列表: 系统提示 + 可选参考文档 + 用户问题
        messages: list[Message] = [
            Message(role="system", content=self.system_prompt),
        ]
        if context:
            messages.append(Message(role="system", content=f"参考资料：\n{context}"))
        messages.append(Message(role="user", content=query))

        # 获取工具 schema（供 LLM 决定是否调用）
        tool_schemas = self.tools.get_tool_schemas()

        for i in range(self.max_iterations):
            logger.info(f"ReAct 迭代 {i + 1}/{self.max_iterations}")

            # 调用 LLM
            response = await self.llm.chat(
                messages=messages,
                tools=tool_schemas if tool_schemas else None,
            )
            messages.append(response)

            # 根据 finish_reason 判断下一步
            finish_reason = response.finish_reason

            # "stop" 或 "length": LLM 已给出最终回答（或达到 token 限制），退出循环
            if finish_reason in ("stop", "length") or not response.tool_calls:
                if finish_reason == "length":
                    logger.warning("LLM 达到最大 token 限制，强制输出当前回答")
                logger.info(f"ReAct 完成: finish_reason={finish_reason}")
                return response.content or ""

            # "tool_calls": 解析并执行工具调用
            if response.tool_calls:
                for tc in response.tool_calls:
                    fn_name = tc.function.get("name", "")
                    fn_args = tc.function.get("arguments", "{}")
                    logger.info(f"工具调用: {fn_name}({fn_args})")

                    # 通过 ToolRegistry 执行对应的 Python 异步函数
                    result = await self.tools.execute(fn_name, fn_args)
                    logger.info(f"工具结果: {result[:200]}...")

                    # 将工具执行结果封装为 role="tool" 的消息追加到历史
                    messages.append(Message(
                        role="tool",
                        content=result,
                        tool_call_id=tc.id,
                    ))

        # 达到最大迭代次数：再做一次 LLM 调用让其总结（而非返回 tool 消息）
        logger.warning(f"ReAct 达到最大迭代次数 ({self.max_iterations})，请求 LLM 总结")
        final_response = await self.llm.chat(
            messages=messages + [Message(
                role="user",
                content="请基于以上推理和工具结果，给出最终回答。如果你没有足够信息，请明确说明。",
            )],
        )
        return final_response.content or "抱歉，推理过程超出限制，请简化问题后重试。"