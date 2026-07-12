"""异步 LLM 客户端 - 基于 OpenAI SDK

封装兼容 OpenAI 格式的异步调用接口，支持：
- 普通对话补全
- 工具调用 (tool_calls)
- Guided Decoding (JSON Constrained Generation)
- finish_reason 传播
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx
from openai import AsyncOpenAI

from config import cfg
from schema.messages import Message, ToolCall

logger = logging.getLogger(__name__)


class AsyncLLMClient:
    """兼容 OpenAI 格式的异步 LLM 调用客户端"""

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
    ):
        base_url = base_url or cfg.llm.base_url
        api_key = api_key or cfg.llm.api_key
        model = model or cfg.llm.model

        # 允许空 api_key 启动（延迟报错到实际调用时），避免服务启动崩溃
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key if api_key else "sk-placeholder",
            timeout=httpx.Timeout(120.0, connect=10.0),
            max_retries=2,
        )
        self.model = model
        self._api_key_configured = bool(api_key)

    async def chat(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Message:
        """调用 LLM，返回 Message（包含 finish_reason）

        参数:
            messages: 对话历史消息列表
            tools: OpenAI Function Calling 格式的工具描述列表
            response_format: Guided Decoding 格式约束，如 {"type": "json_object"} 强制输出 JSON
            temperature: 生成温度（默认取配置值）
            max_tokens: 最大生成 token 数（默认取配置值）

        返回:
            Message 对象，包含 content、tool_calls、finish_reason
        """
        try:
            # 未配置 API key 时直接返回错误，避免发送无效请求
            if not self._api_key_configured:
                logger.error("LLM API key 未配置，请设置 MDT_LLM_API_KEY 环境变量")
                return Message(
                    role="assistant",
                    content="抱歉，系统未配置 LLM 服务，请设置 MDT_LLM_API_KEY 环境变量后重启。",
                    finish_reason="stop",
                )

            kwargs: dict = {
                "model": self.model,
                # 排除 None 值，避免 API 报错
                "messages": [m.model_dump(exclude_none=True) for m in messages],
                "temperature": temperature if temperature is not None else cfg.llm.temperature,
                "max_tokens": max_tokens if max_tokens is not None else cfg.llm.max_tokens,
            }

            # 工具调用: 转换为 OpenAI tools 格式
            if tools:
                kwargs["tools"] = [{"type": "function", "function": t} for t in tools]

            # Guided Decoding: 强制输出 JSON 格式
            if response_format:
                kwargs["response_format"] = response_format

            resp = await self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0]

            # 提取 finish_reason (stop | tool_calls | length)
            finish_reason = choice.finish_reason

            # 解析工具调用
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id,
                        type=tc.type,
                        function={
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    )
                    for tc in choice.message.tool_calls
                ]

            return Message(
                role=choice.message.role,
                content=choice.message.content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        except Exception as e:
            # 医疗系统不能 Crash：所有 LLM 异常必须有兜底处理
            logger.error(f"LLM 调用失败: {e}")
            return Message(
                role="assistant",
                content="抱歉，系统暂时无法响应，请稍后重试。",
                finish_reason="stop",
            )