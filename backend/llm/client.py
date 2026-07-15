"""异步 LLM 客户端 - 基于 OpenAI SDK

封装兼容 OpenAI 格式的异步调用接口，支持：
- 普通对话补全
- 流式输出 (SSE streaming)
- 工具调用 (tool_calls)
- Guided Decoding (JSON Constrained Generation)
- finish_reason 传播
"""
from __future__ import annotations

import logging
from typing import Optional, AsyncIterator

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
        """调用 LLM，返回 Message（包含 finish_reason）"""
        try:
            if not self._api_key_configured:
                logger.error("LLM API key 未配置，请设置 MDT_LLM_API_KEY 环境变量")
                return Message(
                    role="assistant",
                    content="抱歉，系统未配置 LLM 服务。",
                    finish_reason="stop",
                )

            kwargs: dict = {
                "model": self.model,
                "messages": [m.model_dump(exclude_none=True) for m in messages],
                "temperature": temperature if temperature is not None else cfg.llm.temperature,
                "max_tokens": max_tokens if max_tokens is not None else cfg.llm.max_tokens,
            }
            if tools:
                kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            if response_format:
                kwargs["response_format"] = response_format

            resp = await self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0]
            finish_reason = choice.finish_reason

            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    ToolCall(
                        id=tc.id, type=tc.type,
                        function={"name": tc.function.name, "arguments": tc.function.arguments},
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
            logger.error(f"LLM 调用失败: {e}")
            return Message(
                role="assistant",
                content="抱歉，系统暂时无法响应，请稍后重试。",
                finish_reason="stop",
            )

    async def chat_stream(
        self,
        messages: list[Message],
        tools: Optional[list[dict]] = None,
        response_format: Optional[dict] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[dict]:
        """流式调用 LLM，逐 token yield

        每块 yield 格式: {"type": "content"|"done"|"error", "text": str, "finish_reason": str}
        """
        try:
            if not self._api_key_configured:
                yield {"type": "error", "text": "LLM API key 未配置"}
                return

            kwargs: dict = {
                "model": self.model,
                "stream": True,
                "messages": [m.model_dump(exclude_none=True) for m in messages],
                "temperature": temperature if temperature is not None else cfg.llm.temperature,
                "max_tokens": max_tokens if max_tokens is not None else cfg.llm.max_tokens,
            }
            if tools:
                kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
            if response_format:
                kwargs["response_format"] = response_format

            stream = await self.client.chat.completions.create(**kwargs)
            collected = []
            finish_reason = "stop"
            tool_calls_acc: dict[int, dict] = {}
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if not delta:
                    continue
                if delta.content:
                    collected.append(delta.content)
                    yield {"type": "content", "text": delta.content}
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": tc.id or "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_calls_acc[idx]["id"] = tc.id
                        if tc.function and tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function and tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments
                if chunk.choices and chunk.choices[0].finish_reason:
                    finish_reason = chunk.choices[0].finish_reason

            yield {
                "type": "done",
                "text": "".join(collected),
                "finish_reason": finish_reason,
                "tool_calls": [
                    {"id": v["id"], "type": "function", "function": {"name": v["name"], "arguments": v["arguments"]}}
                    for v in tool_calls_acc.values() if v["name"]
                ] if tool_calls_acc else None,
            }

        except Exception as e:
            logger.error(f"LLM stream 失败: {e}")
            yield {"type": "error", "text": str(e)}
