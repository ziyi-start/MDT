"""OpenAI 兼容消息格式

封装与 OpenAI API 交互所需的消息结构，
包括普通消息、工具调用请求和工具执行结果。
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """工具调用请求 - LLM 返回的 function call"""
    id: str = Field(description="工具调用唯一标识，用于关联 tool 消息")
    type: str = Field(default="function", description="调用类型，固定为 function")
    function: dict = Field(description="函数调用信息: {'name': '...', 'arguments': '...'}")


class Message(BaseModel):
    """聊天消息 - 兼容 OpenAI Chat Completion 消息格式

    支持四种角色:
    - system: 系统提示词
    - user: 用户输入
    - assistant: LLM 响应（可能包含 tool_calls）
    - tool: 工具执行结果（需携带 tool_call_id）
    """
    role: str = Field(description="消息角色: system | user | assistant | tool")
    content: Optional[str] = Field(default=None, description="消息文本内容")
    tool_calls: Optional[list[ToolCall]] = Field(default=None, description="LLM 发起的工具调用列表")
    tool_call_id: Optional[str] = Field(default=None, description="工具结果对应的 tool_call ID")
    finish_reason: Optional[str] = Field(default=None, description="结束原因: stop | tool_calls | length")