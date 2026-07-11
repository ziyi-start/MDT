"""工具注册器 - 装饰器模式自动生成 OpenAI Tool JSON Schema

核心机制：
- 提供装饰器将 Python 异步函数注册为 LLM 可调用的工具
- 自动从函数签名推断参数 JSON Schema
- 管理工具描述与执行函数的映射关系
"""
from __future__ import annotations

import inspect
import json
import logging
from typing import Callable, Any, get_type_hints

logger = logging.getLogger(__name__)


class ToolRegistry:
    """工具注册与执行管理器

    每个注册的工具包含:
    - schema: OpenAI Function Calling 格式的 JSON Schema
    - fn: 实际执行的 Python 异步函数
    """

    def __init__(self):
        self._tools: dict[str, dict] = {}  # name -> {"schema": ..., "fn": ...}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict | None = None,
    ):
        """装饰器：将异步函数注册为 LLM 可调用的工具

        参数:
            name: 工具名称，LLM 通过此名称调用
            description: 工具功能描述，LLM 据此决定是否调用
            parameters: 参数 JSON Schema，不提供则从函数签名自动推断
        """
        def decorator(fn: Callable):
            # 未提供参数 schema 时，从函数签名自动推断
            if parameters is None:
                params = _infer_parameters(fn)
            else:
                params = parameters

            self._tools[name] = {
                "schema": {
                    "name": name,
                    "description": description,
                    "parameters": params,
                },
                "fn": fn,
            }
            logger.info(f"工具注册: {name}")
            return fn
        return decorator

    def get_tool_schemas(self) -> list[dict]:
        """返回所有工具的 OpenAI Function JSON Schema，供 LLM 调用使用"""
        return [t["schema"] for t in self._tools.values()]

    async def execute(self, tool_name: str, arguments: str) -> str:
        """执行工具调用

        参数:
            tool_name: 工具名称
            arguments: JSON 格式的参数字符串

        返回:
            工具执行结果（JSON 字符串）
        """
        if tool_name not in self._tools:
            logger.warning(f"未知工具: {tool_name}")
            return json.dumps({"error": f"未知工具: {tool_name}"}, ensure_ascii=False)

        tool = self._tools[tool_name]
        try:
            args = json.loads(arguments)
            result = await tool["fn"](**args)
            # 确保返回字符串
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"工具参数 JSON 解析失败 {tool_name}: {e}")
            return json.dumps({"error": f"参数格式错误: {e}"}, ensure_ascii=False)
        except Exception as e:
            # 医疗系统不能 Crash：工具执行异常必须有兜底
            logger.error(f"工具执行失败 {tool_name}: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)


def _infer_parameters(fn: Callable) -> dict:
    """从函数签名自动推断 JSON Schema parameters

    根据类型注解映射 Python 类型到 JSON Schema 类型:
    - int → integer
    - float → number
    - bool → boolean
    - 其他 → string（默认）
    """
    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    properties: dict = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        # 根据类型注解推断 JSON 类型
        prop: dict = {"type": "string"}
        if param.annotation != inspect.Parameter.empty:
            ann = param.annotation
            if ann is int:
                prop["type"] = "integer"
            elif ann is float:
                prop["type"] = "number"
            elif ann is bool:
                prop["type"] = "boolean"

        # 无默认值的参数为必填
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
        properties[param_name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


# 全局工具注册器 - 所有工具默认注册到此实例
global_tool_registry = ToolRegistry()