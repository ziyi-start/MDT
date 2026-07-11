"""LLM 结构化意图路由

设计文档 3.2 节: LLM 意图解析（慢车道）
- 对灰度问题调用 LLM
- 开启 Guided Decoding (JSON Constrained Generation)
- 强制输出路由 JSON（包含 route_path 和 departments）
"""
from __future__ import annotations

import json
import logging

from schema.models import RouteDecision
from llm.client import AsyncLLMClient
from llm.prompt_templates import ROUTER_SYSTEM_PROMPT, ROUTER_USER_TEMPLATE
from schema.messages import Message
from config import cfg

logger = logging.getLogger(__name__)


class LLMRouter:
    """LLM 慢车道路由

    使用 Guided Decoding 强制 LLM 输出结构化路由 JSON，
    避免 LLM 自由文本输出导致解析失败。
    """

    def __init__(self, llm: AsyncLLMClient):
        self.llm = llm

    async def route(self, query: str, profile_summary: str = "") -> RouteDecision:
        """调用 LLM 进行意图路由

        参数:
            query: 用户问题
            profile_summary: 患者画像摘要

        返回:
            RouteDecision（route_path + departments）
        """
        user_msg = ROUTER_USER_TEMPLATE.format(query=query, profile=profile_summary or "无")

        # 开启 Guided Decoding 强制输出 JSON
        resp = await self.llm.chat(
            messages=[
                Message(role="system", content=ROUTER_SYSTEM_PROMPT),
                Message(role="user", content=user_msg),
            ],
            response_format={"type": "json_object"},  # Guided Decoding
            temperature=cfg.llm.temperatures["router"],
        )

        # 已知的有效路由路径和科室
        VALID_ROUTE_PATHS = {"simple_rag", "mdt"}
        KNOWN_DEPARTMENTS = {"心内科", "风湿科", "消化科", "内分泌科", "神经内科", "肾内科", "呼吸科", "全科"}

        try:
            data = json.loads(resp.content or "{}")
            # 校验 route_path
            route_path = data.get("route_path", "simple_rag")
            if route_path not in VALID_ROUTE_PATHS:
                logger.warning(f"LLM 路由返回无效路径 '{route_path}'，降级为 simple_rag")
                route_path = "simple_rag"

            # 校验 departments
            departments = data.get("departments", [])
            if not isinstance(departments, list):
                departments = []
            departments = [d for d in departments if isinstance(d, str) and d in KNOWN_DEPARTMENTS]

            decision = RouteDecision(
                route_path=route_path,
                departments=departments or ["全科"],
            )
        except (json.JSONDecodeError, Exception) as e:
            # JSON 解析失败: 降级为 simple_rag（安全降级策略）
            logger.warning(f"LLM 路由解析失败: {e}, 降级为 simple_rag")
            decision = RouteDecision(route_path="simple_rag", departments=[])

        logger.info(f"LLM 路由决策: path={decision.route_path}, departments={decision.departments}")
        return decision