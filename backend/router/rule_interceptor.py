"""规则拦截器 - NER + 正则快速分流

设计文档 3.2 节: 规则拦截（快车道）
- 使用医疗 NER 模型（BERT-CRF）提取实体
- 结合正则匹配冲突关键词
- 若实体数 <= 1 且无冲突关键词 → 直接路由至 Simple RAG（耗时 <50ms）

MVP: 优先调用外部 NER 服务，不可用时退化为正则匹配
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

from schema.models import RouteDecision
from config import cfg

logger = logging.getLogger(__name__)

# 医疗实体关键词正则（MVP: 正则替代 BERT-CRF NER）
MEDICAL_ENTITY_PATTERNS = [
    # 疾病
    r"(高血压|糖尿病|胃溃疡|痛风|冠心病|肾病|肺炎|肝炎|哮喘|心衰|房颤|脑梗|甲亢|甲减|贫血|骨折|消化道出血|CKD|肾功能不全)",
    # 药物
    r"(布洛芬|阿司匹林|氯吡格雷|秋水仙碱|对乙酰氨基酚|甲氨蝶呤|别嘌醇|二甲双胍|硝苯地平|阿莫西林|头孢|奥美拉唑|泼尼松|塞来昔布)",
    # 症状
    r"(发热|头痛|腹痛|咳嗽|恶心|呕吐|腹泻|便秘|失眠|水肿|胸闷|心悸|头晕|皮疹|关节痛)",
    # 过敏
    r"(青霉素过敏|磺胺过敏|花粉过敏|食物过敏|药物过敏)",
]

# 冲突/复杂度关键词（正则匹配）
CONFLICT_KEYWORDS = [
    "同时", "联用", "能不能一起", "相互作用", "冲突", "禁忌",
    "并发症", "合并", "既有.*又有", "多个", "综合",
]

# 科室映射表: 实体 → 对应科室
DEPARTMENT_MAP = {
    "高血压": "心内科", "冠心病": "心内科", "心衰": "心内科",
    "房颤": "心内科", "心悸": "心内科", "胸闷": "心内科",
    "胃溃疡": "消化科", "腹痛": "消化科", "恶心": "消化科",
    "呕吐": "消化科", "腹泻": "消化科", "消化道出血": "消化科",
    "痛风": "风湿科", "关节痛": "风湿科",
    "糖尿病": "内分泌科", "甲亢": "内分泌科", "甲减": "内分泌科",
    "肾病": "肾内科", "水肿": "肾内科", "CKD": "肾内科", "肾功能不全": "肾内科",
    "肺炎": "呼吸科", "咳嗽": "呼吸科", "哮喘": "呼吸科",
    "脑梗": "神经内科", "头痛": "神经内科", "头晕": "神经内科",
}


class RuleInterceptor:
    """规则拦截器（快车道）

    优先调用外部 NER 服务提取医疗实体，不可用时退化为正则匹配。
    根据实体数量和冲突关键词判断路由路径。
    """

    def __init__(self, ner_service_url: str = ""):
        """
        参数:
            ner_service_url: 外部 NER 服务地址（BERT-CRF 模型部署的 HTTP API）
        """
        self.ner_service_url = ner_service_url or cfg.services.ner_url

    def intercept(self, query: str) -> RouteDecision | None:
        """规则拦截，返回 None 表示灰度问题需走 LLM 路由

        路由逻辑:
        - 实体数 <= 1 且无冲突关键词 → Simple RAG（快车道）
        - 实体数 > 2 或有冲突关键词 → MDT（复杂问题）
        - 实体数 == 2 且无冲突关键词 → None（灰度问题，交 LLM 路由）
        """
        # 优先使用外部 NER 服务
        entities = self._ner_extract(query)
        has_conflict = self._has_conflict_keywords(query)

        logger.info(f"规则拦截: entities={entities}, has_conflict={has_conflict}")

        # 快车道: 简单问题直接走 Simple RAG
        if len(entities) <= 1 and not has_conflict:
            logger.info("规则拦截命中: 简单问题 → Simple RAG")
            return RouteDecision(route_path="simple_rag", departments=[])

        # 复杂问题直接走 MDT
        if len(entities) > 2 or has_conflict:
            departments = list(set(
                DEPARTMENT_MAP.get(e, "全科") for e in entities if e in DEPARTMENT_MAP
            ))
            if not departments:
                departments = ["全科"]
            logger.info(f"规则拦截命中: 复杂问题 → MDT, departments={departments}")
            return RouteDecision(route_path="mdt", departments=departments)

        # 灰度问题: 实体数 == 2 且无冲突关键词，交由 LLM 路由判断
        logger.info("规则拦截未命中: 灰度问题 → LLM 路由")
        return None

    def _ner_extract(self, query: str) -> list[str]:
        """提取医疗实体

        优先调用外部 NER 服务（BERT-CRF），不可用时退化为正则匹配。
        设计文档要求: 必须调用外部医疗 NER 服务
        """
        # 尝试调用外部 NER 服务
        if self.ner_service_url:
            try:
                entities = self._call_ner_service(query)
                if entities:
                    return entities
            except Exception as e:
                logger.warning(f"NER 服务调用失败，退化为正则匹配: {e}")

        # 退化: 正则匹配实体
        return self._regex_extract(query)

    def _call_ner_service(self, query: str) -> list[str]:
        """调用外部 NER 服务（BERT-CRF 模型）

        期望 API 格式:
        POST /ner
        Body: {"text": "患者有高血压..."}
        Response: {"entities": ["高血压", "胃溃疡", ...]}
        """
        resp = httpx.post(
            self.ner_service_url,
            json={"text": query},
            timeout=cfg.services.ner_timeout,
        )
        data = resp.json()
        entities = data.get("entities", [])
        logger.info(f"NER 服务提取: {entities}")
        return entities

    def _regex_extract(self, query: str) -> list[str]:
        """正则匹配提取医疗实体（NER 服务不可用时的退化方案）"""
        import re
        entities = []
        for pattern in MEDICAL_ENTITY_PATTERNS:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        return list(set(entities))

    def _has_conflict_keywords(self, query: str) -> bool:
        """检查是否包含冲突/复杂度关键词"""
        import re
        for kw in CONFLICT_KEYWORDS:
            if re.search(kw, query):
                return True
        return False