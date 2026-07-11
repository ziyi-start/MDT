"""药物冲突查询工具 - Agent 可调用的药物相互作用检查

内置简易药物冲突知识库（MVP），生产环境应接入专业药物数据库 API。
设计文档 3.3 节: 专家 Agent 可主动调用此工具检查药物相互作用。
"""
from __future__ import annotations

import json
import logging

from engine.tool_registry import global_tool_registry

logger = logging.getLogger(__name__)

# 简易药物冲突知识库（MVP 内存替代，生产环境应接入真实药物数据库）
DRUG_INTERACTIONS: dict[str, list[dict]] = {
    "布洛芬": [
        {"conflict_with": "氯吡格雷", "risk": "增加出血风险", "severity": "高"},
        {"conflict_with": "胃溃疡", "risk": "加重胃肠道损伤，可能引发消化道出血", "severity": "高"},
        {"conflict_with": "高血压", "risk": "可能导致血压升高，影响降压药效果", "severity": "中"},
        {"conflict_with": "肾功能不全", "risk": "NSAIDs可加重肾损伤", "severity": "高"},
    ],
    "阿司匹林": [
        {"conflict_with": "氯吡格雷", "risk": "联用显著增加出血风险", "severity": "高"},
        {"conflict_with": "胃溃疡", "risk": "加重胃肠道出血风险", "severity": "高"},
    ],
    "秋水仙碱": [
        {"conflict_with": "肾功能不全", "risk": "肾功能不全者需减量，CKD3期以上禁用常规剂量", "severity": "高"},
    ],
    "对乙酰氨基酚": [
        {"conflict_with": "肝功能不全", "risk": "可能加重肝损伤", "severity": "中"},
    ],
}


@global_tool_registry.register(
    name="check_drug_interaction",
    description="查询药物之间的相互作用和禁忌。输入药物名称，返回该药物的已知冲突和风险。",
    parameters={
        "type": "object",
        "properties": {
            "drug_name": {
                "type": "string",
                "description": "要查询的药物名称，如'布洛芬'、'阿司匹林'",
            },
            "patient_conditions": {
                "type": "string",
                "description": "患者当前疾病或状态，逗号分隔，如'胃溃疡,高血压'",
            },
        },
        "required": ["drug_name"],
    },
)
async def check_drug_interaction(drug_name: str, patient_conditions: str = "") -> str:
    """查询药物冲突

    参数:
        drug_name: 药物名称
        patient_conditions: 患者当前状况（逗号分隔），用于筛选相关冲突

    返回:
        JSON 格式的药物冲突信息
    """
    interactions = DRUG_INTERACTIONS.get(drug_name, [])
    if not interactions:
        return json.dumps(
            {"drug": drug_name, "result": "未发现已知药物冲突", "interactions": []},
            ensure_ascii=False,
        )

    # 筛选与患者状况相关的冲突
    conditions = [c.strip() for c in patient_conditions.split(",") if c.strip()] if patient_conditions else []
    relevant = []
    for inter in interactions:
        if not conditions or any(c in inter["conflict_with"] for c in conditions):
            relevant.append(inter)

    return json.dumps({
        "drug": drug_name,
        "result": f"发现 {len(relevant)} 条相关药物冲突",
        "interactions": relevant or interactions,  # 无精确匹配时返回全部冲突
    }, ensure_ascii=False)