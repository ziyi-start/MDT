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
    "氯吡格雷": [
        {"conflict_with": "布洛芬", "risk": "增加出血风险", "severity": "高"},
        {"conflict_with": "阿司匹林", "risk": "联用显著增加出血风险", "severity": "高"},
        {"conflict_with": "胃溃疡", "risk": "双抗治疗期间消化道出血风险升高", "severity": "高"},
    ],
    "硝苯地平": [
        {"conflict_with": "高血压", "risk": "降压药，需注意剂量调整", "severity": "低"},
    ],
    "二甲双胍": [
        {"conflict_with": "肾功能不全", "risk": "肾功能不全者需减量或禁用，防止乳酸酸中毒", "severity": "高"},
    ],
    "甲氨蝶呤": [
        {"conflict_with": "肾功能不全", "risk": "肾功能不全者清除率降低，毒性增加", "severity": "高"},
        {"conflict_with": "胃溃疡", "risk": "可能加重胃肠道黏膜损伤", "severity": "中"},
    ],
    "别嘌醇": [
        {"conflict_with": "肾功能不全", "risk": "需根据肾功能调整剂量", "severity": "中"},
    ],
}


@global_tool_registry.register(
    name="check_drug_interaction",
    description="查询药物之间的相互作用和禁忌。可输入单个或多个药物（逗号分隔），返回药物间的已知冲突和风险。",
    parameters={
        "type": "object",
        "properties": {
            "drug_names": {
                "type": "string",
                "description": "药物名称，多个用逗号分隔，如'布洛芬,氯吡格雷'",
            },
            "patient_conditions": {
                "type": "string",
                "description": "患者当前疾病或状态，逗号分隔，如'胃溃疡,高血压'",
            },
        },
        "required": ["drug_names"],
    },
)
async def check_drug_interaction(drug_names: str, patient_conditions: str = "") -> str:
    """查询药物冲突

    参数:
        drug_names: 药物名称（多个逗号分隔），如"布洛芬,氯吡格雷"
        patient_conditions: 患者当前状况（逗号分隔），用于筛选相关冲突

    返回:
        JSON 格式的药物冲突信息
    """
    drugs = [d.strip() for d in drug_names.split(",") if d.strip()]
    if not drugs:
        return json.dumps({"error": "未提供药物名称"}, ensure_ascii=False)

    conditions = [c.strip() for c in patient_conditions.split(",") if c.strip()] if patient_conditions else []

    all_interactions = []
    pair_conflicts = []

    # 单药冲突检查
    for drug in drugs:
        interactions = DRUG_INTERACTIONS.get(drug, [])
        if interactions:
            relevant = []
            for inter in interactions:
                if not conditions or any(c in inter["conflict_with"] for c in conditions):
                    relevant.append(inter)
            all_interactions.extend([{"drug": drug, **r} for r in (relevant or interactions)])

    # 多药两两冲突检查
    if len(drugs) > 1:
        for i in range(len(drugs)):
            for j in range(i + 1, len(drugs)):
                d1, d2 = drugs[i], drugs[j]
                # 检查 d1 对 d2 的冲突
                for inter in DRUG_INTERACTIONS.get(d1, []):
                    if d2 in inter["conflict_with"]:
                        pair_conflicts.append({
                            "drug_pair": [d1, d2],
                            "risk": inter["risk"],
                            "severity": inter["severity"],
                        })
                # 检查 d2 对 d1 的冲突
                for inter in DRUG_INTERACTIONS.get(d2, []):
                    if d1 in inter["conflict_with"]:
                        pair_conflicts.append({
                            "drug_pair": [d1, d2],
                            "risk": inter["risk"],
                            "severity": inter["severity"],
                        })

    # 去重 pair_conflicts
    seen = set()
    unique_pairs = []
    for pc in pair_conflicts:
        key = tuple(sorted(pc["drug_pair"]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pc)

    return json.dumps({
        "drugs": drugs,
        "single_drug_interactions": all_interactions,
        "drug_pair_conflicts": unique_pairs,
        "total_conflicts": len(all_interactions) + len(unique_pairs),
    }, ensure_ascii=False)