"""域模型定义 - Pydantic V2

定义系统各模块间传递的核心数据结构，确保类型安全和数据校验。
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PatientProfile(BaseModel):
    """患者画像 - 渐进式构建的结构化患者信息

    每轮对话后通过 LLM 抽取更新，用于画像约束检索（硬约束+软约束）
    """
    user_id: str = Field(description="用户唯一标识，作为 Milvus Patient_Profile 主键")
    diseases: list[str] = Field(default_factory=list, description="既往疾病列表，如['高血压','胃溃疡']")
    medications: list[str] = Field(default_factory=list, description="当前用药列表，如['氯吡格雷']")
    allergies: list[str] = Field(default_factory=list, description="过敏史列表，如['青霉素过敏']")

    def get_contraindications(self) -> list[str]:
        """获取全部禁忌项 = 疾病 + 过敏（均可能成为检索过滤条件）"""
        return list(set(self.diseases + self.allergies))


class ReflectionTriple(BaseModel):
    """归因式反思三元组 - 系统级"免疫记忆"

    当系统回答被打回时，强制输出结构化三元组，
    向量化存入 Reflection_Mem 集合，下次遇到相似问题时拦截预警。
    """
    intent: str = Field(description="原始意图，如'为CKD患者开止痛药'")
    cause: str = Field(description="归因分析，如'忽略NSAIDs肾损风险'")
    avoid_action: str = Field(description="避坑动作，如'必须核查肾功能禁用NSAIDs'")


class RouteDecision(BaseModel):
    """路由决策结果 - 闭环动态路由的输出

    规则拦截或 LLM 路由生成的路由指令，决定走 Simple RAG 还是 MDT 会诊。
    """
    route_path: str = Field(description="路由路径: simple_rag | mdt")
    departments: list[str] = Field(default_factory=list, description="MDT 招募科室列表，如['心内科','消化科']")


class MedicalQuery(BaseModel):
    """用户查询请求"""
    query: str = Field(description="用户输入的医疗咨询问题")
    user_id: str = Field(default="default_user", description="用户标识，关联患者画像")  # 默认值由 config.default_user_id 管理


class MedicalResponse(BaseModel):
    """系统响应 - 包含回答、路由信息、置信度和安全标记"""
    answer: str = Field(description="系统生成的回答文本")
    route_path: str = Field(description="实际路由路径: simple_rag | mdt | safe_fallback | error")
    departments: list[str] = Field(default_factory=list, description="参与的会诊科室")
    sources: list[str] = Field(default_factory=list, description="引用的文献来源")
    confidence: float = Field(default=1.0, description="置信度评分 0.0-1.0")
    is_safe_fallback: bool = Field(default=False, description="是否为安全退避回复（宁拒答不幻觉）")


class Skill(BaseModel):
    """可复用技能 - 从成功回答中提取的规范化处理策略

    设计理念: 将"怎么做"从"知道什么"中分离出来。
    - intent: 触发条件（何时用这个技能）
    - action: 处理策略（怎么做）
    - departments: 适用科室
    - provenance: 来源追踪
    - version: 版本号，merge 时递增
    - usage_count: 被检索使用次数
    - last_used: 最后使用时间
    - status: active | superseded | discarded
    """
    skill_id: str = ""
    intent: str = Field(description="技能触发意图，如'高血压合并痛风患者止痛方案'")
    action: str = Field(description="规范化处理策略，如'禁用NSAIDs，优先对乙酰氨基酚或秋水仙碱'")
    departments: list[str] = Field(default_factory=list, description="适用科室列表")
    source_query: str = Field(default="", description="来源用户查询")
    route_path: str = Field(default="", description="来源路由路径")
    provenance: str = Field(default="", description="来源追踪标识")
    version: int = Field(default=1, description="版本号")
    usage_count: int = Field(default=0, description="被检索使用次数")
    last_used: str = Field(default="", description="最后使用时间")
    status: str = Field(default="active", description="active | superseded | discarded")


class DocumentChunk(BaseModel):
    """检索文档片段 - 混合检索和重排的基本单元"""
    doc_id: str = Field(description="文档唯一标识")
    content: str = Field(description="文档内容文本")
    source: str = Field(default="", description="文献来源")
    score: float = Field(default=0.0, description="检索/重排得分")
    metadata: dict = Field(default_factory=dict, description="附加元数据（科室、禁忌等）")