"""顶层闭环编排器 - 路由→会诊→评估→退避

设计文档四大核心层的闭环实现:
1. 交互与动态路由层: 规则拦截 → LLM路由 → 置信度评估 → 携因打回升级
2. 多专家会诊层: 反思拦截 → 并发会诊 → 共识提炼
3. 记忆与检索协同层: 画像更新 → 混合检索 → 共识引导检索 → 重排 → CoT退避
4. 反思与决策层: Decision Maker → 归因反思 → 安全退避

完整闭环: 查询 → 画像更新 → 路由 → 执行 → 共识引导检索 → 评估 → (升级/退避/输出)
"""
from __future__ import annotations

import json
import logging

from schema.models import MedicalQuery, MedicalResponse, PatientProfile, RouteDecision
from llm.client import AsyncLLMClient
from llm.prompt_templates import SAFE_FALLBACK_RESPONSE, DECISION_MAKER_PROMPT
from router.rule_interceptor import RuleInterceptor
from router.llm_router import LLMRouter
from router.confidence_checker import ConfidenceChecker, RouteEscalationException
from workflow.simple_rag import SimpleRAGWorkflow
from workflow.mdt_consultation import MDTConsultationWorkflow
from memory.profile_extractor import ProfileExtractor
from memory.reflection_manager import ReflectionManager, InsufficientInformationException
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import MedicalReranker
from schema.messages import Message
from config import cfg

logger = logging.getLogger(__name__)


class DecisionMaker:
    """决策器（安全阀）- 设计文档第 4 层

    评估多专家共识质量:
    1. 评估置信度（0.0-1.0）
    2. 检测幻觉风险
    3. 识别逻辑漏洞
    4. 标记安全风险

    不通过时: 触发反思/退避
    通过时: 输出最终医嘱
    """

    def __init__(self, llm: AsyncLLMClient):
        self.llm = llm

    async def evaluate(
        self, report: str, profile: PatientProfile | None = None
    ) -> dict:
        """评估 MDT 共识报告的质量和安全性

        返回: {"approved": bool, "quality_score": float, "reason": str, ...}
        """
        profile_text = ""
        if profile:
            profile_text = f"疾病={profile.diseases}, 用药={profile.medications}, 过敏={profile.allergies}"

        resp = await self.llm.chat(
            messages=[Message(
                role="user",
                content=DECISION_MAKER_PROMPT.format(report=report, profile=profile_text or "无"),
            )],
            response_format={"type": "json_object"},
            temperature=cfg.llm.temperatures["decision_maker"],
        )

        try:
            data = json.loads(resp.content or "{}")
            # 确保必要字段存在
            data.setdefault("approved", True)
            data.setdefault("quality_score", 0.5)
            data.setdefault("hallucination_risk", "low")
            data.setdefault("logical_gaps", [])
            data.setdefault("safety_concerns", [])
            data.setdefault("reason", "")
            return data
        except json.JSONDecodeError:
            # LLM 输出解析失败时，默认通过（避免误拦截）
            logger.warning("Decision Maker 输出解析失败，默认通过")
            return {"approved": True, "quality_score": 0.5, "reason": "评估结果解析失败"}


class MedicalOrchestrator:
    """顶层闭环编排器

    完整闭环流程:
    1. 渐进式画像更新
    2. CoT 安全退避预检（检索质量检查）
    3. 闭环动态路由（规则拦截 → LLM路由）
    4. 执行工作流（Simple RAG / MDT）
    5. 置信度评估（Simple RAG 末尾，可能升级到 MDT）
    6. 共识引导检索（MDT 内部：共识摘要 → Hybrid检索 → Reranker → LLM验证）
    7. Decision Maker 评估（MDT 末尾，基于证据验证后的共识，可能触发反思/退避）
    8. 失败处理（反思沉淀 / CoT退避 / 携因打回）
    """

    def __init__(
        self,
        llm: AsyncLLMClient,
        retriever: HybridRetriever,
        reranker: MedicalReranker,
        profile_extractor: ProfileExtractor,
        reflection_manager: ReflectionManager,
        ner_service_url: str = "",
    ):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.profile_extractor = profile_extractor
        self.reflection = reflection_manager

        # 路由组件
        self.rule_interceptor = RuleInterceptor(ner_service_url=ner_service_url)
        self.llm_router = LLMRouter(llm)
        self.confidence_checker = ConfidenceChecker(llm)

        # 决策器（安全阀）
        self.decision_maker = DecisionMaker(llm)

    async def process(self, query: MedicalQuery) -> MedicalResponse:
        """处理用户查询 - 主入口"""
        logger.info(f"开始处理查询: '{query.query}' (user={query.user_id})")

        # ---- Step 0: LLM 可用性检查 ----
        # 未配置 API key 时，LLM 不可用，走纯检索路径
        llm_available = getattr(self.llm, '_api_key_configured', True)

        try:
            # ---- Step 1: 渐进式画像更新 ----
            # 将画像注入到工具模块，使 Agent 在 ReAct 循环中调用检索时也能享受画像约束
            from tools.literature_search import set_current_profile
            # LLM 可用时才做画像抽取（需要调用 LLM），否则使用空画像
            if llm_available:
                profile = await self.profile_extractor.extract_and_update(
                    query.user_id, query.query
                )
            else:
                profile = self.profile_extractor.get_profile(query.user_id)

            # 注入画像到工具模块，使 Agent 调用 literature_search 时自动携带画像约束
            set_current_profile(profile)

            # ---- Step 2: CoT 安全退避预检 ----
            # 检索是否有结果。若为空，直接触发退避。
            quick_docs = await self.retriever.retrieve(query.query, profile=profile, top_k=cfg.retrieval.quick_check_top_k)
            if not quick_docs:
                logger.warning("CoT 安全退避: 检索结果为空")
                return MedicalResponse(
                    answer=SAFE_FALLBACK_RESPONSE,
                    route_path="safe_fallback",
                    confidence=0.0,
                    is_safe_fallback=True,
                )
            # 保存预检结果供后续复用，避免重复检索
            self._pre_retrieved = quick_docs

            # ---- Step 3: 闭环动态路由 ----
            if llm_available:
                decision = await self._route_async(query.query, profile)
            else:
                # LLM 不可用时，只用规则拦截（快车道），灰度问题走 simple_rag
                decision = self.rule_interceptor.intercept(query.query)
                if decision is None:
                    decision = RouteDecision(route_path="simple_rag", departments=[])

            # ---- Step 4: 执行工作流 ----
            if not llm_available:
                # LLM 不可用: 走纯检索路径，拼接文档作为回答
                return await self._retrieval_only_response(query, profile, decision)

            if decision.route_path == "simple_rag":
                # 简单 RAG 路径
                response, docs = await self._run_simple_rag(query, profile)

                # ---- Step 5: 置信度评估 → 可能升级到 MDT ----
                try:
                    await self.confidence_checker.evaluate(response.answer, docs)
                    return response
                except RouteEscalationException as e:
                    # 携因打回: 将失败原因注入 MDT
                    logger.warning(f"置信度不足，升级至 MDT: {e.reason}")
                    await self._store_reflection(query.query, e.reason)
                    return await self._run_mdt(
                        query, profile,
                        departments=decision.departments or self._infer_departments(query.query),
                        escalation_reason=e.reason,
                    )
            else:
                # MDT 路径
                return await self._run_mdt(
                    query, profile,
                    departments=decision.departments,
                )

        except InsufficientInformationException:
            # ---- CoT 安全退避 ----
            # 设计文档: "切断 ReAct 循环，禁止发散推理，走硬编码的保守策略链路"
            logger.warning("信息不足异常: CoT 安全退避")
            return MedicalResponse(
                answer=SAFE_FALLBACK_RESPONSE,
                route_path="safe_fallback",
                confidence=0.0,
                is_safe_fallback=True,
            )
        except Exception as e:
            # 医疗系统不能 Crash: 所有异常必须有兜底处理
            logger.error(f"系统异常: {e}", exc_info=True)
            return MedicalResponse(
                answer="抱歉，系统处理过程中出现异常，请稍后重试或建议线下就医。",
                route_path="error",
                confidence=0.0,
            )

    async def _route_async(self, query: str, profile: PatientProfile):
        """异步动态路由: 规则拦截（快车道）→ LLM路由（慢车道）"""
        # 快车道: 规则拦截
        decision = self.rule_interceptor.intercept(query)
        if decision is not None:
            return decision

        # 慢车道: LLM 结构化意图路由
        profile_summary = f"疾病:{profile.diseases} 用药:{profile.medications} 过敏:{profile.allergies}"
        return await self.llm_router.route(query, profile_summary)

    async def _run_simple_rag(self, query: MedicalQuery, profile: PatientProfile):
        """执行简单 RAG"""
        workflow = SimpleRAGWorkflow(self.llm, self.retriever, self.reranker, self.reflection)
        return await workflow.run(query, profile)

    async def _run_mdt(
        self,
        query: MedicalQuery,
        profile: PatientProfile,
        departments: list[str],
        escalation_reason: str = "",
    ):
        """执行 MDT 会诊（含共识引导检索）+ Decision Maker 评估"""
        workflow = MDTConsultationWorkflow(
            self.llm, self.reflection, self.retriever, self.reranker, profile
        )
        response = await workflow.run(query, departments, escalation_reason)

        # ---- Decision Maker 安全阀评估（基于证据验证后的共识） ----
        try:
            evaluation = await self.decision_maker.evaluate(response.answer, profile)

            if not evaluation.get("approved", True):
                # 共识质量不通过
                quality_score = evaluation.get("quality_score", 0)
                reason = evaluation.get("reason", "共识质量评估未通过")
                hallucination_risk = evaluation.get("hallucination_risk", "low")
                safety_concerns = evaluation.get("safety_concerns", [])
                logger.warning(f"Decision Maker 拦截: quality={quality_score}, hallucination={hallucination_risk}, reason={reason}")

                # 仅当高幻觉风险 或 质量极低时走 CoT 安全退避
                # 中等质量问题（如"缺乏个体化评估"）仍输出回答，仅附加警告
                if hallucination_risk == "high" or quality_score < cfg.decision_maker.quality_threshold:
                    await self._store_reflection(query.query, reason)
                    return MedicalResponse(
                        answer=SAFE_FALLBACK_RESPONSE,
                        route_path="safe_fallback",
                        confidence=quality_score,
                        is_safe_fallback=True,
                    )

                # 中等质量问题 → 记录反思但仍输出回答（附加风险提示）
                await self._store_reflection(query.query, reason)
                response.confidence = quality_score
                if evaluation.get("logical_gaps"):
                    response.answer += f"\n\n⚠️ 注意：{'; '.join(evaluation['logical_gaps'])}"

        except Exception as e:
            logger.warning(f"Decision Maker 评估失败: {e}")

        return response

    async def _retrieval_only_response(
        self,
        query: MedicalQuery,
        profile: PatientProfile,
        decision: RouteDecision,
    ) -> MedicalResponse:
        """LLM 不可用时的纯检索路径

        不调用 LLM，仅检索知识库并拼接文档片段作为回答。
        保证系统在无 LLM 时仍可提供基础知识检索功能。
        """
        docs = await self.retriever.retrieve(query.query, profile=profile, top_k=cfg.retrieval.retrieval_only_top_k)
        reranked = await self.reranker.rerank(query.query, docs, top_k=cfg.retrieval.retrieval_only_rerank_top_k)

        if not reranked:
            return MedicalResponse(
                answer="未检索到相关医学知识。请配置 MDT_LLM_API_KEY 环境变量以启用完整功能。",
                route_path="retrieval_only",
                confidence=0.0,
            )

        # 拼接检索结果作为回答
        answer_parts = ["【检索结果（LLM 未配置，以下为原始文献片段）】\n"]
        for i, doc in enumerate(reranked):
            answer_parts.append(f"[{i+1}] {doc.content}\n—— 来源：{doc.source}\n")

        sources = [doc.source for doc in reranked if doc.source]
        return MedicalResponse(
            answer="\n".join(answer_parts),
            route_path="retrieval_only",
            departments=decision.departments,
            sources=sources,
            confidence=cfg.decision_maker.retrieval_only_confidence,
        )

    def _infer_departments(self, query: str) -> list[str]:
        """从查询推断科室（升级时可能需要）"""
        from router.rule_interceptor import DEPARTMENT_MAP, MEDICAL_ENTITY_PATTERNS
        import re

        entities = []
        for pattern in MEDICAL_ENTITY_PATTERNS:
            entities.extend(re.findall(pattern, query))

        departments = list(set(
            DEPARTMENT_MAP.get(e, "全科") for e in set(entities) if e in DEPARTMENT_MAP
        ))
        return departments or ["全科"]

    async def _store_reflection(self, query: str, reason: str):
        """存储归因反思记忆"""
        try:
            await self.reflection.generate_and_store(query, reason)
        except Exception as e:
            logger.error(f"反思存储失败: {e}")