"""顶层闭环编排器 - 路由→会诊→评估→退避

设计文档四大核心层的闭环实现:
1. 交互与动态路由层: 规则拦截 → LLM路由 → 置信度评估 → 携因打回升级
2. 多专家会诊层: 反思拦截 → 并发会诊 → 共识提炼
3. 记忆与检索协同层: 画像更新 → 混合检索 → 共识引导检索 → 重排 → CoT退避
4. 反思与决策层: Decision Maker → 归因反思 → 安全退避

Harness 增强 (Agent = Model + Harness):
5. 追踪观测层: TraceID 传播 + Span 记录 + 执行图
6. 评估层: 7维确定性评分 + 基线对比
7. 安全守卫层: 工具调用验证 + 限流 + 成本追踪
8. 上下文管理层: 四层记忆层级 + 多轮对话 + 上下文窗口调度 + 智能组装

上下文架构 (v2.0):
  L0 Working Memory  — 系统提示、核心约束 (~2K tokens, 始终在线)
  L1 Short-Term Memory — 对话历史滑动窗口 (~4K tokens, 时间/重要性管理)
  L2 Long-Term Memory  — 持久化技能+反思 (~8K tokens, 按需检索注入)
  L3 External Memory   — 检索结果+外部知识 (~8K tokens, 按需加载)

完整闭环: 查询 → 画像更新 → 路由 → 执行 → 共识引导检索 → 评估 → (升级/退避/输出)
"""
from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

from schema.models import MedicalQuery, MedicalResponse, PatientProfile, RouteDecision, DocumentChunk
from llm.client import AsyncLLMClient
from llm.prompt_templates import SAFE_FALLBACK_RESPONSE, DECISION_MAKER_PROMPT
from router.rule_interceptor import RuleInterceptor
from router.llm_router import LLMRouter
from router.confidence_checker import ConfidenceChecker, RouteEscalationException
from workflow.simple_rag import SimpleRAGWorkflow
from workflow.mdt_consultation import MDTConsultationWorkflow
from memory.profile_extractor import ProfileExtractor
from memory.reflection_manager import ReflectionManager, InsufficientInformationException
from memory.skill_manager import SkillManager
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import MedicalReranker
from monitoring.metrics import PipelineTimer
from monitoring.tracing import TraceContext, trace_manager
from schema.messages import Message
from config import cfg
from memory.event_memory import EventType

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
            logger.warning("Decision Maker 输出解析失败，保守退避")
            return {"approved": False, "quality_score": 0.0, "hallucination_risk": "high",
                    "logical_gaps": [], "safety_concerns": ["评估结果解析失败"],
                    "reason": "评估结果解析失败，保守退避"}


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

    Harness 增强:
    9. 追踪观测: TraceID 贯穿全流程 + Span 阶段计时 + 执行图
    10. 安全守卫: 工具调用验证 + 限流 + 成本追踪
    11. 上下文管理: 四层记忆 + 多轮对话 + 上下文窗口调度 + 智能组装
    12. 在线评估: 7维确定性评分
    """

    def __init__(
        self,
        llm: AsyncLLMClient,
        retriever: HybridRetriever,
        reranker: MedicalReranker,
        profile_extractor: ProfileExtractor,
        reflection_manager: ReflectionManager,
        skill_manager: Optional[SkillManager] = None,
        ner_service_url: str = "",
        enable_tracing: bool = True,
        safety_guard=None,
        context_manager=None,
        harness_evaluator=None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.profile_extractor = profile_extractor
        self.reflection = reflection_manager

        # Agent 自进化: 技能管理器
        self.skill_manager = skill_manager

        # 路由组件
        self.rule_interceptor = RuleInterceptor(ner_service_url=ner_service_url)
        self.llm_router = LLMRouter(llm)
        self.confidence_checker = ConfidenceChecker(llm)

        # 决策器（安全阀）
        self.decision_maker = DecisionMaker(llm)

        # Harness 组件
        self.enable_tracing = enable_tracing
        self.safety_guard = safety_guard
        self.context_manager = context_manager
        self.harness_evaluator = harness_evaluator

        # ---- Run-level 记忆隔离 ----
        from context.run_memory import RunMemoryManager
        self.run_memory = RunMemoryManager()

        # ---- 事件驱动的情节记忆 ----
        from memory.event_memory import EventMemory
        self.event_memory = EventMemory()

        # ---- Agent 自主记忆工具: 注入 ContextManager 引用 ----
        from tools.memory_tools import set_context_manager
        set_context_manager(context_manager)

    async def process(self, query: MedicalQuery) -> MedicalResponse:
        """处理用户查询 - 主入口"""
        start_time = time.time()
        logger.info(f"开始处理查询: '{query.query}' (user={query.user_id})")

        # ---- Run-level 隔离: 开始新 Run ----
        run_ctx = self.run_memory.begin_run(
            session_id=getattr(self.context_manager, 'session_id', ''),
            user_id=query.user_id,
        )

        # ---- 事件记忆: 摄取用户查询为第一个事件 ----
        self.event_memory.ingest(
            f"用户查询: {query.query}",
            event_type=EventType.SYMPTOM_ANALYSIS,
            metadata={"user_id": query.user_id},
        )

        # ---- Harness: 上下文管理器初始化 ----
        if self.context_manager:
            if not self.context_manager._session_started:
                self.context_manager.begin_session()
            self.context_manager.set_system_prompt(
                "你是一个医疗知识助手，只基于检索到的文献回答。禁止编造。",
                importance=1.0,
            )
            self.context_manager.append_system_prompt(
                "安全约束: 无法回答时建议线下就医，不可冒险推测。",
                importance=0.9,
            )

        # ---- Harness: 创建追踪上下文 ----
        trace = None
        if self.enable_tracing:
            trace = trace_manager.begin_trace(
                name=f"query:{query.query[:30]}",
                trace_id=f"t_{query.user_id}_{int(start_time)}",
            )
            trace.begin_span("process", metadata={"query": query.query, "user_id": query.user_id})

        # ---- Step 0: LLM 可用性检查 ----
        # 未配置 API key 时，LLM 不可用，走纯检索路径
        llm_available = getattr(self.llm, '_api_key_configured', True)

        try:
            # ---- Step 1: 渐进式画像更新 ----
            timer = PipelineTimer()
            if trace:
                trace.begin_span("profile_update", parent_id=trace._roots[0].span_id if trace._roots else None)
            # 将画像注入到工具模块，使 Agent 在 ReAct 循环中调用检索时也能享受画像约束
            from tools.literature_search import set_current_profile
            # LLM 可用时才做画像抽取（需要调用 LLM），否则使用空画像
            if llm_available:
                with timer.stage("profile_update"):
                    profile = await self.profile_extractor.extract_and_update(
                        query.user_id, query.query
                    )
            else:
                profile = self.profile_extractor.get_profile(query.user_id)

            # 注入画像到工具模块，使 Agent 调用 literature_search 时自动携带画像约束
            set_current_profile(profile)

            # ---- Harness 上下文: 注入患者画像到穿透项 ----
            if self.context_manager and profile:
                self.context_manager.inject_profile(
                    f"疾病={profile.diseases}, 用药={profile.medications}, 过敏={profile.allergies}"
                )

            if trace:
                trace.end_span(trace._stack[-1].span_id if trace._stack else "")

            # ---- Step 2: CoT 安全退避预检 ----
            with timer.stage("cot_precheck"):
                if trace:
                    trace.begin_span("cot_precheck")
                quick_docs = await self.retriever.retrieve(query.query, profile=profile, top_k=cfg.retrieval.quick_check_top_k)
                if trace:
                    trace.end_span(trace._stack[-1].span_id if trace._stack else "",
                                   metadata={"num_docs": len(quick_docs)})
            if not quick_docs:
                logger.warning("CoT 安全退避: 检索结果为空")
                if trace:
                    trace.end_span(trace._roots[0].span_id if trace._roots else "", status="fallback")
                    trace_manager.end_trace(trace.trace_id)
                return MedicalResponse(
                    answer=SAFE_FALLBACK_RESPONSE,
                    route_path="safe_fallback",
                    confidence=0.0,
                    is_safe_fallback=True,
                )

            # ---- Agent 自进化: 检索相关技能注入上下文 ----
            skill_hints = ""
            if self.skill_manager and cfg.skill.extraction_enabled:
                try:
                    skills = await self.skill_manager.search_skills(query.query)
                    if skills:
                        skill_lines = []
                        for s in skills:
                            hint = f"{s.intent}: {s.action}"
                            skill_lines.append(f"[技能] {hint}")
                            if self.context_manager:
                                self.context_manager.inject_skill(hint)
                        skill_hints = "\n参考既往成功经验：\n" + "\n".join(skill_lines)
                        logger.info(f"技能命中: {len(skills)} 条")
                        # 事件记忆: 技能命中
                        self.event_memory.ingest(f"技能命中: {len(skills)} 条", event_type=EventType.SAFETY_CHECK)
                except Exception as e:
                    logger.debug(f"技能检索失败: {e}")

            # ---- Step 3: 闭环动态路由 ----
            if llm_available:
                with timer.stage("routing"):
                    if trace:
                        trace.begin_span("routing")
                    decision = await self._route_async(query.query, profile)
                    if trace:
                        trace.end_span(trace._stack[-1].span_id if trace._stack else "",
                                       metadata={"route": decision.route_path, "departments": decision.departments})
            else:
                decision = self.rule_interceptor.intercept(query.query)
                if decision is None:
                    decision = RouteDecision(route_path="simple_rag", departments=[])

            # ---- 获取多轮对话历史（上下文穿透 + 摘要 + 最近对话） ----
            history_context = ""
            if self.context_manager:
                history_context = self.context_manager.get_conversation_context()
                if history_context:
                    budget_warning = self.context_manager.check_budget_and_warn()
                    if budget_warning:
                        logger.warning(budget_warning)

            # ---- Step 4: 执行工作流 ----
            if not llm_available:
                response = await self._retrieval_only_response(query, profile, decision)
                self.event_memory.ingest("LLM不可用，纯检索模式", event_type=EventType.DECISION)
                self._finalize_run(run_ctx, response, context_manager=self.context_manager)
                return response

            if decision.route_path == "simple_rag":
                if trace:
                    trace.begin_span("execute_simple_rag")
                self.event_memory.ingest("Simple RAG 检索开始", event_type=EventType.LITERATURE_SEARCH)
                response, docs = await self._run_simple_rag(query, profile, skill_hints=skill_hints, history_context=history_context)
                if trace:
                    trace.end_span(trace._stack[-1].span_id if trace._stack else "")
                # 事件记忆: 检索结果
                self.event_memory.ingest(
                    f"Simple RAG 检索返回 {len(docs)} 篇文献，信心={response.confidence}",
                    event_type=EventType.LITERATURE_SEARCH,
                    metadata={"source_count": len(docs), "confidence": response.confidence},
                )
                # Run 产物: 缓存检索结果
                run_ctx.cache_retrieval(query.query, docs)
                run_ctx.route_path = "simple_rag"

                # ---- Step 5: 置信度评估 → 可能升级到 MDT ----
                try:
                    if trace:
                        trace.begin_span("confidence_check")
                    await self.confidence_checker.evaluate(response.answer, docs)
                    if trace:
                        trace.end_span(trace._stack[-1].span_id if trace._stack else "")
                    await self._log_pipeline(timer, "simple_rag")
                    await self._maybe_extract_skill(response, query)
                    # ---- 上下文: 记录多轮对话 ----
                    if self.context_manager:
                        self.context_manager.add_conversation(
                            query=query.query,
                            response=response.answer,
                            route_path="simple_rag",
                            confidence=response.confidence,
                        )
                    self._finalize_run(run_ctx, response, context_manager=self.context_manager)
                    if trace:
                        trace_manager.end_trace(trace.trace_id)
                    return response
                except RouteEscalationException as e:
                    logger.warning(f"置信度不足，升级至 MDT: {e.reason}")
                    self.event_memory.ingest(
                        f"Simple RAG 升级至 MDT: {e.reason}",
                        event_type=EventType.REFLECTION,
                        metadata={"escalation_reason": e.reason},
                    )
                    if trace:
                        trace.add_event("escalation_to_mdt", {"reason": e.reason})
                    await self._store_reflection(query.query, e.reason)
                    run_ctx.route_path = "mdt_escalated"
                    result = await self._run_mdt(
                        query, profile,
                        departments=decision.departments or self._infer_departments(query.query),
                        escalation_reason=e.reason,
                        skill_hints=skill_hints,
                        history_context=history_context,
                    )
                    await self._log_pipeline(timer, "mdt_escalated")
                    # ---- 上下文: 记录升级路径 ----
                    if self.context_manager:
                        self.context_manager.add_conversation(
                            query=query.query,
                            response=result.answer,
                            route_path="mdt_escalated",
                            confidence=result.confidence,
                            departments=result.departments,
                        )
                    self._finalize_run(run_ctx, result, context_manager=self.context_manager)
                    if trace:
                        trace_manager.end_trace(trace.trace_id)
                    return result
            else:
                run_ctx.route_path = "mdt"
                run_ctx.departments = decision.departments
                self.event_memory.ingest(
                    f"MDT 会诊开始: 科室={decision.departments}",
                    event_type=EventType.CONSENSUS_FORMATION,
                )
                result = await self._run_mdt(
                    query, profile,
                    departments=decision.departments,
                    skill_hints=skill_hints,
                    history_context=history_context,
                )
                self.event_memory.ingest(
                    f"MDT 会诊完成: 信心={result.confidence}",
                    event_type=EventType.DECISION,
                )
                await self._log_pipeline(timer, "mdt")
                await self._maybe_extract_skill(result, query)
                # ---- 上下文: 记录会诊 ----
                if self.context_manager:
                    self.context_manager.add_conversation(
                        query=query.query,
                        response=result.answer,
                        route_path="mdt",
                        confidence=result.confidence,
                        departments=result.departments,
                    )
                self._finalize_run(run_ctx, result, context_manager=self.context_manager)
                if trace:
                    trace_manager.end_trace(trace.trace_id)
                return result

        except InsufficientInformationException:
            if trace:
                trace.end_span(trace._roots[0].span_id if trace._roots else "", status="fallback")
                trace_manager.end_trace(trace.trace_id)
            logger.warning("信息不足异常: CoT 安全退避")
            self.event_memory.ingest("CoT安全退避: 信息不足", event_type=EventType.SAFETY_CHECK)
            response = MedicalResponse(
                answer=SAFE_FALLBACK_RESPONSE,
                route_path="safe_fallback",
                confidence=0.0,
                is_safe_fallback=True,
            )
            self._finalize_run(run_ctx, response)
            return response
        except Exception as e:
            if trace:
                trace.end_span(trace._roots[0].span_id if trace._roots else "", status="error",
                                metadata={"error": str(e)})
                trace_manager.end_trace(trace.trace_id)
            logger.error(f"系统异常: {e}", exc_info=True)
            self.event_memory.ingest(f"系统异常: {e}", event_type=EventType.SAFETY_CHECK)
            response = MedicalResponse(
                answer="抱歉，系统处理过程中出现异常，请稍后重试或建议线下就医。",
                route_path="error",
                confidence=0.0,
            )
            self._finalize_run(run_ctx, response)
            return response

    async def _log_pipeline(self, timer: PipelineTimer, path: str):
        logger.info(
            f"Pipeline [{path}]: profile={timer.get('profile_update'):.0f}ms, "
            f"precheck={timer.get('cot_precheck'):.0f}ms, "
            f"routing={timer.get('routing'):.0f}ms, total={timer.total_ms:.0f}ms"
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

    async def _run_simple_rag(self, query: MedicalQuery, profile: PatientProfile, skill_hints: str = "", history_context: str = ""):
        """执行简单 RAG"""
        workflow = SimpleRAGWorkflow(self.llm, self.retriever, self.reranker, self.reflection)
        return await workflow.run(query, profile, skill_hints=skill_hints, history_context=history_context)

    async def _run_mdt(
        self,
        query: MedicalQuery,
        profile: PatientProfile,
        departments: list[str],
        escalation_reason: str = "",
        skill_hints: str = "",
        history_context: str = "",
    ):
        """执行 MDT 会诊（含共识引导检索）+ Decision Maker 评估"""
        workflow = MDTConsultationWorkflow(
            self.llm, self.reflection, self.retriever, self.reranker, profile
        )
        response = await workflow.run(query, departments, escalation_reason, skill_hints=skill_hints, history_context=history_context)

        # ---- Decision Maker 安全阀评估（基于证据验证后的共识） ----

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
            if self.context_manager:
                self.context_manager.inject_reflection(f"历史教训: {reason}")
        except Exception as e:
            logger.error(f"反思存储失败: {e}")

    def _finalize_run(self, run_ctx, response, context_manager=None):
        """结束 Run，关联指标并持久化有价值内容"""
        run_ctx.route_path = response.route_path
        run_ctx.confidence = response.confidence
        run_ctx.departments = response.departments or []
        self.run_memory.finish_run(context_manager=context_manager)

    async def _maybe_extract_skill(self, response: MedicalResponse, query: MedicalQuery):
        """从成功回答中提取技能（Agent 自进化）"""
        if not self.skill_manager or not cfg.skill.extraction_enabled:
            return
        if response.is_safe_fallback or response.route_path in ("error", "retrieval_only"):
            return
        if response.confidence < cfg.skill.min_confidence_for_extraction:
            return
        try:
            skill = await self.skill_manager.extract_from_success(
                query=query.query,
                answer=response.answer,
                route_path=response.route_path,
                departments=response.departments,
            )
            if skill:
                await self.skill_manager.store(skill)
        except Exception as e:
            logger.debug(f"技能提取失败: {e}")

    async def process_stream(self, query: MedicalQuery) -> AsyncIterator[dict]:
        """流式处理 — 使用 chat_stream 实现真正的 token 级流式输出

        yield 格式: {"type":"status"|"content"|"done"|"error", ...}
        """
        start_time = time.time()
        yield {"type": "status", "text": "开始处理查询..."}

        # Run-level 隔离
        run_ctx = self.run_memory.begin_run(
            session_id=getattr(self.context_manager, 'session_id', ''),
            user_id=query.user_id,
        )
        self.event_memory.ingest(
            f"用户查询: {query.query}", event_type=EventType.SYMPTOM_ANALYSIS, metadata={"user_id": query.user_id},
        )

        if self.context_manager:
            if not self.context_manager._session_started:
                self.context_manager.begin_session()
            self.context_manager.set_system_prompt("你是一个医疗知识助手，只基于检索到的文献回答。禁止编造。", importance=1.0)

        llm_available = getattr(self.llm, '_api_key_configured', True)

        # Image update
        from tools.literature_search import set_current_profile
        if llm_available:
            yield {"type": "status", "text": "更新患者画像..."}
            profile = await self.profile_extractor.extract_and_update(query.user_id, query.query)
        else:
            profile = self.profile_extractor.get_profile(query.user_id)
        set_current_profile(profile)
        if self.context_manager and profile:
            self.context_manager.inject_profile(f"疾病={profile.diseases}, 用药={profile.medications}, 过敏={profile.allergies}")

        # Precheck
        quick_docs = await self.retriever.retrieve(query.query, profile=profile, top_k=cfg.retrieval.quick_check_top_k)
        if not quick_docs:
            yield {"type": "status", "text": "CoT 安全退避: 检索为空"}
            self.event_memory.ingest("CoT安全退避", event_type=EventType.SAFETY_CHECK)
            yield {
                "type": "done",
                "text": "抱歉，缺乏权威文献支撑，强烈建议线下就医。",
                "route_path": "safe_fallback",
                "confidence": 0.0,
                "sources": [],
            }
            self._finalize_run(run_ctx, MedicalResponse(answer="", route_path="safe_fallback", confidence=0.0))
            return

        # Skills
        skill_hints = ""
        if self.skill_manager and cfg.skill.extraction_enabled:
            try:
                skills = await self.skill_manager.search_skills(query.query)
                if skills:
                    skill_hints = "\n参考既往成功经验：\n" + "\n".join(f"[技能] {s.intent}: {s.action}" for s in skills)
                    self.event_memory.ingest(f"技能命中: {len(skills)} 条", event_type=EventType.SAFETY_CHECK)
            except Exception:
                pass

        # Conversation history
        history_context = ""
        if self.context_manager:
            history_context = self.context_manager.get_conversation_context()

        # Route
        if llm_available:
            decision = await self._route_async(query.query, profile)
            yield {"type": "status", "text": f"路由: {decision.route_path}"}
        else:
            decision = self.rule_interceptor.intercept(query.query)
            if decision is None:
                decision = RouteDecision(route_path="simple_rag", departments=[])

        if not llm_available:
            yield {"type": "error", "text": "LLM 不可用"}
            return

        # Execute based on route
        if decision.route_path == "simple_rag":
            run_ctx.route_path = "simple_rag"
            self.event_memory.ingest("Simple RAG 流式执行", event_type=EventType.LITERATURE_SEARCH)

            wf = SimpleRAGWorkflow(self.llm, self.retriever, self.reranker, self.reflection)
            full_text = ""
            async for chunk in wf.run_stream(query, profile, skill_hints=skill_hints, history_context=history_context):
                if chunk["type"] == "content":
                    full_text += chunk["text"]
                elif chunk["type"] == "done":
                    full_text = chunk.get("text", full_text)
                    # Post-process: add metadata
                    docs_data = chunk.get("documents", [])
                    run_ctx.cache_retrieval(query.query, [
                        DocumentChunk(doc_id=f"stream_{i}", content=d["content"], source=d.get("source",""), score=d.get("score",0))
                        for i, d in enumerate(docs_data)
                    ])
                    _sources = chunk.get("sources", [])
                    chunk["confidence"] = 1.0
                    if self.context_manager:
                        self.context_manager.add_conversation(
                            query=query.query, response=full_text, route_path="simple_rag", confidence=1.0,
                        )
                    self._finalize_run(run_ctx, MedicalResponse(answer=full_text, route_path="simple_rag", sources=_sources, confidence=1.0))
                yield chunk

            # 技能提取
            await self._maybe_extract_skill(
                MedicalResponse(answer=full_text, route_path="simple_rag", sources=chunk.get("sources", [])),
                query,
            )

        else:
            run_ctx.route_path = "mdt"
            run_ctx.departments = decision.departments
            self.event_memory.ingest(f"MDT 流式会诊: 科室={decision.departments}", event_type=EventType.CONSENSUS_FORMATION)

            wf = MDTConsultationWorkflow(self.llm, self.reflection, self.retriever, self.reranker, profile)
            full_text = ""
            async for chunk in wf.run_stream(
                query, departments=decision.departments,
                escalation_reason="", skill_hints=skill_hints, history_context=history_context,
            ):
                if chunk["type"] == "content":
                    full_text += chunk["text"]
                elif chunk["type"] == "done":
                    full_text = chunk.get("text", full_text)
                    if self.context_manager:
                        self.context_manager.add_conversation(
                            query=query.query, response=full_text, route_path="mdt",
                            confidence=1.0, departments=decision.departments,
                        )
                    self._finalize_run(
                        run_ctx,
                        MedicalResponse(answer=full_text, route_path="mdt", departments=decision.departments, sources=chunk.get("sources", []), confidence=1.0),
                    )
                yield chunk

            # 技能提取
            await self._maybe_extract_skill(
                MedicalResponse(answer=full_text, route_path="mdt", departments=decision.departments, sources=chunk.get("sources", [])),
                query,
            )

        elapsed = time.time() - start_time
        self.event_memory.ingest(f"流式处理完成 ({elapsed:.0f}s)", event_type=EventType.DECISION)