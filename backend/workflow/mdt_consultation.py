"""多专家 MDT 会诊工作流

设计文档 3.3 节 + 3.4 节: 多专家会诊层
- 根据 LLM 路由返回的 departments，动态实例化对应专科的 ReactEngine
- 共识引导检索: ① 给各专科 Expert 注入患者画像，要求主动构建专业检索词；
  ② 共识提炼后用共识文本回检知识库，经 HybridRetriever→Reranker→LLM 验证
- 反思拦截: 执行前检索 Reflection_Mem，高相似度命中则强插 Hint
- 并发会诊: asyncio.gather 并发执行多专家 ReAct 循环
- 共识提炼: 收集多专家结果，调用 LLM 提炼最终会诊报告

CoT 安全退避:
- 检索阶段若 Reranker 最高分低于极低阈值 → 触发 InsufficientInformationException
"""
from __future__ import annotations

import asyncio
import logging

from schema.models import MedicalQuery, MedicalResponse, PatientProfile
from llm.client import AsyncLLMClient
from llm.prompt_templates import EXPERT_SYSTEM_PROMPT, CONSENSUS_PROMPT, CONSENSUS_VERIFICATION_PROMPT, SAFE_FALLBACK_RESPONSE
from engine.react_engine import ReactEngine
from engine.tool_registry import global_tool_registry
from memory.reflection_manager import ReflectionManager, InsufficientInformationException
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import MedicalReranker
from schema.messages import Message
from config import cfg

logger = logging.getLogger(__name__)


class MDTConsultationWorkflow:
    """多专家 MDT 异步编排

    核心流程:
    1. 反思拦截 → 2. 并发会诊 → 3. 共识提炼 → 4. 共识引导检索（检索→重排→验证） → 5. 返回有证据支撑的最终报告
    """

    def __init__(
        self,
        llm: AsyncLLMClient,
        reflection_manager: ReflectionManager,
        retriever: HybridRetriever,
        reranker: MedicalReranker,
        profile: PatientProfile | None = None,
    ):
        self.llm = llm
        self.reflection = reflection_manager
        self.retriever = retriever
        self.reranker = reranker
        self.profile = profile

    async def run(
        self,
        query: MedicalQuery,
        departments: list[str],
        escalation_reason: str = "",
        skill_hints: str = "",
    ) -> MedicalResponse:
        """执行 MDT 会诊

        参数:
            query: 用户查询
            departments: 招募的科室列表
            escalation_reason: 如果是从 Simple RAG 升级而来，携带失败原因
            skill_hints: 从成功经验中提取的可复用技能提示
        """
        logger.info(f"MDT 会诊: departments={departments}")

        # 1. 反思拦截: 检查反思记忆，命中则注入 Hint
        reflection_hint = ""
        try:
            hint = await self.reflection.search_reflection(query.query)
            if hint:
                reflection_hint = hint
                logger.info(f"反思拦截命中: {hint}")
        except Exception as e:
            logger.warning(f"反思检索失败: {e}")

        # 2. 编排器已做预检（空结果检查 + is_insufficient），
        #    MDT 不再重复，专家 Agent 会在 ReAct 循环中自行调用 literature_search

        # 3. 为每个科室创建专家 ReactEngine（共享全局工具注册器）
        experts = []
        for dept in departments:
            system_prompt = EXPERT_SYSTEM_PROMPT.format(
                department=dept,
                reflection_hint=reflection_hint,
            )
            if skill_hints:
                system_prompt += f"\n\n{skill_hints}"
            # 注入患者画像到专家 System Prompt
            if self.profile:
                system_prompt += (
                    f"\n\n患者画像：疾病={self.profile.diseases},"
                    f" 用药={self.profile.medications},"
                    f" 过敏={self.profile.allergies}"
                )
            # 如果是从 Simple RAG 升级而来，注入失败原因
            if escalation_reason:
                system_prompt += f"\n\n⚠️注意：此问题此前经简单检索未能给出可靠回答，原因：{escalation_reason}。请特别注意此问题。"

            expert = ReactEngine(
                llm_client=self.llm,
                tool_registry=global_tool_registry,
                system_prompt=system_prompt,
            )
            experts.append((dept, expert))

        # 4. 并发会诊: asyncio.gather 并发执行多专家 ReAct 循环
        tasks = [
            self._run_expert(dept, expert, query.query)
            for dept, expert in experts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 收集各专家结果，异常结果不进入共识，避免污染 LLM 输入
        successful_opinions = []
        failed_departments: list[str] = []
        for (dept, _), result in zip(experts, results):
            if isinstance(result, Exception):
                logger.error(f"专家 {dept} 会诊失败: {result}")
                failed_departments.append(dept)
            else:
                successful_opinions.append(f"[{dept}]\n{result}")

        # E2: 最少专家成功阈值 —— 全部失败时触发安全退避
        if not successful_opinions:
            logger.warning(f"所有 {len(experts)} 位专家均会诊失败 ({failed_departments})，触发安全退避")
            return MedicalResponse(
                answer=SAFE_FALLBACK_RESPONSE,
                route_path="safe_fallback",
                confidence=0.0,
                departments=departments,
                is_safe_fallback=True,
            )

        if failed_departments:
            logger.warning(f"部分专家失败 ({len(failed_departments)}/{len(experts)}): {failed_departments}，以 {len(successful_opinions)} 位成功专家意见继续")

        # 5. 共识提炼: 仅用成功专家结果，调用 LLM 提炼最终会诊报告
        profile_text = ""
        if self.profile:
            profile_text = f"疾病={self.profile.diseases}, 用药={self.profile.medications}, 过敏={self.profile.allergies}"

        consensus_prompt = CONSENSUS_PROMPT.format(
            expert_opinions="\n\n".join(successful_opinions),
            profile=profile_text or "无",
        )

        consensus_resp = await self.llm.chat(
            messages=[Message(role="user", content=consensus_prompt)],
            temperature=cfg.llm.temperatures["consensus"],
        )
        consensus_text = consensus_resp.content or "会诊未能得出结论"

        # 6. 共识引导检索: 用共识摘要检索知识库，验证共识主张
        sources: list[str] = []
        try:
            consensus_docs = await self.retriever.retrieve(
                consensus_text,
                profile=self.profile,
                top_k=cfg.retrieval.consensus_retrieval_top_k,
            )
            if consensus_docs:
                reranked = await self.reranker.rerank(
                    consensus_text,
                    consensus_docs,
                    top_k=cfg.retrieval.consensus_rerank_top_k,
                )
                if reranked:
                    evidence_lines = []
                    for doc in reranked:
                        evidence_lines.append(f"[{len(evidence_lines) + 1}] {doc.content}")
                        if doc.source:
                            sources.append(doc.source)
                    evidence_text = "\n".join(evidence_lines)
                    profile_text_v = ""
                    if self.profile:
                        profile_text_v = f"疾病={self.profile.diseases}, 用药={self.profile.medications}, 过敏={self.profile.allergies}"
                    verify_prompt = CONSENSUS_VERIFICATION_PROMPT.format(
                        consensus=consensus_text,
                        evidence=evidence_text,
                        profile=profile_text_v or "无",
                    )
                    verified_resp = await self.llm.chat(
                        messages=[Message(role="user", content=verify_prompt)],
                        temperature=cfg.llm.temperatures["consensus"],
                    )
                    final_answer = verified_resp.content or consensus_text
                else:
                    final_answer = consensus_text
            else:
                final_answer = consensus_text
        except Exception as e:
            logger.warning(f"共识引导检索失败，输出未经验证的共识: {e}")
            final_answer = consensus_text + "\n\n⚠️ 注意：此结论的检索验证步骤失败，建议谨慎参考。"

        response = MedicalResponse(
            answer=final_answer,
            route_path="mdt",
            departments=departments,
            sources=sources,
        )

        logger.info("MDT 会诊完成")
        return response

    async def _run_expert(self, department: str, engine: ReactEngine, query: str) -> str:
        """运行单个专家的 ReAct 循环"""
        logger.info(f"启动专家: {department}")
        result = await engine.run(query)
        logger.info(f"专家 {department} 完成, 回答长度: {len(result)}")
        return result