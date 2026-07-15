"""简单 RAG 工作流

设计文档: Simple RAG 分支
流程: 反思拦截 → 混合检索 → 重排去噪 → LLM 生成（带引用）
末尾由编排器执行置信度评估，不通过则升级到 MDT。

支持两种模式:
- run(): 完整执行，返回 (response, docs)
- run_stream(): 流式执行，yield SSE chunks (content + done)
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from schema.models import MedicalQuery, MedicalResponse, DocumentChunk, PatientProfile
from llm.client import AsyncLLMClient
from llm.prompt_templates import SIMPLE_RAG_SYSTEM_PROMPT
from rag.hybrid_retriever import HybridRetriever
from rag.reranker import MedicalReranker
from memory.reflection_manager import ReflectionManager
from schema.messages import Message
from config import cfg

logger = logging.getLogger(__name__)


class SimpleRAGWorkflow:
    """简单 RAG 流程: 检索 → 重排 → 生成"""

    def __init__(
        self,
        llm: AsyncLLMClient,
        retriever: HybridRetriever,
        reranker: MedicalReranker,
        reflection_manager: ReflectionManager | None = None,
    ):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker
        self.reflection = reflection_manager

    async def run(
        self, query: MedicalQuery, profile: PatientProfile | None = None,
        skill_hints: str = "",
    ) -> tuple[MedicalResponse, list[DocumentChunk]]:
        """执行简单 RAG 流程

        返回: (响应, 检索文档列表) —— 文档列表用于后续置信度评估
        """
        logger.info(f"Simple RAG: query='{query.query}'")

        # 0. 反思拦截: 检查反思记忆，命中则注入 Hint 到 System Prompt
        reflection_hint = ""
        if self.reflection:
            try:
                hint = await self.reflection.search_reflection(query.query)
                if hint:
                    reflection_hint = hint
                    logger.info(f"Simple RAG 反思拦截命中: {hint}")
            except Exception as e:
                logger.warning(f"Simple RAG 反思检索失败: {e}")

        # 1. 完整检索（必须用完整 top_k，预检的少量文档不足以做重排判断）
        documents = await self.retriever.retrieve(
            query=query.query,
            profile=profile,
            top_k=cfg.retrieval.top_k,
        )

        # 2. 重排去噪（Medical Reranker 打压临床逻辑无关的噪声）
        reranked = await self.reranker.rerank(query.query, documents, top_k=cfg.retrieval.rerank_top_k)

        # 2.1 CoT 安全退避检查
        if not reranked or self.reranker.is_insufficient(reranked):
            from memory.reflection_manager import InsufficientInformationException
            max_score = reranked[0].score if reranked else 0
            raise InsufficientInformationException(
                f"重排得分极低 (最高 {max_score:.3f})，知识库无可靠相关知识"
            )

        # 3. 构建上下文（带文档编号，供 LLM 引用）
        context = "\n\n".join(
            f"[Source: Doc {i+1}] {doc.content}"
            for i, doc in enumerate(reranked)
        )

        # 4. LLM 生成回答（要求输出引用来源）
        system_prompt = SIMPLE_RAG_SYSTEM_PROMPT
        if reflection_hint:
            system_prompt += f"\n\n{reflection_hint}"
        if skill_hints:
            system_prompt += f"\n\n{skill_hints}"

        resp = await self.llm.chat(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=f"参考文献：\n{context}\n\n问题：{query.query}"),
            ],
        )

        sources = [doc.source for doc in reranked if doc.source]
        response = MedicalResponse(
            answer=resp.content or "无法生成回答",
            route_path="simple_rag",
            sources=sources,
        )

        return response, reranked

    async def run_stream(
        self, query: MedicalQuery, profile: PatientProfile | None = None,
        skill_hints: str = "",
    ) -> AsyncIterator[dict]:
        """流式执行 Simple RAG — 检索+重排后，用 chat_stream 逐 token 输出

        yield 格式: {"type":"status"|"content"|"done"|"error", "text":..., ...}
        """
        yield {"type": "status", "text": "反思拦截中..."}
        reflection_hint = ""
        if self.reflection:
            try:
                hint = await self.reflection.search_reflection(query.query)
                if hint:
                    reflection_hint = hint
                    yield {"type": "status", "text": f"命中反思记忆: {hint[:40]}..."}
            except Exception:
                pass

        yield {"type": "status", "text": "检索中..."}
        documents = await self.retriever.retrieve(
            query=query.query, profile=profile, top_k=cfg.retrieval.top_k,
        )

        yield {"type": "status", "text": f"检索完成({len(documents)}篇)，重排中..."}
        reranked = await self.reranker.rerank(query.query, documents, top_k=cfg.retrieval.rerank_top_k)

        if not reranked or self.reranker.is_insufficient(reranked):
            max_score = reranked[0].score if reranked else 0
            yield {"type": "error", "text": f"知识库无可靠相关知识 (最高分 {max_score:.3f})"}
            return

        context = "\n\n".join(
            f"[Source: Doc {i+1}] {doc.content}" for i, doc in enumerate(reranked)
        )

        system_prompt = SIMPLE_RAG_SYSTEM_PROMPT
        if reflection_hint:
            system_prompt += f"\n\n{reflection_hint}"
        if skill_hints:
            system_prompt += f"\n\n{skill_hints}"

        yield {"type": "status", "text": "LLM 生成中..."}
        collected = []
        try:
            async for chunk in self.llm.chat_stream(
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=f"参考文献：\n{context}\n\n问题：{query.query}"),
                ],
            ):
                if chunk["type"] == "content":
                    collected.append(chunk["text"])
                    yield {"type": "content", "text": chunk["text"]}
        except Exception as e:
            if not collected:
                yield {"type": "error", "text": f"LLM 生成失败: {e}"}
                return

        full_text = "".join(collected)
        sources = [doc.source for doc in reranked if doc.source]
        yield {
            "type": "done",
            "text": full_text,
            "route_path": "simple_rag",
            "confidence": 1.0,
            "sources": sources,
            "documents": [
                {"content": d.content, "source": d.source, "score": d.score}
                for d in reranked
            ],
        }