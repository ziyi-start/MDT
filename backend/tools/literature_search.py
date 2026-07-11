"""文献检索工具 - Agent 可调用的文献搜索工具

注册为 LLM 可调用的工具，专家 Agent 在 ReAct 循环中主动调用此工具检索文献。
设计文档要求专家"主动构思专业的检索词去调用工具，取代用户原始 Query 进行检索"。
"""
from __future__ import annotations

import json
import logging
from contextvars import ContextVar

from engine.tool_registry import global_tool_registry
from rag.hybrid_retriever import HybridRetriever, load_in_memory_kb
from rag.reranker import MedicalReranker
from schema.models import PatientProfile
from config import cfg

logger = logging.getLogger(__name__)

# 全局组件实例（在 main.py 中通过 set_retriever 注入）
_retriever: HybridRetriever | None = None
_reranker: MedicalReranker | None = None

# 请求级患者画像，使用 ContextVar 实现并发安全
_current_profile: ContextVar[PatientProfile | None] = ContextVar("current_profile", default=None)


def set_retriever(retriever: HybridRetriever, reranker: MedicalReranker | None = None):
    """注入全局检索器和重排器实例（由 main.py 在启动时调用）"""
    global _retriever, _reranker
    _retriever = retriever
    _reranker = reranker


def set_current_profile(profile: PatientProfile | None):
    """设置当前请求的患者画像（请求级隔离，并发安全）"""
    _current_profile.set(profile)


@global_tool_registry.register(
    name="literature_search",
    description="检索医学文献数据库，获取与查询相关的医学知识。根据患者情况主动构思专业检索词。",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "专业检索词，应结合患者禁忌和病史改写，而非直接使用患者原话",
            },
            "department": {
                "type": "string",
                "description": "限定检索的科室范围，如'心内科'、'消化科'",
            },
        },
        "required": ["query"],
    },
)
async def literature_search(query: str, department: str = "") -> str:
    """检索医学文献

    参数:
        query: 专家主动构建的专业检索词
        department: 可选，限定科室范围

    返回:
        JSON 格式的检索结果
    """
    if _retriever is None:
        return json.dumps({"error": "检索器未初始化"}, ensure_ascii=False)

    try:
        # 传入当前请求的患者画像，使 Agent 调用检索时也能享受画像约束
        results = await _retriever.retrieve(
            query,
            profile=_current_profile.get(),
            top_k=cfg.retrieval.literature_search_top_k,
            departments=[department] if department else None,
        )
        if not results:
            return json.dumps({"result": "未检索到相关文献", "documents": []}, ensure_ascii=False)

        # 重排去噪：MDT 专家检索的文献也必须过 reranker
        if _reranker:
            results = await _reranker.rerank(query, results, top_k=cfg.retrieval.rerank_top_k)

        # 如果指定了科室，过滤非目标科室的文档
        # 严格过滤：未匹配时不回退到全部结果，避免无关科室文档混入
        if department:
            filtered = [d for d in results if department in d.metadata.get("department", "")]
            if filtered:
                results = filtered
            else:
                logger.warning(f"科室 '{department}' 无匹配文档，返回全部结果并标记")
                for d in results:
                    d.metadata["department_unmatched"] = True

        docs = [
            {"id": d.doc_id, "content": d.content[:500], "source": d.source, "score": round(d.score, 3)}
            for d in results
        ]
        return json.dumps(
            {"result": f"检索到 {len(docs)} 篇相关文献", "documents": docs},
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"文献检索失败: {e}")
        return json.dumps({"error": str(e)}, ensure_ascii=False)