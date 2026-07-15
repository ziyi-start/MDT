"""FastAPI 主入口 - 医疗多智能体协同 RAG 系统

系统架构（对应设计文档四大核心层）:
1. 交互与动态路由层: 规则拦截 + LLM结构化路由 + 置信度评估 + 携因打回
2. 多专家会诊层: ReAct引擎 + 工具调用 + 共识提炼
3. 记忆与检索协同层: Milvus混合检索 + 画像约束 + Reranker
4. 反思与决策层: Decision Maker + 归因反思 + CoT安全退避

启动方式:
  内存模式: python main.py
  Milvus模式: MDT_USE_MILVUS=true python main.py
  NER服务:   MDT_NER_SERVICE_URL=http://ner:8000/ner python main.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# 确保 backend 目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time as _time

from config import cfg
from llm.client import AsyncLLMClient
from rag.milvus_client import MilvusManager
from rag.hybrid_retriever import HybridRetriever, load_in_memory_kb
from rag.reranker import MedicalReranker
from memory.profile_extractor import ProfileExtractor
from memory.reflection_manager import ReflectionManager
from memory.skill_manager import SkillManager
from workflow.medical_orchestrator import MedicalOrchestrator
from tools.literature_search import set_retriever
from schema.models import MedicalQuery, MedicalResponse
from monitoring.metrics import SessionMetrics, RequestMetrics

# 用户反馈存储
_feedback_store: list[dict] = []
_event_chain_cache: list[dict] = []

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# 内置种子知识库（Milvus 不可用时使用内存模式）
# ============================================================
BUILTIN_KB = [
    {"doc_id": "kb_001", "content": "布洛芬属于非甾体抗炎药(NSAIDs)，具有解热、镇痛、抗炎作用。常见不良反应：胃肠道不适、溃疡出血、肾功能损害。禁忌：活动性消化道溃疡、严重肾功能不全。与氯吡格雷联用增加出血风险。高血压患者慎用，可能影响降压药效果。", "source": "《新编药物学》第18版", "department": "风湿科", "contraindications": "胃溃疡,肾功能不全,出血风险"},
    {"doc_id": "kb_002", "content": "秋水仙碱是痛风急性发作的一线用药。用法：急性期首剂1mg，1小时后0.5mg。不良反应：腹泻、恶心呕吐。注意事项：肾功能不全者需减量，CKD 3期以上禁用常规剂量。", "source": "《痛风诊疗指南》2023版", "department": "风湿科", "contraindications": "肾功能不全,CKD3期"},
    {"doc_id": "kb_003", "content": "对乙酰氨基酚是安全性较高的解热镇痛药，无抗炎作用。与NSAIDs相比，不损伤胃黏膜，不影响肾功能，不增加出血风险。痛风患者如不能使用NSAIDs，对乙酰氨基酚可作为替代选择，但对急性炎症控制效果较弱。", "source": "《中国疼痛医学诊疗指南》", "department": "风湿科", "contraindications": "严重肝功能不全"},
    {"doc_id": "kb_004", "content": "高血压患者用药注意：NSAIDs类药物可引起水钠潴留，降低降压药疗效，升高血压；选择止痛药时应优先考虑对乙酰氨基酚；氯吡格雷与NSAIDs联用显著增加胃肠道出血风险。", "source": "《中国高血压防治指南》2023修订版", "department": "心内科", "contraindications": "NSAIDs,出血风险"},
    {"doc_id": "kb_005", "content": "胃溃疡患者用药禁忌：禁用NSAIDs类药物（布洛芬、阿司匹林、双氯芬酸等），因其抑制胃黏膜前列腺素合成，加重溃疡和出血风险。痛风合并胃溃疡时，优先选择秋水仙碱或对乙酰氨基酚止痛。", "source": "《消化性溃疡诊断与治疗共识》2022版", "department": "消化科", "contraindications": "NSAIDs,阿司匹林,胃出血"},
    {"doc_id": "kb_006", "content": "氯吡格雷联合阿司匹林双抗治疗期间出血风险显著升高，需避免联用NSAIDs类药物。消化道出血高危患者应联合PPI保护。", "source": "《冠心病抗血小板治疗中国专家共识》", "department": "心内科", "contraindications": "活动性出血,NSAIDs"},
    {"doc_id": "kb_007", "content": "糖尿病合并慢性肾病患者的止痛策略：避免使用NSAIDs，因可加重肾损伤并影响血糖控制；对乙酰氨基酚为首选；控制血糖和血压是保护肾功能的基础。", "source": "《糖尿病肾病防治专家共识》2023", "department": "内分泌科", "contraindications": "NSAIDs,肾功能不全"},
    {"doc_id": "kb_008", "content": "痛风急性发作治疗方案：轻中度发作可用秋水仙碱或NSAIDs；合并胃溃疡者避免NSAIDs，使用秋水仙碱+PPI保护或对乙酰氨基酚；合并肾功能不全者避免NSAIDs和常规剂量秋水仙碱，使用糖皮质激素。", "source": "《中国痛风诊疗指南》2023版", "department": "风湿科", "contraindications": "胃溃疡,肾功能不全"},
]

# 全局组件
orchestrator: MedicalOrchestrator | None = None
milvus_connected: bool = False
session_metrics = SessionMetrics()

# Harness 全局组件
from monitoring.tracing import trace_manager
from harness.evaluator import HarnessEvaluator
from harness.experiment import ExperimentTracker
from engine.safety_guard import SafetyGuard, RateLimiter, CostTracker, validate_tool_args
from context.manager import ContextManager

harness_evaluator = HarnessEvaluator()
experiment_tracker = ExperimentTracker()
safety_guard: SafetyGuard | None = None

# MCP 客户端全局实例
mcp_client = None


class QueryRequest(BaseModel):
    """查询请求体"""
    query: str
    user_id: str = cfg.default_user_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 - 启动时初始化所有组件"""
    global orchestrator, milvus_connected

    logger.info("初始化系统组件...")

    # ---- LLM 客户端 ----
    llm = AsyncLLMClient(
        base_url=cfg.llm.base_url,
        api_key=cfg.llm.api_key,
        model=cfg.llm.model,
    )

    # ---- Milvus 向量数据库 ----
    milvus = None
    if cfg.milvus.use_milvus:
        try:
            milvus = MilvusManager(uri=cfg.milvus.uri)
            milvus.connect()
            milvus.init_all_collections()
            milvus_connected = True
            logger.info("Milvus 连接成功")

            # 自动注入种子知识库（首次启动时 Milvus 为空）
            try:
                existing = milvus.search(
                    collection_name=cfg.milvus.collections.kb,
                    vector=[0.0] * cfg.milvus.embedding_dim,
                    limit=1,
                )
                if not existing:
                    logger.info("Milvus 知识库为空，自动注入种子数据...")
                    from seed_data import seed_milvus
                    await seed_milvus(milvus)
                    logger.info("种子数据注入完成")
            except Exception as e:
                logger.warning(f"种子数据注入检查失败: {e}")

        except Exception as e:
            logger.warning(f"Milvus 连接失败: {e}")
            milvus = None

    # ---- 内存模式: 加载内置知识库 ----
    if not milvus:
        logger.info("使用内存模式运行 (无 Milvus)")
        load_in_memory_kb(BUILTIN_KB)
        logger.info(f"加载 {len(BUILTIN_KB)} 条内置知识库")

    # ---- 检索与重排 ----
    retriever = HybridRetriever(milvus)
    reranker = MedicalReranker(
        low_threshold=cfg.reranker.low_threshold,
    )
    set_retriever(retriever, reranker)

    # ---- 记忆模块 ----
    profile_extractor = ProfileExtractor(llm, milvus)
    reflection_manager = ReflectionManager(llm, milvus)
    skill_manager = SkillManager(llm, milvus)

    # ---- Harness: 安全守卫 ----
    global safety_guard
    if cfg.harness.safety.cost_tracking_enabled:
        rate_limiter = RateLimiter(
            max_calls=cfg.harness.safety.rate_limit_max_calls,
            window_seconds=cfg.harness.safety.rate_limit_window_sec,
        )
        cost_tracker = CostTracker()
        safety_guard = SafetyGuard(rate_limiter=rate_limiter, cost_tracker=cost_tracker)
        safety_guard.add_pre_hook(validate_tool_args)
        logger.info("安全守卫初始化完成 (限流+成本追踪)")

    # ---- Harness: 统一上下文管理器 (四层记忆 + 多轮对话 + 上下文窗口) ----
    from context.memory_hierarchy import MemoryHierarchyConfig
    from context.conversation_memory import ConversationConfig
    from context.context_window import ContextWindowConfig
    from context.context_assembler import AssembleConfig

    hierarchy_cfg = MemoryHierarchyConfig(
        working_capacity=cfg.harness.context.permanent_tokens,
        short_term_capacity=cfg.harness.context.working_tokens,
        long_term_capacity=cfg.harness.context.deep_tokens,
        external_capacity=cfg.harness.context.deep_tokens,
        total_capacity=cfg.harness.context.total_tokens,
        short_term_window_size=cfg.harness.context.short_term_window_size,
        importance_threshold=cfg.harness.context.importance_threshold,
        recency_decay_rate=cfg.harness.context.recency_decay_rate,
    )
    conv_cfg = ConversationConfig(
        max_turns=cfg.harness.context.max_turns,
        max_history_tokens=cfg.harness.context.max_history_tokens,
        summary_threshold_turns=cfg.harness.context.summary_threshold_turns,
        topic_drift_threshold=cfg.harness.context.topic_drift_threshold,
    )
    window_cfg = ContextWindowConfig(
        total_budget=cfg.harness.context.total_tokens,
        permanent_budget=cfg.harness.context.permanent_tokens,
        working_budget=cfg.harness.context.working_tokens,
        deep_budget=cfg.harness.context.deep_tokens,
        warning_threshold=cfg.harness.context.warning_threshold,
        critical_threshold=cfg.harness.context.critical_threshold,
        compression_enabled=cfg.harness.context.compression_enabled,
        compression_summary_ratio=cfg.harness.context.compression_summary_ratio,
        response_reserved_tokens=cfg.harness.context.response_reserved_tokens,
    )
    assemble_cfg = AssembleConfig(
        max_system_tokens=cfg.harness.context.permanent_tokens,
        max_history_tokens=cfg.harness.context.max_history_tokens,
        max_total_tokens=cfg.harness.context.total_tokens,
    )
    from context.compaction import CompactionConfig
    compaction_cfg = CompactionConfig(
        max_compacted_tokens=cfg.harness.context.compaction_max_tokens,
    )
    context_manager = ContextManager(
        user_id=cfg.default_user_id,
        hierarchy_config=hierarchy_cfg,
        conversation_config=conv_cfg,
        window_config=window_cfg,
        assemble_config=assemble_cfg,
        compaction_config=compaction_cfg,
    )
    context_manager.begin_session()
    logger.info(f"统一上下文管理器初始化完成 (四层记忆+多轮对话+上下文窗口, total={hierarchy_cfg.total_capacity})")

    # ---- 顶层编排器 (含 Harness 增强 + Agent 自进化) ----
    orchestrator = MedicalOrchestrator(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        profile_extractor=profile_extractor,
        reflection_manager=reflection_manager,
        skill_manager=skill_manager,
        ner_service_url=cfg.services.ner_url,
        enable_tracing=cfg.harness.tracing.enabled,
        safety_guard=safety_guard,
        context_manager=context_manager,
        harness_evaluator=harness_evaluator,
    )

    logger.info("系统初始化完成！(含 Harness: 追踪+评估+安全守卫+上下文预算)")
    # ---- MCP 客户端初始化 ----
    global mcp_client
    try:
        from engine.mcp_client import MCPClient, MCPServerConfig
        from engine.tool_registry import global_tool_registry
        mcp_client = MCPClient()

        def _make_mcp_wrapper(server_name: str, tool_name: str):
            async def _wrapper(**kwargs):
                return await mcp_client.call_tool(server_name, tool_name, kwargs)
            return _wrapper

        mcp_configs = [
            MCPServerConfig(name="demo", command="python", args=["-c", "print('MCP demo ready')"], enabled=False),
        ]
        for mc in mcp_configs:
            if mc.enabled and mc.command:
                if mc.url:
                    await mcp_client.connect_sse(mc.name, mc.url)
                else:
                    await mcp_client.connect_stdio(mc.name, mc.command, mc.args)
        if mcp_client.stats()["connected_servers"] > 0:
            logger.info(f"MCP 客户端初始化完成: {mcp_client.stats()}")
            for tool in await mcp_client.list_all_tools():
                reg_name = f"mcp_{tool.server_name}_{tool.name}"
                global_tool_registry._tools[reg_name] = {
                    "schema": {
                        "name": reg_name,
                        "description": f"[MCP:{tool.server_name}] {tool.description}",
                        "parameters": tool.parameters or {"type": "object", "properties": {}},
                    },
                    "fn": _make_mcp_wrapper(tool.server_name, tool.name),
                }
            logger.info(f"MCP 工具已注册到 ToolRegistry ({mcp_client.stats()['total_tools']} tools)")
        else:
            logger.info("MCP 客户端就绪 (无活跃连接)")
    except Exception as e:
        logger.warning(f"MCP 客户端初始化跳过: {e}")
    yield
    logger.info("系统关闭")


app = FastAPI(title="医疗多智能体协同 RAG 系统", version=cfg.server.app_version, lifespan=lifespan)


@app.post("/api/query")
async def query(req: QueryRequest):
    """处理医疗查询 - REST API 入口"""
    if orchestrator is None:
        return {"error": "系统尚未初始化"}

    t_start = _time.perf_counter()
    logger.info(f"收到查询: '{req.query}' (user={req.user_id})")
    medical_q = MedicalQuery(query=req.query, user_id=req.user_id)
    response = await orchestrator.process(medical_q)
    latency_ms = (_time.perf_counter() - t_start) * 1000

    # 更新监控指标
    session_metrics.add(RequestMetrics(
        query=req.query,
        route_path=response.route_path,
        total_latency_ms=latency_ms,
        final_confidence=response.confidence,
        is_safe_fallback=response.is_safe_fallback,
    ))

    return {
        "answer": response.answer,
        "route_path": response.route_path,
        "departments": response.departments,
        "sources": response.sources,
        "confidence": response.confidence,
        "is_safe_fallback": response.is_safe_fallback,
        "latency_ms": round(latency_ms, 2),
    }


@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "ok", "milvus": milvus_connected}


@app.get("/api/metrics")
async def metrics():
    """RAG 监控指标"""
    return {
        "session": session_metrics.to_dict(),
        "milvus_connected": milvus_connected,
        "harness": {
            "tracing": trace_manager.stats,
            "safety": safety_guard.cost_tracker.to_dict() if safety_guard else {"status": "disabled"},
        },
    }


# ============================================================
# Harness API 端点
# ============================================================

@app.get("/api/harness/traces")
async def list_traces(n: int = 10):
    """获取最近的追踪记录"""
    return {"traces": trace_manager.get_recent(n)}


@app.get("/api/harness/traces/{trace_id}")
async def get_trace(trace_id: str):
    """获取特定追踪详情"""
    trace = trace_manager.get_active(trace_id)
    if trace:
        return trace.to_dict()
    for t in trace_manager._completed_traces[-100:]:
        if t.trace_id == trace_id:
            return t.to_dict()
    return {"error": "trace not found"}


@app.get("/api/harness/evaluate")
async def evaluate():
    """执行 7维 Harness 评估（基于当前会话指标）"""
    if harness_evaluator is None:
        return {"error": "评估器未初始化"}

    from schema.models import MedicalResponse
    # 构造模拟响应用于评估
    recent = trace_manager.get_recent(50)
    if not recent:
        return {"error": "没有足够的请求数据用于评估"}

    for t in recent:
        resp = MedicalResponse(
            answer="评估样本",
            route_path="simple_rag",
            confidence=0.8,
            sources=["sample"],
        )
        harness_evaluator.add_result(resp, t.get("total_ms", 1000))

    report = harness_evaluator.evaluate()
    return report.to_dict()


@app.get("/api/harness/experiments")
async def list_experiments():
    """列出所有实验记录"""
    return {"experiments": experiment_tracker.list_experiments()}


@app.get("/api/harness/safety")
async def safety_stats():
    """安全守卫统计"""
    if safety_guard is None:
        return {"status": "disabled", "message": "安全守卫未启用"}
    return {
        "status": "enabled",
        "cost_tracking": safety_guard.cost_tracker.to_dict(),
        "rate_limiter": {
            "max_calls": cfg.harness.safety.rate_limit_max_calls,
            "window_sec": cfg.harness.safety.rate_limit_window_sec,
        },
    }


# ============================================================
# 用户反馈 API — 支持显式反馈环 + 事件记忆检索
# ============================================================

@app.post("/api/feedback")
async def submit_feedback(
    trace_id: str = "",
    query: str = "",
    rating: float = 0.0,
    comment: str = "",
    feedback_type: str = "rating",
):
    """提交用户反馈

    rating: 0.0(完全无用) ~ 1.0(非常满意)
    feedback_type: "rating" | "correction" | "follow_up"
    """
    import time
    feedback = {
        "trace_id": trace_id,
        "query": query,
        "rating": rating,
        "comment": comment,
        "feedback_type": feedback_type,
        "timestamp": time.time(),
    }
    _feedback_store.append(feedback)
    if len(_feedback_store) > 1000:
        _feedback_store.pop(0)

    # 高评分反馈可触发技能强化
    if rating >= 0.8 and orchestrator and orchestrator.skill_manager:
        try:
            mq = MedicalQuery(query=query, user_id=cfg.default_user_id)
            await orchestrator._maybe_extract_skill(
                MedicalResponse(
                    answer=f"[用户评价 {rating:.1f}] {comment}",
                    route_path="feedback",
                    confidence=rating,
                ),
                mq,
            )
        except Exception:
            pass

    logger.info(f"用户反馈: rating={rating}, type={feedback_type}, query={query[:50]}")
    return {"status": "ok", "feedback_count": len(_feedback_store)}


@app.get("/api/feedback")
async def get_feedback(n: int = 20):
    """获取最近的用户反馈"""
    return {"feedback": _feedback_store[-n:]}


@app.get("/api/harness/events")
async def list_events(n: int = 20):
    """获取最近的事件记忆"""
    if orchestrator is None:
        return {"error": "系统尚未初始化"}
    events = orchestrator.event_memory.events[-n:]
    return {
        "events": [
            {
                "event_id": e.event_id,
                "type": e.event_type.name,
                "summary": e.summary,
                "timestamp": e.timestamp,
                "surprise_score": round(e.surprise_score, 3),
            }
            for e in events
        ],
        "stats": orchestrator.event_memory.stats(),
    }


@app.get("/api/harness/runs")
async def list_runs(n: int = 10):
    """获取 Run-level 执行记录"""
    if orchestrator is None:
        return {"error": "系统尚未初始化"}
    return {"runs": orchestrator.run_memory.get_completed_runs(n)}


@app.get("/api/harness/context/snapshot")
async def context_snapshot():
    """获取当前上下文管理器快照"""
    if orchestrator is None or orchestrator.context_manager is None:
        return {"error": "上下文管理器未初始化"}
    return orchestrator.context_manager.snapshot()


# ============================================================
# SSE 流式查询 — P1: 流式输出
# ============================================================

@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    """流式查询 — SSE 协议，带心跳保活

    每个事件格式: data: {"type":"content"|"status"|"done"|"error", "text":"..."}
    长时间处理期间每秒发送 heartbeat 防止超时断开。
    """
    import json as _json

    async def generate():
        if orchestrator is None:
            yield f"data: {_json.dumps({'type': 'error', 'text': 'System not initialized'})}\n\n"
            return

        stop_heartbeat = False

        async def heartbeat():
            while not stop_heartbeat:
                await asyncio.sleep(1)
                if not stop_heartbeat:
                    yield f"data: {_json.dumps({'type': 'status', 'text': '处理中...'})}\n\n"

        yield f"data: {_json.dumps({'type': 'status', 'text': '开始处理查询...'})}\n\n"

        medical_q = MedicalQuery(query=req.query, user_id=req.user_id)

        try:
            async def run_process():
                return await orchestrator.process(medical_q)

            process_task = asyncio.create_task(run_process())

            last_status = ""
            while not process_task.done():
                snap = orchestrator.context_manager.snapshot() if orchestrator.context_manager else {}
                status = f"路由中... (memories={snap.get('memories', {}).get('working_count', 0)})"
                if status != last_status:
                    last_status = status
                    yield f"data: {_json.dumps({'type': 'status', 'text': status})}\n\n"
                await asyncio.sleep(2)

            response = process_task.result()
            stop_heartbeat = True

            yield f"data: {_json.dumps({'type': 'status', 'text': f'路由: {response.route_path}, 置信度: {response.confidence}'})}\n\n"

            answer = response.answer or ""
            chunk_size = 80
            for i in range(0, len(answer), chunk_size):
                chunk = answer[i:i + chunk_size]
                yield f"data: {_json.dumps({'type': 'content', 'text': chunk})}\n\n"
                await asyncio.sleep(0.02)

            yield f"data: {_json.dumps({'type': 'done', 'text': answer, 'route_path': response.route_path, 'confidence': response.confidence, 'departments': response.departments, 'sources': response.sources})}\n\n"

        except Exception as e:
            stop_heartbeat = True
            logger.error(f"Stream query error: {e}")
            yield f"data: {_json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================================
# MCP 管理 API — P2: Model Context Protocol 集成
# ============================================================

@app.get("/api/mcp/status")
async def mcp_status():
    """MCP 客户端状态"""
    if mcp_client is None:
        return {"status": "not_initialized", "servers": [], "tools": []}
    return {"status": "connected" if mcp_client.stats()["connected_servers"] > 0 else "idle", **mcp_client.stats()}


@app.post("/api/mcp/connect")
async def mcp_connect(name: str = "", command: str = "", url: str = ""):
    """连接 MCP Server"""
    if mcp_client is None:
        return {"error": "MCP client not initialized"}
    if not name:
        return {"error": "name is required"}
    if url:
        ok = await mcp_client.connect_sse(name, url)
    elif command:
        ok = await mcp_client.connect_stdio(name, command)
    else:
        return {"error": "command or url is required"}
    if ok:
        tools = await mcp_client.list_tools(name)
        return {"status": "connected", "tools_count": len(tools)}
    return {"error": "connection failed"}


@app.post("/api/mcp/disconnect")
async def mcp_disconnect(name: str = ""):
    """断开 MCP Server"""
    if mcp_client is None:
        return {"error": "MCP client not initialized"}
    await mcp_client.disconnect(name)
    return {"status": "disconnected"}


@app.get("/api/mcp/tools")
async def mcp_tools(server: str = ""):
    """列出 MCP 工具"""
    if mcp_client is None:
        return {"tools": []}
    if server:
        tools = await mcp_client.list_tools(server)
    else:
        tools = await mcp_client.list_all_tools()
    return {"tools": [{"name": t.name, "description": t.description, "server": t.server_name} for t in tools]}


@app.websocket("/ws/query")
async def ws_query(websocket: WebSocket):
    """WebSocket 实时查询入口"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if orchestrator is None:
                await websocket.send_json({"error": "系统尚未初始化"})
                continue
            req = json.loads(data)
            medical_q = MedicalQuery(
                query=req.get("query", ""),
                user_id=req.get("user_id", cfg.default_user_id),
            )
            response = await orchestrator.process(medical_q)
            await websocket.send_json({
                "answer": response.answer,
                "route_path": response.route_path,
                "departments": response.departments,
                "confidence": response.confidence,
            })
    except WebSocketDisconnect:
        logger.info("WebSocket 断开")


# 静态文件服务
frontend_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
async def index():
    """前端首页"""
    frontend_file = os.path.join(frontend_path, "index.html")
    if os.path.exists(frontend_file):
        return FileResponse(frontend_file)
    return {"message": "医疗多智能体协同 RAG 系统 API", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=cfg.server.host, port=cfg.server.port, reload=cfg.server.reload)