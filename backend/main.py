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

import json
import logging
import sys
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# 确保 backend 目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import cfg
from llm.client import AsyncLLMClient
from rag.milvus_client import MilvusManager
from rag.hybrid_retriever import HybridRetriever, load_in_memory_kb
from rag.reranker import MedicalReranker
from memory.profile_extractor import ProfileExtractor
from memory.reflection_manager import ReflectionManager
from workflow.medical_orchestrator import MedicalOrchestrator
from tools.literature_search import set_retriever
from schema.models import MedicalQuery

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

    # ---- 顶层编排器 ----
    orchestrator = MedicalOrchestrator(
        llm=llm,
        retriever=retriever,
        reranker=reranker,
        profile_extractor=profile_extractor,
        reflection_manager=reflection_manager,
        ner_service_url=cfg.services.ner_url,
    )

    logger.info("系统初始化完成！")
    yield
    logger.info("系统关闭")


app = FastAPI(title="医疗多智能体协同 RAG 系统", version=cfg.server.app_version, lifespan=lifespan)


@app.post("/api/query")
async def query(req: QueryRequest):
    """处理医疗查询 - REST API 入口"""
    if orchestrator is None:
        return {"error": "系统尚未初始化"}

    logger.info(f"收到查询: '{req.query}' (user={req.user_id})")
    medical_q = MedicalQuery(query=req.query, user_id=req.user_id)
    response = await orchestrator.process(medical_q)

    return {
        "answer": response.answer,
        "route_path": response.route_path,
        "departments": response.departments,
        "sources": response.sources,
        "confidence": response.confidence,
        "is_safe_fallback": response.is_safe_fallback,
    }


@app.get("/api/health")
async def health():
    """健康检查"""
    return {"status": "ok", "milvus": milvus_connected}


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