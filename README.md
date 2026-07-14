<h1 align="center">
  <br>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/FastAPI-0.111+-009688?style=flat&logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/Milvus-2.5.7-00A1EA?style=flat&logo=milvus&logoColor=white">
  <img src="https://img.shields.io/badge/DeepSeek-chat-4B4B4B?style=flat">
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat">
  <br><br>
  MDT — 医疗多智能体协同 RAG 系统
</h1>

<h3 align="center">
  多科室专家 ReAct 会诊 &nbsp;|&nbsp; Hybrid RAG 混合检索 &nbsp;|&nbsp; 四库分层记忆 &nbsp;|&nbsp; Agent 自进化
</h3>

<p align="center">
  基于 <b>Agent = Model + Harness</b> 理念构建的新一代医疗知识问答平台
</p>

<br>

---

## 目录

- [项目背景](#项目背景)
- [系统架构](#系统架构)
- [核心创新点](#核心创新点)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [API 参考](#api-参考)
- [配置管理](#配置管理)
- [数据与语料](#数据与语料)
- [评估体系](#评估体系)
- [Harness 工程框架](#harness-工程框架)
- [架构拓扑图](#架构拓扑图)

---

## 项目背景

临床诊疗中的多学科会诊（Multi-Disciplinary Team, MDT）是处理复杂疑难病例的标准范式。本系统以**大语言模型 + 检索增强生成（RAG）**为底座，将 MDT 诊治理念映射为**多智能体协同推理架构**——每个科室专家以独立 Agent 角色运行 ReAct 推理循环，自主调用工具检索循证文献与药物冲突库，最终通过共识提炼与决策器评估，输出高质量、可溯源的医疗建议。

系统严格遵循三项核心设计原则：

| 原则 | 说明 |
|------|------|
| **安全优先** | 知识库无支撑文献时触发 CoT 安全退避，宁可拒答绝不幻觉 |
| **可观测可评估** | 全链路 TraceID 传播、7 维确定性评分，每次迭代有据可查 |
| **持续自进化** | 从失败中沉淀反思记忆，从成功中抽取可复用技能 |

---

## 系统架构

系统将端到端医疗问答流程划分为 **四个核心层 + 一个工程外壳（Harness）**：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            HARNESS 工程外壳                                  │
│                   追踪观测 · 7维评分 · 安全守卫 · 上下文预算                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ① 交互与动态路由层                                                          │
│     ┌──────────┐     ┌──────────────┐     ┌──────────────────┐              │
│     │ 规则拦截  │────▶│ LLM 结构化路由 │────▶│ 置信度 + 携因打回 │              │
│     │ (NER+正则)│     │ (Guided JSON)│     │ (RouteEscalation)│              │
│     └──────────┘     └──────────────┘     └──────────────────┘              │
│           │                   │                      │                       │
│           ▼                   ▼                      ▼                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ② 多专家会诊层                                                              │
│     ┌──────────┐     ┌──────────┐     ┌──────────┐                          │
│     │ 心内科   │     │ 消化科   │     │ 风湿科   │   ⋯ 并发 ReAct 推理       │
│     │ ReAct    │     │ ReAct    │     │ ReAct    │                          │
│     │  Agent   │     │  Agent   │     │  Agent   │                          │
│     └────┬─────┘     └────┬─────┘     └────┬─────┘                          │
│          └───────────┬───────────────────┘                                  │
│                      ▼                                                       │
│              共识提炼 + 共识引导检索                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ③ 记忆与检索协同层                                                          │
│     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐     │
│     │ Medical_KB   │  │Patient_Profile│  │Reflection_Mem│  │Skill_Mem │     │
│     │ (知识库)     │  │ (患者画像)    │  │ (反思记忆)   │  │ (技能记忆)│     │
│     │              │  │              │  │              │  │          │     │
│     │ BM25 + Dense │  │ 硬约束+软约束 │  │ 归因三元组   │  │ add/merge│     │
│     │ RRF 融合     │  │ 渐进式更新   │  │ 失败→避坑    │  │ /discard │     │
│     └──────────────┘  └──────────────┘  └──────────────┘  └──────────┘     │
│                      ┌──────────────┐                                       │
│                      │ BGE-Reranker │  Cross-Encoder 重排 + CoT 阈值退避    │
│                      └──────────────┘                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ④ 反思与决策层                                                              │
│     ┌─────────────┐     ┌───────────────┐     ┌────────────────┐           │
│     │DecisionMaker│────▶│  归因反思写入   │────▶│ CoT 安全退避输出 │           │
│     │ (质量+幻觉) │     │ Reflection_Mem│     │ (硬编码降级回复) │           │
│     └─────────────┘     └───────────────┘     └────────────────┘           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心创新点

### 1. 闭环动态两级路由

```
用户查询 ──▶ 规则拦截 (NER识别 + 20+科室正则) ──▶ 命中? ──▶ 直接路由 ✓ (<50ms)
                │
                ▼ (未命中)
           LLM 结构化路由 (Guided JSON)
                │
                ├──▶ simple_rag ──▶ 单一检索 + 单一 LLM 生成
                └──▶ mdt        ──▶ 反射拦截 → 多科室并发 → 共识提炼
```

- **规则拦截（快车道）**：基于命名实体识别（NER）+ 科室关键词正则，毫秒级命中科室映射表
- **LLM 路由（慢车道）**：结构化 JSON 输出，自动决定 `route_path` 和 `departments` 征召列表
- **置信度回调**：路由后对 Simple RAG 结果做置信度评估，低于阈值自动**携因打回升级**至 MDT 流程

### 2. 多专家 ReAct 并发会诊

每个科室专家以独立 **ReAct Agent** 运行推理循环：

```
LLM 推理 → 自主决策调用工具 → 文献检索 / 药物冲突查询 → 结果注入 → 继续推理 → stop
                              ↑                              │
                              └──────── 最多 5 轮迭代 ────────┘
```

专家可自主调用的工具集：

| 工具 | 说明 |
|------|------|
| `literature_search` | Agent 自主构建专业检索词，从 Medical_KB 中检索循证文献 |
| `drug_interaction_check` | 查询内置 10+ 种药物冲突知识库，检测交互禁忌 |

并发执行完成后，通过 **共识提炼 Prompt** 汇总各专家意见，生成结构化共识报告，再以共识内容为引导执行二次精确检索。

### 3. Hybrid RAG 混合检索

```
查询 ──▶ 软约束改写 ──▶ ┌── Dense 向量检索 (语义匹配)
                        └── BM25 稀疏检索 (精准药名匹配)
                              │
                              ▼
                        RRF 融合 (手写 Reciprocal Rank Fusion)
                              │
                              ▼
                        BGE-Cross-Encoder 重排
                              │
                              ▼
                        最高分 < 0.2 ──▶ CoT 安全退避
```

- **画像约束**：硬约束（Milvus Boolean filter 过滤禁忌疾病的文档）+ 软约束（查询改写注入禁忌信息）
- **RRF 融合参数**：k=60，标准 OpenSearch 推荐值
- **过采样策略**：先检索 top_k × 2 再过滤，平衡召回与精度

### 4. 四库分层记忆

| 集合 | 主键 | 功能 | 数据流 |
|------|------|------|--------|
| **Medical_KB** | doc_id | 医学知识库，BM25 + Dense 混合索引 | 教科书 → 分段 → 向量化 → 入库 |
| **Patient_Profile** | user_id | 渐进式患者画像，硬约束 + 软约束 | 每轮对话 LLM 抽取 → upsert |
| **Reflection_Mem** | triple_id | 归因反思三元组（intent / cause / avoid_action） | 失败回答 → 反思写入 → 下次拦截预警 |
| **Skill_Mem** | skill_id | 可复用技能（intent / action / department / version） | 成功回答 → 技能提取 → add / merge / discard |

### 5. Agent 自进化 — 从成功中学习

```
成功回答 ──▶ LLM 提取技能 (intent + action + departments)
                │
                ▼
    检索 Skill_Mem 中相似技能 (cosine ≥ 0.75)
                │
    ┌───────────┼───────────┐
    ▼           ▼           ▼
  无相似     版本不同    完全相同
    │           │           │
  add ✓     merge ✓    discard ✗
    │           │
    └─────┬─────┘
          ▼
    写入 Skill_Mem (version + usage_count + provenance)
```

- 技能版本管理：每次 merge 递增 version，记录 provenance（来源追溯）
- 使用统计：每次命中后更新 usage_count，低频技能自动降级
- 提取阈值：仅置信度 ≥ 0.7 的回答触发提取以避免噪声注入

### 6. CoT 三层安全退避

| 层级 | 触发条件 | 动作 |
|------|----------|------|
| **检索层** | Reranker 最高分 < 0.2 | 判定知识库无相关文献，直接返回安全回复 |
| **决策层** | DecisionMaker.quality_score < 0.5 | 拒绝通过，触发反思后返回安全回复 |
| **兜底层** | 异常/超时/幻觉风险高 | 硬编码安全降级回复，提示就医 |

---

## 技术栈

| 层 | 组件 | 选型 | 理由 |
|-----|------|------|------|
| **LLM** | 推理引擎 | DeepSeek-chat / v4-flash / v4-pro | 128K 上下文、OpenAI 兼容、医疗垂域性能 |
| **Embedding** | 语义编码 | BAAI/bge-small-zh-v1.5 | 512d 小尺寸、C-MTEB 中文 SOTA、低延迟 |
| **Cross-Encoder** | 精准重排 | BAAI/bge-reranker-v2-m3 | 多语言、医疗文本对齐优秀 |
| **向量数据库** | 存储 + 检索 | Milvus 2.5.7 Standalone | 原生 hybrid_search（Dense + BM25）、四集合管理 |
| **Web 框架** | API 服务 | FastAPI + Uvicorn | 异步原生、OpenAPI 自动生成、WebSocket 支持 |
| **数据校验** | 类型安全 | Pydantic v2 | 全链路数据模型约束、Rust 核心高性能 |
| **容器编排** | 环境管理 | Docker Compose v3.5 | Milvus + etcd + MinIO 一键部署 |
| **基础设施** | 对象存储 + 元数据 | MinIO + etcd | Milvus Standalone 依赖组件 |

### 核心依赖

```text
openai >= 1.30.0          # LLM 客户端（OpenAI 兼容协议）
pymilvus >= 2.4.0         # Milvus 向量数据库 SDK
pydantic >= 2.0.0         # 数据模型与配置校验
fastapi >= 0.111.0        # REST API + WebSocket
uvicorn[standard] >= 0.30  # ASGI 服务器
httpx >= 0.27.0           # 异步 HTTP 客户端
python-dotenv >= 1.0.0    # 环境变量加载
pyyaml >= 6.0             # YAML 配置解析
sentence-transformers >= 3.0.0  # Embedding 模型加载与推理
```

---

## 项目结构

```
MDT/
│
├── backend/                         # ────── 后端核心 ──────
│   ├── main.py                      #   FastAPI 入口：lifespan 组件初始化 + 全部 API 端点 (357行)
│   ├── config.py                    #   集中配置管理：dataclass + YAML + 环境变量多层加载
│   ├── seed_data.py                 #   Milvus 种子数据注入（8条预计算 512d 向量）
│   ├── seed_vectors.json            #   预计算向量文件
│   │
│   ├── schema/                      #   📦 数据模型层
│   │   ├── models.py                #     MedicalQuery / MedicalResponse / PatientProfile / Skill / ReflectionTriple
│   │   └── messages.py              #     Message / ToolCall (OpenAI 兼容消息格式)
│   │
│   ├── llm/                         #   🤖 LLM 客户端层
│   │   ├── client.py                #     AsyncLLMClient: OpenAI SDK 封装 + tool_calls + JSON constrained
│   │   └── prompt_templates.py      #     全部 Prompt 模板集中管理
│   │
│   ├── engine/                      #   ⚙️ ReAct 执行引擎 + Harness 安全外壳
│   │   ├── react_engine.py          #     手写 ReAct 循环（推理→工具调用→结果注入→继续推理）
│   │   ├── tool_registry.py         #     全局工具注册器（装饰器注册 + schema 生成 + 异步执行）
│   │   └── safety_guard.py          #     [Harness] 限流 + 验证 + 成本追踪
│   │
│   ├── router/                      #   🧭 动态路由层
│   │   ├── rule_interceptor.py      #     NER + 正则拦截（快车道，含 20+ 科室映射表）
│   │   ├── llm_router.py            #     LLM 结构化路由（慢车道，Guided JSON）
│   │   └── confidence_checker.py    #     置信度评估（文档一致性 + 生成自验证 + 携因打回升级）
│   │
│   ├── rag/                         #   🔍 检索增强生成层
│   │   ├── milvus_client.py         #     Milvus 四集合管理 + hybrid_search
│   │   ├── embedding.py             #     BGE-small-zh-v1.5 编码器（512d, L2归一化, LRU缓存）
│   │   ├── hybrid_retriever.py      #     Dense + BM25 + RRF 融合 + 画像约束 (245行)
│   │   └── reranker.py              #     BGE-Cross-Encoder Reranker + n-gram 后备 + CoT 退避
│   │
│   ├── memory/                      #   🧠 记忆模块
│   │   ├── profile_extractor.py     #     渐进式患者画像抽取 (LLM → upsert Patient_Profile)
│   │   ├── reflection_manager.py    #     归因反思管理 (失败教训 → Reflection_Mem)
│   │   └── skill_manager.py         #     [Agent自进化] 技能提取/存储/add-merge-discard/版本管理 (359行)
│   │
│   ├── tools/                       #   🔧 Agent 可调用工具
│   │   ├── literature_search.py     #     文献检索工具 (Agent 自主构建专业检索词)
│   │   └── drug_interaction.py      #     药物冲突查询工具 (内置 10+ 种药物冲突知识库)
│   │
│   ├── workflow/                    #   📋 工作流编排层
│   │   ├── medical_orchestrator.py  #     顶层闭环编排器 + DecisionMaker (459行, 含 Harness 集成)
│   │   ├── simple_rag.py            #     Simple RAG 流程 (反思拦截→检索→重排→LLM生成)
│   │   └── mdt_consultation.py      #     MDT 会诊流程 (并发专家ReAct→共识提炼→共识引导检索)
│   │
│   ├── monitoring/                  #   📊 监控与评估
│   │   ├── metrics.py               #     PipelineTimer + RequestMetrics + SessionMetrics
│   │   ├── tracing.py               #     [Harness] TraceID 传播 + Span 树 + 执行图
│   │   └── rag_evaluator.py         #     离线评估 (MRR/Hit@k + Accuracy/Faithfulness)
│   │
│   ├── harness/                     #   ⛑️ [Harness] 工程框架
│   │   ├── evaluator.py             #     7 维确定性评分引擎 (403行, 零 LLM 调用)
│   │   └── experiment.py            #     A/B 实验框架
│   │
│   └── context/                     #   📐 [Harness] 上下文预算
│       └── manager.py               #     三层上下文预算 (Permanent/Working/Deep)
│
├── config/
│   └── default.yaml                 #   ────── 默认配置 (135行, 全部参数 + 注释) ──────
│
├── frontend/
│   └── index.html                   #   ────── 单页交互式 UI (383行, 示例问题 + 状态栏) ──────
│
├── scripts/                         #   ────── 运维脚本 ──────
│   ├── evaluate_rag.py              #     端到端评测 (MedQA test.jsonl, 支持断点续评, 634行)
│   ├── import_textbooks.py          #     英文教科书 Milvus 导入
│   ├── import_zh_textbooks.py       #     中文教科书向量导入
│   ├── precompute_vectors.py        #     预计算种子向量
│   ├── reseed_milvus.py             #     重建种子数据
│   └── test_rerank.py               #     重排器测试
│
├── data/                            #   ────── 语料与测试集 ──────
│   └── textbooks/                   #     英文教科书 125,847 条 + 中文 116,216 条
│       └── zh_raw/data_clean/
│           └── questions/Mainland/test.jsonl  # 测试集 3,426 题
│
├── docker-compose.yml               #   ────── 容器编排 (etcd + minio + milvus + attu) ──────
├── .env.example                     #   ────── 环境变量模板 ──────
├── config.md                        #   ────── 配置文档 ──────
└── README.md
```

---

## 快速开始

### 前置要求

- **Python** 3.10+（推荐 3.11）
- **Docker** 24.0+ + **Docker Compose** v2+

### 1. 克隆项目

```bash
git clone <repo-url>
cd MDT
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入 `MDT_LLM_API_KEY`：

```env
MDT_LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
MDT_LLM_BASE_URL=https://api.deepseek.com
MDT_LLM_MODEL=deepseek-chat
MDT_USE_MILVUS=true
```

### 3. 启动 Milvus 基础设施

```bash
docker compose up -d
```

服务清单：

| 服务 | 端口 | 说明 |
|------|------|------|
| `milvus` | 19530 | Milvus Standalone 向量数据库 |
| `etcd` | 2379 | 元数据协调服务 |
| `minio` | 9000 | 对象存储后端 |
| `attu` | 8001 | Milvus Web UI 管理面板 |

### 4. 安装 Python 依赖

```bash
cd backend
pip install -r requirements.txt
```

### 5. 启动应用

```bash
# 内存模式（无需 Milvus，使用内置 8 条种子知识库快速体验）
python main.py

# 完整模式（连接 Milvus，支持全量教科书检索 + 四库记忆）
MDT_USE_MILVUS=true python main.py
```

### 6. 验证

```bash
# 健康检查
curl http://localhost:8000/api/health

# 发送测试查询
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "高血压患者痛风发作能用布洛芬吗？", "user_id": "test_user"}'
```

浏览器访问：
- 前端 UI：http://localhost:8000
- Swagger 文档：http://localhost:8000/docs
- Milvus 管理：http://localhost:8001

---

## API 参考

### 核心接口

#### `POST /api/query` — 医疗问答

**请求体：**

```json
{
  "query": "患者有高血压和胃溃疡，最近痛风发作，能吃布洛芬吗？",
  "user_id": "default_user"
}
```

**响应体：**

```json
{
  "answer": "根据多科室会诊意见，该患者不宜使用布洛芬...",
  "route_path": "mdt",
  "departments": ["心内科", "消化科", "风湿科"],
  "sources": ["《新编药物学》第18版", "《痛风诊疗指南》2023版"],
  "confidence": 0.85,
  "is_safe_fallback": false,
  "latency_ms": 2340.50
}
```

**路由路径说明：**

| `route_path` | 含义 |
|-------------|------|
| `simple_rag` | 规则拦截命中的简单问题，直接检索 + LLM 生成 |
| `mdt` | 复杂交叉问题，触发多科室专家并发会诊 |
| `safe_fallback` | CoT 安全退避，知识库无相关文献，返回硬编码安全回复 |
| `error` | 系统异常 |

### 全量 API 端点

| 方法 | 路径 | 分类 | 说明 |
|------|------|------|------|
| `POST` | `/api/query` | 核心 | 医疗问答（REST） |
| `GET` | `/api/health` | 运维 | 健康检查 + Milvus 连接状态 |
| `GET` | `/api/metrics` | 监控 | 运行时聚合指标（成功率/退避率/路由分布/延迟） |
| `WS` | `/ws/query` | 核心 | WebSocket 实时流式查询 |
| `GET` | `/api/harness/traces` | Harness | 获取最近 N 条全链路追踪记录 |
| `GET` | `/api/harness/traces/{id}` | Harness | 获取指定 Trace 详情（含完整 Span 树） |
| `GET` | `/api/harness/evaluate` | Harness | 执行 7 维确定性评分 |
| `GET` | `/api/harness/experiments` | Harness | A/B 实验列表 |
| `GET` | `/api/harness/safety` | Harness | 安全守卫统计（成本/限流） |
| `GET` | `/` | 前端 | 交互式 UI |
| `GET` | `/docs` | 文档 | Swagger / OpenAPI 3.0 |

---

## 配置管理

### 三级配置优先级

```
环境变量 (> .env)  >  config/custom.yaml  >  config/default.yaml
    最高优先级              中间覆盖              默认基准
```

### 关键配置项速览

| 配置路径 | 类型 | 默认值 | 说明 |
|----------|------|--------|------|
| `llm.model` | str | `deepseek-chat` | LLM 模型名称 |
| `llm.temperature` | float | `0.3` | 默认生成温度 |
| `llm.temperatures.router` | float | `0.1` | 路由决策温度（低=确定性） |
| `milvus.use_milvus` | bool | `true` | 启用 Milvus / 内存模式 |
| `milvus.embedding_dim` | int | `512` | 向量维度 |
| `retrieval.top_k` | int | `10` | 检索返回文档数 |
| `retrieval.rrf_k` | int | `60` | RRF 融合参数 |
| `reranker.low_threshold` | float | `0.2` | CoT 退避触发阈值 |
| `react.max_iterations` | int | `5` | ReAct 循环最大迭代轮次 |
| `confidence.min_confidence` | float | `0.6` | 最低置信度 |
| `decision_maker.quality_threshold` | float | `0.5` | 决策器通过阈值 |
| `skill.similarity_threshold` | float | `0.75` | 技能冲突检测阈值 |
| `skill.min_confidence_for_extraction` | float | `0.7` | 技能提取最低置信度 |
| `harness.tracing.enabled` | bool | `true` | 链路追踪开关 |
| `harness.context.total_tokens` | int | `14000` | 上下文总 Token 预算 |

完整参数见 `config/default.yaml`（135 行，含逐项注释）。

---

## 数据与语料

### 语料库规模

| 语料 | 来源 | 条目数 | 语言 | 用途 |
|------|------|--------|------|------|
| 英文教科书 | MedQA-US (18 本) | 125,847 | EN | 英文医学检索 |
| 中文教科书 | MedQA Mainland 段落版 | 116,216 | ZH | 中文医学检索（主力） |
| 种子知识库 | 手工精选 8 条 | 8 | ZH | 内存模式兜底 / 快速体验 |
| 测试集 | MedQA Mainland test | 3,426 | ZH | 端到端评测 |

### 数据导入命令

```bash
# 导入中文教科书到 Milvus
python scripts/import_zh_textbooks.py

# 导入英文教科书到 Milvus
python scripts/import_textbooks.py

# 预计算种子向量（供种子数据使用）
python scripts/precompute_vectors.py

# 重建/更新种子知识库
python scripts/reseed_milvus.py
```

---

## 评估体系

### 三层评估金字塔

```
              ┌─────────────────────────┐
              │   Harness 7维确定性评分   │  ← 零 LLM 调用，完全可复现
              │   每次请求后在线实时评估    │
              ├─────────────────────────┤
              │   离线评估 (evaluate_rag)  │  ← MedQA test.jsonl 批量评测
              │   检索 MRR/Hit@k + 准确率  │
              ├─────────────────────────┤
              │   在线监控 (GET /metrics)  │  ← 会话级聚合指标
              │   成功率/退避率/延迟分布    │
              └─────────────────────────┘
```

### 离线评估

```bash
# 仅评估生成准确率
python scripts/evaluate_rag.py --mode generation

# 仅评估检索召回率（MRR, Hit@1/3/5/10）
python scripts/evaluate_rag.py --mode retrieval

# 同时评估检索 + 生成
python scripts/evaluate_rag.py --mode both

# 断点续评（防止中断丢失进度）
python scripts/evaluate_rag.py --mode generation --resume

# 保存基线（供后续 regression 检测）
python scripts/evaluate_rag.py --mode generation --save-baseline

# 对比现有基线
python scripts/evaluate_rag.py --mode generation --compare-baseline

# 通过 API 评估已部署服务
python scripts/evaluate_rag.py --mode generation --api-url http://localhost:8000
```

### Harness 7 维确定性评分

| 维度 | 权重 | 评分规则 |
|------|------|----------|
| **流程完整性** | 22% | 路由决策、检索执行、ReAct 迭代等关键节点是否经过 |
| **答案正确性** | 22% | 是否命中安全标记 / 禁忌检查 / 科室覆盖 |
| **接口验收** | 18% | answer 是否有实质内容、sources 是否可溯源 |
| **输出质量** | 15% | 回复长度、专业术语密度 |
| **效率** | 10% | 延迟（相对于基线）、Token 消耗 |
| **安全合规** | 8% | 是否触发 CoT 退避、是否规避禁忌场景 |
| **迭代能力** | 5% | 是否经历了路由升级 / Reflection 触发 |

核心设计原则：**"宁要可复现的粗糙分，不要会漂移的精准分"**——所有评分基于确定性规则和已有指标，零 LLM 调用，确保同一组输入产出完全一致的结果。

---

## Harness 工程框架

基于 **Agent = Model + Harness** 设计理念，Harness 是模型之外的工程外壳，提供四个维度的工程保障：

### 追踪观测 (Tracing)

```
TraceID 生成 ──▶ 贯穿全链路 (路由→检索→ReAct→决策)
    │
    ├── Span: 路由决策
    ├── Span: 反思拦截
    ├── Span: 检索 (KB + 画像约束)
    ├── Span: 重排 (Reranker)
    ├── Span: 科室1 ReAct  ──┬── Tool Call: 文献检索
    │                        └── Tool Call: 药物冲突
    ├── Span: 科室2 ReAct  ...
    ├── Span: 共识提炼
    ├── Span: DecisionMaker
    └── Span: 反思 + 技能提取
```

- 每条请求生成唯一 `trace_id`，跨模块传播
- API 端点支持查询最近 N 条追踪记录及单条详情
- 执行图可视化记录完整的函数调用链

### 安全守卫 (Safety Guard)

| 机制 | 说明 |
|------|------|
| **工具调用限流** | 20 次 / 60s 窗口，防止 Agent 失控频繁调用工具 |
| **工具参数验证** | `validate_tool_args` 预钩子，在工具执行前拦截非法参数 |
| **成本追踪** | 实时累计 LLM Token 消耗，按模型和阶段分解 |

### 上下文预算 (Context Budget)

三层分层管理 LLM 上下文窗口：

```
┌──────────────────────────────────────┐
│  PERMANENT 层 (2000 tokens)          │  ← 系统提示、核心约束（不可覆盖）
├──────────────────────────────────────┤
│  WORKING 层   (4000 tokens)          │  ← 对话历史、工具调用记录
├──────────────────────────────────────┤
│  DEEP 层      (8000 tokens)          │  ← 检索结果、文献上下文
├──────────────────────────────────────┤
│  TOTAL:       (14000 tokens)         │  ← 硬上限，超出触发裁剪
└──────────────────────────────────────┘
```

### A/B 实验框架

支持相同查询在不同配置下的对比评测，自动记录 `config_hash`、各维度评分和延迟，检测版本间 regression。

---

## 架构拓扑图

```
                                    ┌─────────────────────┐
                                    │     User / Client    │
                                    └──────────┬──────────┘
                                    HTTP/WS    │
                                    ┌──────────▼──────────┐
                                    │    FastAPI Server    │
                                    │   (Uvicorn ASGI)     │
                                    └──────────┬──────────┘
                                               │
                          ┌────────────────────┼────────────────────┐
                          │                    │                    │
                    ┌─────▼─────┐      ┌──────▼──────┐     ┌──────▼──────┐
                    │ RuleInterceptor │  │  LLMRouter   │     │ Confidence  │
                    │  (NER + Regex)  │  │ (Guided JSON)│     │   Checker   │
                    └─────┬─────┘      └──────┬──────┘     └──────┬──────┘
                          │                    │                    │
                          └────────────────────┼────────────────────┘
                                               │
                                   ┌───────────▼───────────┐
                                   │   MedicalOrchestrator  │
                                   │   (Harness Integrator) │
                                   └─────┬───────────┬─────┘
                                         │           │
                              ┌──────────▼──┐  ┌─────▼──────────┐
                              │  Simple RAG  │  │  MDT Consultation│
                              └──────┬───────┘  └─────┬──────────┘
                                     │                 │
                         ┌───────────┼───┐     ┌───────┼───────────┐
                         │           │   │     │       │           │
                    ┌────▼────┐ ┌───▼───▼─┐ ┌─▼───┐ ┌─▼───┐ ┌────▼────┐
                    │Reflection│ │Retriever│ │Cardio│ │Gastro│ │Rheumatol│
                    │ Manager  │ │ +       │ │ReAct │ │ReAct │ │ ReAct   │  ...
                    └─────────┘ │Reranker │ └──┬───┘ └──┬───┘ └────┬────┘
                                └────┬─────┘    │       │         │
                                     │          └───┬───┴────┬────┘
                                     │              │        │
                               ┌─────▼──────┐  ┌───▼────────▼───┐
                               │   Milvus   │  │ ConsensusBuilder│
                               │   (Collections)  │  + 共识引导检索  │
                               │            │  └────────┬────────┘
                               │ Medical_KB │           │
                               │ Patient_Profile  ┌────▼─────┐
                               │ Reflection_Mem   │Decision  │
                               │  Skill_Mem  │    │ Maker    │
                               └────────────┘    └────┬─────┘
                                                      │
                                           ┌──────────▼──────────┐
                                           │   Response + CoT    │
                                           │   Safe Fallback     │
                                           └─────────────────────┘
```

---

## 致谢

- **Embedding 模型**: [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)
- **Cross-Encoder**: [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- **LLM 服务**: DeepSeek API
- **向量数据库**: [Milvus](https://milvus.io/)
- **评估基准**: MedQA (US + Mainland)
