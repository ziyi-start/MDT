# MDT — 医疗多智能体协同 RAG 系统

基于多智能体协作、混合检索与动态路由的医学知识问答系统。通过多科室专家 ReAct 会诊 + 四库分层记忆 + Harness 工程框架，提供高质量、可追溯的医疗咨询回答。

## 核心能力

- **动态两级路由**：规则拦截（<50ms 快车道）+ LLM 结构化路由（慢车道），自动分流至 Simple RAG 或 MDT 多专家会诊
- **多专家 ReAct 会诊**：并发启动多个科室专家（心内科、消化科、风湿科等），每个专家自主调用文献检索与药物冲突检查工具
- **Hybrid RAG 检索**：BM25 + Dense 向量混合检索（Milvus native hybrid_search），RRF 融合 + Cross-Encoder 重排
- **四库分层记忆**：Medical_KB → Patient_Profile → Reflection_Mem（失败教训）→ Skill_Mem（成功经验）
- **Agent 自进化**：从成功回答中自动提取可复用技能（add / merge / discard），从失败回答中提取反思教训
- **CoT 安全退避**：知识库无相关文献时切断 LLM 生成，返回硬编码安全回复，杜绝幻觉
- **Harness 工程框架**：分布式追踪、7 维确定性评分、安全守卫（限流 + 成本追踪）、分层上下文预算

## 技术栈

| 层次 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| Web 框架 | FastAPI + Uvicorn |
| LLM | DeepSeek (OpenAI 兼容 API) |
| 向量数据库 | Milvus 2.5+ (Standalone) |
| Embedding | BAAI/bge-small-zh-v1.5 (512d) |
| Cross-Encoder | BAAI/bge-reranker-v2-m3 |
| 数据校验 | Pydantic v2 |
| 容器化 | Docker Compose |

## 项目结构

```
MDT/
├── backend/                        # 后端核心代码
│   ├── main.py                     # FastAPI 入口 + API 端点
│   ├── config.py                   # 集中配置管理
│   ├── seed_data.py                # Milvus 种子数据注入
│   ├── schema/                     # 数据模型层（Pydantic）
│   ├── llm/                        # LLM 客户端 + Prompt 模板
│   ├── engine/                     # ReAct 引擎 + 工具注册 + 安全守卫
│   ├── router/                     # 动态路由（规则拦截 + LLM 路由 + 置信度）
│   ├── rag/                        # 检索增强（Milvus + 混合检索 + 重排）
│   ├── memory/                     # 记忆模块（画像 + 反思 + 技能）
│   ├── tools/                      # Agent 工具（文献检索 + 药物冲突）
│   ├── workflow/                   # 工作流编排（协调器 + Simple RAG + MDT 会诊）
│   ├── monitoring/                 # 监控评估（指标 + 追踪 + 离线评估）
│   ├── harness/                    # Harness 工程（7维评分 + A/B 实验）
│   └── context/                    # 上下文预算管理
├── config/
│   └── default.yaml                # 默认 YAML 配置
├── frontend/
│   └── index.html                  # 单页交互式 UI
├── scripts/                        # 运维脚本（评估 + 数据导入）
├── data/                           # 数据集（教科书 + 测试集）
├── docker-compose.yml              # 容器编排
├── .env.example                    # 环境变量模板
└── config.md                       # 配置说明
```

## 快速开始

### 1. 环境要求

- Python 3.10+
- Docker + Docker Compose

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 MDT_LLM_API_KEY
```

### 3. 启动 Milvus 基础设施

```bash
docker compose up -d etcd minio milvus
```

这将启动 Milvus (端口 19530)、etcd、MinIO 以及 Attu (Web UI, 端口 8001)。

### 4. 安装 Python 依赖

```bash
cd backend
pip install -r requirements.txt
```

### 5. 启动服务

```bash
# 内存模式（无需 Milvus，使用内置 8 条种子知识库）
cd backend && python main.py

# Milvus 模式
MDT_USE_MILVUS=true python main.py
```

服务启动后访问 `http://localhost:8000`，API 文档见 `http://localhost:8000/docs`。

## API 端点

### REST API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/query` | 核心查询接口 |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/api/metrics` | 运行时监控指标 |
| `GET` | `/api/harness/traces` | 获取追踪记录 |
| `GET` | `/api/harness/traces/{id}` | 追踪详情 |
| `GET` | `/api/harness/evaluate` | 7 维 Harness 评估 |
| `GET` | `/api/harness/experiments` | 实验列表 |
| `GET` | `/api/harness/safety` | 安全守卫统计 |
| `GET` | `/` | 前端首页 |
| `GET` | `/docs` | Swagger API 文档 |

### WebSocket

| 路径 | 说明 |
|------|------|
| `ws://host:8000/ws/query` | 实时查询流式交互 |

### 查询请求格式

```json
{
  "query": "患者有高血压和胃溃疡，最近痛风发作，能吃布洛芬吗？",
  "user_id": "default_user"
}
```

### 查询响应格式

```json
{
  "answer": "...详细回答...",
  "route_path": "mdt",
  "departments": ["心内科", "消化科", "风湿科"],
  "sources": ["《新编药物学》第18版", "《痛风诊疗指南》2023版"],
  "confidence": 0.85,
  "is_safe_fallback": false,
  "latency_ms": 2340.50
}
```

## 配置

配置优先级：**环境变量 > `config/custom.yaml` > `config/default.yaml`**

| 配置方式 | 说明 |
|----------|------|
| 环境变量 | 复制 `.env.example` 为 `.env`，填入实际值 |
| YAML 覆盖 | 复制 `config/default.yaml` 为 `config/custom.yaml` 按需修改 |

主要配置项参见 `config/default.yaml`（含 LLM、Milvus、检索、重排、置信度、ReAct、反思、技能、Harness 等全部参数）。

## 数据导入

```bash
# 中文教科书向量导入
python scripts/import_zh_textbooks.py

# 英文教科书导入
python scripts/import_textbooks.py

# 重建种子数据
python scripts/reseed_milvus.py
```

## 评估

```bash
# 端到端评估（生成准确率）
python scripts/evaluate_rag.py --mode generation

# 检索召回率评估
python scripts/evaluate_rag.py --mode retrieval

# 同时评测检索 + 生成
python scripts/evaluate_rag.py --mode both

# 断点续评
python scripts/evaluate_rag.py --mode generation --resume

# 保存/对比基线
python scripts/evaluate_rag.py --mode generation --save-baseline
python scripts/evaluate_rag.py --mode generation --compare-baseline
```
