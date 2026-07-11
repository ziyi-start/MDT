# 医疗垂直领域多智能体协同 RAG 系统技术设计文档

**项目周期**：2026 年 5 月 - 至今
**文档版本**：v1.1
**核心技术栈**：LLM (DeepSeek), Multi-Agent (ReAct), Milvus 2.4+, Hybrid RAG (BM25 + Dense), CoT, BGE-Reranker Cross-Encoder, BGE-small Embedding, FastAPI, Python

---

## 1. 项目概述

### 1.1 业务痛点
传统医疗问答线性 RAG 架构面临三大核心挑战：
1.  **检索噪声大**：单向量检索缺乏精准度，常召回"表面语义相似但临床无关"的文献。
2.  **专业深度不足**：遇到跨科室并发症时，单模型无法覆盖多专科知识，回答笼统。
3.  **高危幻觉难控**：检索结果冲突或匮乏时，模型易强行生成（如编造药物相互作用），存在极大临床风险。

### 1.2 核心解法
构建模拟临床 MDT（多学科会诊）的多智能体会诊框架，引入**闭环动态路由、渐进式画像约束检索、共识引导检索与归因式反思沉淀**机制，实现从"被动检索生成"到具备主动决策与临床级安全纠偏能力的架构跃升。

---

## 2. 系统整体架构

系统由 `MedicalOrchestrator` 统一编排。以下架构图反映实际代码的 7 步闭环处理流水线。

```text
┌──────────────────────────────────────────────────────────────────┐
│                        用户提问 + 患者画像                        │
│                    POST /api/query  或  WS /ws/query               │
└────────────────────────────┬─────────────────────────────────────┘
                             ▼
┌───────────────────────────────────┐
│     MedicalOrchestrator.process() │  顶层编排器
└───────────────────┬───────────────┘
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
┌──────────┐ ┌────────────┐ ┌─────────────────┐
│ Step 1   │ │ Step 2     │ │ Step 3: 动态路由  │
│ 画像更新  │ │ 快速预检   │ │                  │
│          │ │            │ │ ┌─────────────┐  │
│ Profile  │ │ Hybrid     │ │ │规则拦截器    │  │  快车道
│ Extractor│ │ Retriever  │ │ │(NER+关键词) │  │  <50ms
│ (LLM抽取)│ │ top_k=3    │ │ └──┬──────────┘  │
│          │ │            │ │    │ 未命中       │
│ 写入     │ │ 结果为空?  │ │    ▼              │
│ Patient  │ │ → 安全退避  │ │ ┌────────────┐  │  慢车道
│ _Profile │ │            │ │ │LLM Router  │  │  (异步)
└──────────┘ └────────────┘ │ │(Guided JSON)│  │  LLM 调用
                             │ └──┬─────────┘  │
                             │    │             │
                             └────┼─────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼                           ▼
         ┌──────────────────┐        ┌──────────────────┐
         │  Step 4a:        │        │  Step 4b:        │
         │  Simple RAG 分支  │        │  MDT 多专家分支   │
         │                  │  低置信 │                  │
         │ 反射拦截         │ ──────▶│ 反射拦截         │
         │ 混合检索+重排    │ 携因打回│ 并发专家ReAct    │
         │ LLM生成答案      │        │   (ReAct+工具)   │
         │ 置信度评估       │        │ 共识提炼         │
         │ 低置信/冲突?     │        │ 共识引导检索     │
         │  → 自动升级MDT   │        │  (检索→重排→验证)│
         │                  │        │ Decision Maker   │
         └────────┬─────────┘        └────────┬─────────┘
                  │                           │
                  └──────────┬────────────────┘
                             │
      ┌──────────────────────┼──────────────────────┐
      ▼                      ▼                      ▼
┌───────────┐     ┌───────────────┐     ┌──────────────────┐
│ 正常输出   │     │ CoT 安全退避   │     │ 反思沉淀 (异步)    │
│ 最终医嘱   │     │ 硬编码保守回复  │     │ 意图-归因-避坑三元组│
│           │     │ "建议线下就医"  │     │ → Reflection_Mem  │
└───────────┘     └───────────────┘     └──────────────────┘


══════════════ 检索与记忆层 ══════════════

┌────────────────────────────────────────────────────────────────┐
│                     Milvus 三集合架构                           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────┐    │
│  │ Medical_KB   │  │Patient_      │  │Reflection_Mem     │    │
│  │(512维Dense   │  │Profile       │  │(失败反思三元组)    │    │
│  │ +BM25混合)   │  │(患者画像)     │  │                   │    │
│  └──────┬───────┘  └──────┬───────┘  └────────┬──────────┘    │
│         │                 │                    │               │
│         ▼                 ▼                    ▼               │
│  ┌──────────────────────────────────────────────────────┐     │
│  │              HybridRetriever                          │     │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │     │
│  │  │ 画像软约束   │  │ 画像硬约束    │  │ RRF 融合    │ │     │
│  │  │ query 改写  │  │ filter 过滤  │  │ k=60        │ │     │
│  │  └─────────────┘  └──────────────┘  └─────────────┘ │     │
│  │                         │                            │     │
│  └─────────────────────────┼────────────────────────────┘     │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────┐     │
│  │              MedicalReranker                          │     │
│  │  ┌──────────────────────────────────────────────┐   │     │
│  │  │ BGE-Reranker-v2-m3 Cross-Encoder (本地)       │   │     │
│  │  │ query-doc pair 逐对打分, 后备: n-gram 重叠    │   │     │
│  │  └──────────────────────────────────────────────┘   │     │
│  └───────────────────────┬──────────────────────────────┘     │
│                          │ 最高分 < 0.2 → CoT退避              │
└──────────────────────────┼────────────────────────────────────┘
                           │
                           ▼
                   [下游 LLM 生成]

────────────────── 嵌入模型 (独立延迟加载) ──────────────────

  BGE-small-zh-v1.5 (512维, SentenceTransformer)
  ├── 查询编码: 首次用户查询时加载, 用于 query → vector
  └── 种子向量: seed_vectors.json (预计算, 8条, 启动免加载)

────────────────── LLM 调用链 ──────────────────

  AsyncLLMClient (DeepSeek Chat API, OpenAI 兼容)
  ├── 路由解析 (LLMRouter)
  ├── 画像抽取 (ProfileExtractor)
  ├── RAG 生成 (SimpleRAGWorkflow)
  ├── 置信校验 (ConfidenceChecker)
  ├── 反思三元组 (ReflectionManager)
  ├── ReAct 循环 (ReactEngine, 最多5次)
  ├── 共识提炼 + 共识验证 (MDTConsultationWorkflow)
  └── 质量评估 (DecisionMaker)
```

---

## 3. 核心模块详细设计与实现

### 3.1 多角色 Prompt 构造与专家模拟
**目标**：让通用大模型具备"专科医生"的执业能力。

*   **当前实现**：使用单一 LLM 实例（DeepSeek Chat API），通过 `EXPERT_SYSTEM_PROMPT` 为不同科室注入专科角色 Prompt，实现多专家角色模拟。MDT 会诊时，每个科室独立创建 `ReactEngine` 实例，通过 `asyncio.gather` 并发执行。
*   **生产演进方向**：Llama-Factory QLoRA 微调（`lora_rank=64`, `lora_alpha=128`，目标模块 `q_proj, k_proj, v_proj, o_proj`），vLLM 部署优化（PagedAttention + Continuous Batching），使多 Agent 并发请求吞吐量提升 2-3 倍。

### 3.2 闭环动态路由与置信度评估
**目标**：平衡系统效率与深度，实现 RAG 失败后的自动跃升。

*   **两级混合路由分流**（`MedicalOrchestrator._route_async` 编排）：
    *   **规则拦截（快车道）**：`RuleInterceptor.intercept()` — 同步执行。默认使用正则匹配医疗实体与冲突关键词，可配置 `MDT_NER_SERVICE_URL` 调用外部 BERT-CRF NER 服务。决策逻辑：实体数 ≤1 且无冲突词 → SimpleRAG；实体数 >2 或存在冲突词 → MDT；实体数 =2 且无冲突词 → 灰度区返回 None 交由 LLM 路由。
    *   **LLM 路由（慢车道）**：`LLMRouter.route()` — 异步调用 LLM，开启 Guided Decoding（`response_format={"type": "json_object"}`），强制输出 `RouteDecision` JSON（包含 `route_path` 和 `departments`）。解析失败时安全降级为 SimpleRAG。
*   **置信度评估机制**（`ConfidenceChecker`）：
    *   **文档一致性校验**：Reranker 打分后，若 Top-1 与 Top-2 分数差距 < 阈值（`score_gap_threshold=0.15`）且结论相悖，判定为"检索冲突"。
    *   **生成自验证**：调用 LLM 验证生成答案的核心结论是否有文档支撑（`CONFIDENCE_CHECK_PROMPT`），无支撑则判定为"低置信度"。
*   **携因打回与动态升级**：触发低置信/冲突时，抛出 `RouteEscalationException(reason)`。顶层编排器捕获异常后，将失败原因作为 `escalation_reason` 注入 MDT 各专家的 System Prompt，实现有上下文的质量升级。

### 3.3 渐进式画像构建与共识引导检索
**目标**：解决表面语义相似但临床无关的误召回痛点。

*   **渐进式画像构建**：用户每轮对话后，异步触发轻量级 LLM 执行信息抽取（IE），将非结构化文本转化为结构化 JSON（`diseases`, `medications`, `allergies`）。以 `user_id` 为主键，通过 Milvus 的 Upsert 操作动态更新 `Patient_Profile` 集合。
*   **画像约束检索边界**：
    *   **硬约束**：若画像存在禁忌（如胃溃疡），在 Milvus 检索时增加元数据 Filter 条件，直接在数据库层排雷。
    *   **软约束**：在检索 Query 中拼接禁忌信息进行查询改写。
*   **多专家主动生成 Query**：Expert Agent 在 ReAct 循环中，主动调用 `Literature_Search` 工具生成检索词（如：肾内科专家主动生成"CKD 3期 痛风 安全止痛药替代方案"），取代用户原始 Query 进行检索。
*   **共识引导检索**：多专家共识提炼完成后，将共识摘要文本作为 Query，发起第二轮检索（`HybridRetriever.retrieve(consensus_text, ...)`）。该步骤强制将 LLM 生成的共识"接地"回知识库证据，避免专家仅凭自身知识发声却脱离文献依据。检索返回后经 `MedicalReranker` 逐对重排打分，取 Top-N 高相关文档。
*   **检索证据验证共识**：将重排后的 Top-N 文档与共识摘要拼接为 `CONSENSUS_VERIFICATION_PROMPT`，调用 LLM 逐条比对验证。每条共识结论标记为 `[Supported]`（有文献支撑）、`[Revised]`（与文献矛盾需修正）或 `[Removed]`（无文献支撑删除），最终产出一份有文献锚定的会诊报告。
*   **混合检索与重排**：基于 Milvus 2.4+ 的 Hybrid Search，同时执行 BM25（精准匹配药名）和 Dense（语义匹配），通过 RRF 算法合并结果。
    *   **Dense 向量模型**：BAAI/bge-small-zh-v1.5（512 维），轻量化中文语义向量模型，延迟加载，首次查询时初始化。
    *   **种子知识库预计算**：内置 8 条跨科室医学知识，向量提前通过 bge-small 模型预计算并持久化为 `seed_vectors.json`。启动时直接读取 JSON 注入 Milvus，无需加载 embedding 模型，大幅缩短冷启动时间。
    *   **重排模型**：BAAI/bge-reranker-v2-m3 Cross-Encoder 模型在本地进行 query-document pair 逐对相关性打分。模型延迟加载，首次重排时初始化。
    *   **去噪逻辑**：Reranker 打分后，取 top_k 条高相关性文档送入下游生成。

### 3.4 归因式反思沉淀与 CoT 安全退避
**目标**：应对极端噪声，建立系统级"免疫系统"，坚守临床安全底线。

*   **归因式反思沉淀**（`ReflectionManager`）：
    *   **生成**：Decision-maker 评估质量不合格时，调用 `ReflectionManager.generate_and_store()`，LLM 强制输出结构化三元组：`<意图, 归因, 避坑动作>`（例：`<为CKD开止痛药, 忽略NSAIDs肾损风险, 必须核查肾功能禁用NSAIDs>`）。
    *   **写入与拦截**：将三元组向量化存入 Milvus `Reflection_Mem` 集合。每次 SimpleRAG 和 MDT 流程开始时，`ReflectionManager.search_reflection(query)` 先检索反思记忆，若高相似度命中（`search_threshold=0.5`），则将 `历史教训：{avoid_action}` 作为 Hint 注入 System Prompt。
*   **CoT 安全退避机制**（`InsufficientInformationException`）：
    *   **触发场景**：① Step 2 快速预检返回空结果；② Reranker 最高得分低于 `low_threshold`（0.2），表明知识库无相关知识；③ DecisionMaker 评估存在高幻觉风险。
    *   **强制中断**：抛出 `InsufficientInformationException`，顶层编排器捕获后切断 ReAct 循环，禁止发散推理。
    *   **退避执行**：走硬编码的保守策略链路，输出 `SAFE_FALLBACK_RESPONSE`："抱歉，缺乏权威文献支撑，强烈建议线下就医"，严禁编造具体用药方案（宁拒答不幻觉）。

---

## 4. 核心数据流转全生命周期（案例推演）

**场景**：*"患者有高血压和胃溃疡，最近痛风发作，能吃布洛芬吗？"*

1.  **路由阶段**：NER 提取出 `高血压`、`胃溃疡`、`痛风`、`布洛芬`。规则拦截器判定实体数>2，直接路由至 **MDT 模式**，招募 [心内科, 消化科, 风湿科]。
2.  **画像与检索阶段**：从 Milvus `Patient_Profile` 读出患者长期服用"氯吡格雷"；风湿科专家主动生成 Query："胃溃疡 痛风 急性发作 非NSAIDs止痛方案"。
3.  **会诊与评估阶段**：风湿科建议用秋水仙碱，消化科提示仍需护胃。共识提炼后，将共识摘要作为 Query 检索知识库，经 Reranker 重排后调用 LLM 验证共识结论，标记 Supported/Revised/Removed。验证后的共识报告经 Decision-maker 评估，确认无冲突且置信度达标。
4.  **反思/退避阶段（分支假设）**：若知识库完全没有关于该复杂并发症的用药记录，检索得分极低，系统将触发 **CoT 退避**，拒绝提供用药方案，输出安全就医建议；并将此次"无证据支持"的路径写入反思记忆。

---

## 5. 代码目录结构

```text
medical_rag_system/
├── main.py                     # 入口
├── schema/                     # Pydantic 数据模型
│   ├── models.py               # MedicalQuery, PatientProfile, ReflectionTriple, RouteDecision
│   └── messages.py             # 兼容 OpenAI 格式的 Message, ToolCall, ToolMessage
├── llm/                        # LLM 底层调用封装
│   ├── client.py               # 基于 httpx/openai 的异步 LLM 调用客户端 (支持 Guided Decoding)
│   └── prompt_templates.py     # 所有 Prompt 常量与模板
├── engine/                     # 核心手写引擎
│   ├── react_engine.py         # 手写 ReAct 循环 (解析 Tool Calls 并自动执行)
│   └── tool_registry.py        # 工具注册器 (管理工具描述与执行函数映射)
├── tools/                      # Agent 可调用工具实现
│   ├── literature_search.py    # 文献检索工具
│   └── drug_interaction.py     # 药物冲突查询工具
├── memory/                     # 记忆管理模块
│   ├── profile_extractor.py    # 渐进式画像抽取器
│   └── reflection_manager.py   # 归因式反思三元组管理
├── rag/                        # 检索增强模块
│   ├── milvus_client.py        # Milvus 多集合管理 (KB, Profile, Reflection)
│   ├── embedding.py            # BGE-small-zh-v1.5 Dense 向量编码（512 维）
│   ├── hybrid_retriever.py     # BM25 + Dense 混合检索 + 手写 RRF 融合
│   └── reranker.py             # BGE-Reranker Cross-Encoder 本地重排
├── router/                     # 动态路由引擎
│   ├── rule_interceptor.py     # 规则拦截器 (调用外部 NER 服务 + 正则)
│   ├── llm_router.py           # LLM 结构化意图路由
│   └── confidence_checker.py   # 置信度评估器 (文档一致性+生成自验证)
└── workflow/                   # 业务工作流编排
    ├── simple_rag.py           # 简单 RAG 流程
    ├── mdt_consultation.py     # 多专家会诊流程
    └── medical_orchestrator.py # 顶层闭环编排器 (路由->会诊->退避)
```

---

## 6. 核心配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `milvus.embedding_dim` | 512 | 向量维度（BGE-small） |
| `retrieval.top_k` | 10 | 检索返回文档数量 |
| `retrieval.rerank_top_k` | 5 | 重排后保留文档数量 |
| `retrieval.consensus_retrieval_top_k` | 8 | 共识引导检索返回文档数 |
| `retrieval.consensus_rerank_top_k` | 4 | 共识引导检索重排后保留数 |
| `retrieval.quick_check_top_k` | 3 | CoT 预检查检索数量 |
| `retrieval.rrf_k` | 60 | RRF 融合参数 |
| `reranker.low_threshold` | 0.2 | CoT 退避阈值 |
| `confidence.score_gap_threshold` | 0.15 | 文档一致性校验分数差距阈值 |
| `decision_maker.quality_threshold` | 0.5 | 质量分低于此值触发 CoT 退避 |
| `react.max_iterations` | 5 | ReAct 循环最大迭代次数 |
| `reflection.search_threshold` | 0.5 | 反思检索相似度阈值 |
