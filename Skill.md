
---

```markdown
# Role: 资深 Python AI 底层架构师
# Task: 根据以下设计规范，从零手写实现医疗多智能体协同 RAG 系统的完整 Python 代码。绝不使用 LangChain/LlamaIndex 等框架，纯手搓工作流与 Agent 循环。严格忠实于提供的技术设计文档，不遗漏任何核心机制。

## 1. 系统概述与技术栈约束
本系统模拟临床 MDT 会诊，核心机制包括：闭环动态路由、渐进式画像约束检索、归因式反思沉淀与 CoT 安全退避。

**强制技术栈约束：**
*   **语言**: Python 3.9+ (全面使用 type hints 和 Pydantic V2 做数据校验)
*   **LLM 交互**: 使用 `httpx` 或 `openai` 原生 SDK 调用兼容 OpenAI 的 API (如 vLLM 部署的 Baichuan)。**禁止引入 LangChain 等编排框架**。
*   **向量数据库**: 使用 `pymilvus` 原生 SDK。
*   **异步编程**: 核心工作流必须使用 `asyncio` 实现，多专家会诊需并发执行 (`asyncio.gather`)。
*   **工作流引擎**: 纯手写 ReAct 循环和 Tool Call 解析引擎。

## 2. 核心目录结构设计
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
│   ├── hybrid_retriever.py     # BM25 + Dense 混合检索 + 手写 RRF 融合
│   └── reranker.py             # 调用外部 Rerank API
├── router/                     # 动态路由引擎
│   ├── rule_interceptor.py     # 规则拦截器 (调用外部 NER 服务 + 正则)
│   ├── llm_router.py           # LLM 结构化意图路由
│   └── confidence_checker.py   # 置信度评估器 (文档一致性+生成自验证)
└── workflow/                   # 业务工作流编排
    ├── simple_rag.py           # 简单 RAG 流程
    ├── mdt_consultation.py     # 多专家会诊流程
    └── medical_orchestrator.py # 顶层闭环编排器 (路由->会诊->退避)
```

## 3. 核心逻辑实现指令 (严格按此逻辑编写，不得遗漏)

### Phase 1: 基础域模型与 LLM 原生客户端
1.  **Pydantic 模型**: 定义 `PatientProfile` (diseases, medications, allergies), `ReflectionTriple` (intent, cause, avoid_action), `RouteDecision` (route_path, departments)。
2.  **LLM 客户端**: 封装 `AsyncLLMClient`，核心实现 `async def chat(messages, tools, response_format)`。必须支持 `response_format={"type": "json_object"}` (Guided Decoding) 以强制输出 JSON。
3.  **Prompt 模板**: 将所有 Prompt 提取为常量。

### Phase 2: 手写 ReAct 引擎与工具注册 (最核心！)
1.  **工具注册器**: 实现 `ToolRegistry` 类。提供装饰器将 Python 异步函数注册为工具，并自动生成 OpenAI Tool JSON Schema。
2.  **手写 ReAct 引擎**: 实现 `ReactEngine`：
    *   循环调用 LLM。
    *   如果 LLM 返回 `finish_reason == "tool_calls"`，解析出工具名和参数。
    *   通过 `ToolRegistry` 执行对应的 Python 异步函数获得结果。
    *   将结果封装为 `role: "tool"` 的 Message 追加到历史中，继续调用 LLM。
    *   如果 LLM 返回 `finish_reason == "stop"`，退出循环，返回最终文本。
    *   **必须限制最大迭代次数 (如 max_iterations=5) 防死循环**。

### Phase 3: 记忆管道与混合检索 (严格还原文档细节)
1.  **Milvus 管理**: 封装 `MilvusManager`，实现 `Medical_KB`, `Patient_Profile`, `Reflection_Mem` 三个集合的建表、Upsert 和 Search 接口。
2.  **渐进式画像构建**: 实现 `ProfileExtractor`，异步调用 LLM 抽取结构化 JSON，并 Upsert 到 `Patient_Profile`。
3.  **混合检索与画像约束**: 在 `HybridRetriever` 中实现：
    *   **硬约束**: 根据 `PatientProfile` 中的禁忌，动态拼接 Milvus 的 boolean filter 表达式 (如 `contraindications not contain ["胃溃疡"]`)。
    *   **软约束**: 在检索 Query 中拼接禁忌信息进行查询改写。
    *   **混合检索**: 调用 Milvus 的 hybrid_search，手写 RRF (Reciprocal Rank Fusion) 算法合并 BM25 和 Dense 结果。
4.  **重排**: 调用外部 BGE-Reranker API。
5.  **归因式反思沉淀**: 实现 `ReflectionManager`，打回时强制 LLM 输出 `ReflectionTriple` 结构化数据，向量化后存入 `Reflection_Mem`。

### Phase 4: 闭环动态路由引擎 (严格还原文档细节)
1.  **规则拦截 (快车道)**: 实现 `RuleInterceptor`，**必须调用外部医疗 NER 服务 (如 BERT-CRF 模型部署的 HTTP API)** 提取医疗实体，同时结合正则匹配冲突关键词。若 `实体数 <= 1` 且无冲突词，直接路由至 `SimpleRAGWorkflow`。
2.  **LLM 路由 (慢车道)**: 对灰度问题，调用 LLM 并开启 Guided Decoding，强制输出 `RouteDecision` JSON，解析出 `departments` 列表。
3.  **置信度评估与动态升级**: 在 `SimpleRAGWorkflow` 末尾，运行 `ConfidenceChecker`：
    *   **文档一致性校验**: 若 Reranker Top-1 与 Top-2 分数差 < 阈值且结论相悖，判定为“检索冲突”。
    *   **生成自验证**: 要求 LLM 生成答案时输出引用来源 `[Source: Doc 1]`。若核心结论无对应文档引用，判定为“低置信度”。
    *   **携因打回**: 触发低置信/冲突时，将失败原因（如“检索指南冲突，缺乏肾功能考量”）作为新 Context 注入，抛出 `RouteEscalationException`，由顶层编排器捕获并强制转入 MDT 模式。

### Phase 5: 多专家 MDT 异步编排
1.  **MDT 编排**: 实现 `MDTConsultationWorkflow`：
    *   根据 LLM 路由返回的 `departments`，动态实例化对应专科的 `ReactEngine`。
    *   **共识引导检索**: 给各专科 Expert 的 System Prompt 注入患者画像，并在 Prompt 中要求："你必须根据患者禁忌，主动构思专业的检索词去调用工具，不要直接使用患者的原话检索"。
    *   **反思拦截**: 执行前检索 `Reflection_Mem`，若高相似度命中，将避坑动作作为 Hint 强制追加到 System Prompt 中 (如："⚠️历史教训：务必核查肾功能，禁用NSAIDs")。
    *   **并发会诊**: 使用 `asyncio.gather(*[expert.run(query) for expert in experts])` 并发执行多专家 ReAct 循环。
    *   **共识提炼**: 收集多专家结果，调用 LLM 提炼最终会诊报告。

### Phase 6: 决策与 CoT 退避 (严格还原文档细节)
1.  **Decision Maker**: 实现安全阀逻辑，评估多专家共识质量。
2.  **CoT 安全退避**: 若在检索阶段，Reranker 最高分低于极低阈值 (如 0.2)，表明知识库无相关知识，直接抛出 `InsufficientInformationException`。
3.  **强制中断与保守策略**: 顶层捕获此异常后，切断一切 LLM 继续生成的可能，直接返回硬编码的安全回复："抱歉，缺乏权威文献支撑，强烈建议线下就医"（宁拒答不幻觉）。

## 4. 代码规范要求
*   纯手写逻辑，绝不出现 `from langchain...` 或 `from llama_index...`。
*   全面使用 `async/await`，特别是 LLM 调用、Milvus 检索、工具执行。
*   使用 Pydantic V2 做严格的输入输出校验。
*   日志完善：关键决策节点（路由分流、NER提取结果、工具调用、反思拦截、退避触发）必须有清晰的 `logging.info/warning`。
*   所有异常（LLM超时、工具执行失败、解析错误）必须有兜底处理，医疗系统不能 Crash。

## 5. 执行指令
请从 **Phase 1** 开始，逐步输出完整的 Python 代码实现。每输出完一个 Phase，暂停并询问我："Phase X 已生成，是否继续？"
```