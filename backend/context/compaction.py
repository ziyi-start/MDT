"""内容感知上下文压缩 — 参考 Letta Compaction + LLMLingua 选择性压缩

设计理念:
  "不是按字符截断，而是按语义保留 —— 像人类记忆一样，记住重要的，遗忘冗余的"
  "Content-aware compaction 是 Harness 与 Harness 之间的核心差异点"

参考架构:
  - Letta Content-aware Compaction: Compaction should preserve key facts, not just truncate
  - Microsoft LLMLingua: 选择性压缩，非均匀降采样，保留高 perplexity tokens
  - Anthropic Prompt Caching: 缓存不变部分，只发送变化部分
  - LangChain "Your harness, your memory": Content-aware vs content-blind compaction

核心策略:
  1. 结构化压缩: 不是把所有内容合并成一段摘要，而是保留结构化格式
  2. 优先级标记: 诊断结论 > 药物禁忌 > 剂量方案 > 一般信息 > 冗余重述
  3. LLM-guided 选择: 用小模型 (如 deepseek-chat, temperature=0.1) 判断哪些内容必须保留
  4. 去重合并: 同一事实的不同表述合并为一个版本
  5. 关键事实萃取: 从多轮对话中提取不变的医学事实

压缩策略对比:
  ┌────────────────────┬──────────────────┬──────────────────────┐
  │ Content-blind       │ 简单截断           │ 可能丢失关键禁忌      │
  │ (当前实现)          │ TokenBudget.truncate│                     │
  ├────────────────────┼──────────────────┼──────────────────────┤
  │ Rule-based          │ 按重要性评分裁剪   │ 规则固定，不灵活       │
  │ (部分实现)          │ _score_entry()    │                      │
  ├────────────────────┼──────────────────┼──────────────────────┤
  │ Content-aware (新增) │ LLM 理解语义后选择 │ 最准确，有小额 LLM 成本 │
  │                     │ 保留关键事实        │                      │
  └────────────────────┴──────────────────┴──────────────────────┘
"""

from __future__ import annotations

import json
import logging
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Callable

from context.memory_hierarchy import TokenEstimator

logger = logging.getLogger(__name__)


@dataclass
class CompactionConfig:
    """压缩配置"""
    max_compacted_tokens: int = 2000
    llm_temperature: float = 0.1
    min_chars_for_llm_compaction: int = 500
    dedup_similarity_threshold: float = 0.85

    fact_types: dict = field(default_factory=lambda: {
        "diagnosis": 1.0,
        "contraindication": 0.95,
        "drug_interaction": 0.95,
        "dosage": 0.85,
        "treatment_plan": 0.80,
        "symptom": 0.70,
        "general_info": 0.40,
        "redundant": 0.10,
    })


class ContentAwareCompactor:
    """内容感知压缩器

    三层压缩策略级联:
    1. 快速去重 (MD5 hash) — 零 LLM 成本
    2. 规则评分裁剪 (fact_type priority) — 确定性
    3. LLM-guided 选择 (可选, 仅在意义模糊时) — 精度最高
    """

    def __init__(self, config: Optional[CompactionConfig] = None, llm_client=None):
        self.config = config or CompactionConfig()
        self.llm = llm_client
        self._seen_hashes: set[str] = set()

    def compact_conversation(self, entries: list[dict], max_tokens: int = 0) -> str:
        """压缩多轮对话 — 保留关键医学事实，丢弃冗余"""
        if max_tokens <= 0:
            max_tokens = self.config.max_compacted_tokens

        if not entries:
            return ""

        # Stage 1: 去重
        deduped = self._deduplicate(entries)

        # Stage 2: 按优先级标记医学事实
        scored = self._score_by_fact_type(deduped)

        # Stage 3: 按分数排序，预算内保留
        kept = self._select_by_budget(scored, max_tokens)

        # Stage 4: 格式化为结构化摘要
        return self._format_compacted(kept)

    def compact_documents(self, documents: list, query: str, max_tokens: int = 0) -> str:
        """压缩检索文档 — 只保留与查询真正相关的段落"""
        if max_tokens <= 0:
            max_tokens = self.config.max_compacted_tokens
        if not documents:
            return ""

        query_terms = set(query.lower().split())
        scored_docs = []
        for doc in documents:
            content = getattr(doc, "content", "") if hasattr(doc, "content") else str(doc)
            source = getattr(doc, "source", "") if hasattr(doc, "source") else ""
            score = getattr(doc, "score", 0.0) if hasattr(doc, "score") else 0.0

            relevance = self._compute_doc_relevance(content, query_terms)
            fact_score = self._detect_fact_importance(content)

            combined = 0.35 * score + 0.35 * relevance + 0.30 * fact_score
            scored_docs.append({
                "content": content,
                "source": source,
                "combined_score": combined,
                "tokens": TokenEstimator.estimate(content),
            })

        scored_docs.sort(key=lambda x: x["combined_score"], reverse=True)
        kept = []
        budget = max_tokens
        for d in scored_docs:
            if budget <= 0:
                break
            content = d["content"]
            if d["tokens"] > budget:
                content = self._extractive_snip(content, budget, query_terms)
                d["tokens"] = TokenEstimator.estimate(content)
            if d["tokens"] <= budget:
                label = f"[Source: {d['source']}]" if d["source"] else ""
                kept.append(f"{label}\n{content}")
                budget -= d["tokens"]

        return "\n\n".join(kept)

    async def llm_guided_compact(self, content: str, context_hint: str = "", max_tokens: int = 0) -> str:
        """LLM-guided 压缩 — 用 LLM 判断哪些内容必须保留

        仅在内容复杂度高、规则不确定时调用。
        使用 structured output 强制 JSON 输出保留事实列表。
        """
        if not self.llm:
            return self._fallback_compact(content, max_tokens)

        if max_tokens <= 0:
            max_tokens = self.config.max_compacted_tokens

        from config import cfg
        resp = await self.llm.chat(
            messages=[{
                "role": "system",
                "content": (
                    "你是一个上下文压缩器。从以下对话历史中提取必须保留的关键事实（JSON格式）。"
                    "按医学重要性排序：诊断结论 > 药物禁忌 > 相互作用 > 剂量 > 治疗方案 > 症状 > 一般信息。"
                    "输出 JSON: {\"kept_facts\": [\"事实1\", \"事实2\", ...]}"
                ),
            }, {
                "role": "user",
                "content": f"上下文提示: {context_hint}\n\n需要压缩的内容:\n{content[:8000]}",
            }],
            response_format={"type": "json_object"},
            temperature=self.config.llm_temperature,
        )

        try:
            data = json.loads(resp.content or "{}")
            facts = data.get("kept_facts", [])
            if facts:
                return "\n".join(facts)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM-guided compaction failed, fallback: {e}")

        return self._fallback_compact(content, max_tokens)

    def _deduplicate(self, entries: list[dict]) -> list[dict]:
        """基于内容 hash 的快速去重"""
        result = []
        for entry in entries:
            content = entry.get("content", "")
            h = hashlib.md5(content[:500].encode()).hexdigest()
            if h not in self._seen_hashes:
                self._seen_hashes.add(h)
                result.append(entry)
        if len(self._seen_hashes) > 2000:
            self._seen_hashes = set(list(self._seen_hashes)[-1000:])
        return result

    def _score_by_fact_type(self, entries: list[dict]) -> list[dict]:
        """按医学事实类型评分"""
        fact_keywords = {
            "diagnosis": ["诊断", "确诊", "判断为", "考虑为", "不排除"],
            "contraindication": ["禁忌", "禁用", "避免", "不可使用", "慎用", "禁止"],
            "drug_interaction": ["相互作用", "联用", "配伍", "冲突", "拮抗"],
            "dosage": ["剂量", "mg", "每次", "每日", "用量", "用法"],
            "treatment_plan": ["治疗方案", "首选", "替代", "方案", "建议使用"],
            "symptom": ["症状", "表现为", "主诉", "体征", "疼痛", "发热"],
        }

        for entry in entries:
            content = entry.get("content", "")
            importance = entry.get("importance", 0.5)

            fact_boost = 0.0
            for ftype, keywords in fact_keywords.items():
                if any(kw in content for kw in keywords):
                    fact_boost = max(fact_boost, self.config.fact_types.get(ftype, 0.5))

            if "redundant" not in entry.get("metadata", {}):
                is_redundant = any(
                    kw in content.lower()
                    for kw in ["再次强调", "综上所述", "如前所述", "重复一遍"]
                )
                if is_redundant:
                    fact_boost = self.config.fact_types["redundant"]

            entry["compaction_score"] = 0.5 * importance + 0.5 * fact_boost

        return entries

    def _select_by_budget(self, scored: list[dict], max_tokens: int) -> list[dict]:
        """按分数排序，预算内保留"""
        scored.sort(key=lambda x: x.get("compaction_score", 0.5), reverse=True)
        kept = []
        budget = max_tokens
        for entry in scored:
            content = entry.get("content", "")
            tokens = entry.get("token_count", TokenEstimator.estimate(content))
            if tokens <= budget:
                kept.append(entry)
                budget -= tokens
            elif budget > 100:
                entry["content"] = TokenEstimator.truncate(content, budget)
                kept.append(entry)
                break
            else:
                break
        return kept

    def _format_compacted(self, kept: list[dict]) -> str:
        """格式化为结构化摘要"""
        if not kept:
            return ""
        parts = ["[对话压缩摘要]"]

        seen_types = set()
        for entry in kept:
            content = entry.get("content", "")
            fact_type = self._classify_fact_type(content)
            if fact_type and fact_type not in seen_types:
                seen_types.add(fact_type)
            parts.append(content[:200])

        return "\n".join(parts)

    def _classify_fact_type(self, content: str) -> str:
        for ftype in ["diagnosis", "contraindication", "drug_interaction", "dosage", "treatment_plan"]:
            for kw in {
                "diagnosis": ["诊断", "确诊"],
                "contraindication": ["禁忌", "禁用"],
                "drug_interaction": ["相互作用", "联用"],
                "dosage": ["剂量", "mg"],
                "treatment_plan": ["方案", "首选"],
            }[ftype]:
                if kw in content:
                    return ftype
        return ""

    def _compute_doc_relevance(self, content: str, query_terms: set) -> float:
        if not query_terms:
            return 0.5
        content_lower = content.lower()
        matched = sum(1 for t in query_terms if t.lower() in content_lower)
        return min(matched / max(len(query_terms), 1), 1.0)

    def _detect_fact_importance(self, content: str) -> float:
        for ftype in ["diagnosis", "contraindication", "drug_interaction"]:
            for kw in {
                "diagnosis": ["诊断", "确诊", "判断为"],
                "contraindication": ["禁忌", "禁用", "避免使用"],
                "drug_interaction": ["相互作用", "联用风险"],
            }[ftype]:
                if kw in content:
                    return self.config.fact_types[ftype]
        return 0.5

    def _extractive_snip(self, content: str, max_tokens: int, query_terms: set) -> str:
        """抽取式截取 — 保留包含查询关键词最多的段落"""
        estimated = TokenEstimator.estimate(content)
        if estimated <= max_tokens:
            return content
        ratio = max_tokens / max(estimated, 1)
        sentences = content.replace("。", "。\n").split("\n")
        scored_sents = []
        for sent in sentences:
            if not sent.strip():
                continue
            score = sum(1 for t in query_terms if t.lower() in sent.lower())
            scored_sents.append((score, sent))
        scored_sents.sort(key=lambda x: x[0], reverse=True)
        kept = []
        budget = max_tokens
        for score, sent in scored_sents:
            tokens = TokenEstimator.estimate(sent)
            if tokens <= budget:
                kept.append(sent)
                budget -= tokens
        return "\n".join(kept)

    def _fallback_compact(self, content: str, max_tokens: int) -> str:
        return TokenEstimator.truncate(content, max_tokens)
