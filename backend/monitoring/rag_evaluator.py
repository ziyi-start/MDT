"""RAG 评估器 - 检索召回率、答案准确率、忠实度评估

核心指标:
  - Retrieval: hit_rate@k, recall@k, MRR
  - Generation: answer_accuracy (exact match in options)
  - Quality: faithfulness, relevance (LLM-as-judge)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievalEvalResult:
    """单次检索评估结果"""
    query: str = ""
    num_retrieved: int = 0
    num_relevant: int = 0
    hit_at_k: dict[int, bool] = field(default_factory=dict)
    reciprocal_rank: float = 0.0

    @property
    def recall_at_1(self) -> float:
        return self.hit_at_k.get(1, False) and 1.0 or 0.0

    @property
    def recall_at_3(self) -> float:
        return any(self.hit_at_k.get(i, False) for i in range(1, 4)) and 1.0 or 0.0

    @property
    def recall_at_5(self) -> float:
        return any(self.hit_at_k.get(i, False) for i in range(1, 6)) and 1.0 or 0.0

    @property
    def recall_at_10(self) -> float:
        return any(self.hit_at_k.get(i, False) for i in range(1, 11)) and 1.0 or 0.0


@dataclass
class GenerationEvalResult:
    """生成评估结果"""
    query: str = ""
    predicted_answer: str = ""
    expected_answer: str = ""
    is_correct: bool = False
    faithful: Optional[bool] = None
    relevant: Optional[bool] = None
    confidence: float = 0.0


@dataclass
class EvalReport:
    """评估汇总报告"""
    total_queries: int = 0
    retrieval_results: list[RetrievalEvalResult] = field(default_factory=list)
    generation_results: list[GenerationEvalResult] = field(default_factory=list)

    @property
    def retrieval_mrr(self) -> float:
        if not self.retrieval_results:
            return 0.0
        return sum(r.reciprocal_rank for r in self.retrieval_results) / len(self.retrieval_results)

    @property
    def retrieval_hit1(self) -> float:
        if not self.retrieval_results:
            return 0.0
        return sum(r.recall_at_1 for r in self.retrieval_results) / len(self.retrieval_results)

    @property
    def retrieval_hit3(self) -> float:
        if not self.retrieval_results:
            return 0.0
        return sum(r.recall_at_3 for r in self.retrieval_results) / len(self.retrieval_results)

    @property
    def retrieval_hit5(self) -> float:
        if not self.retrieval_results:
            return 0.0
        return sum(r.recall_at_5 for r in self.retrieval_results) / len(self.retrieval_results)

    @property
    def retrieval_hit10(self) -> float:
        if not self.retrieval_results:
            return 0.0
        return sum(r.recall_at_10 for r in self.retrieval_results) / len(self.retrieval_results)

    @property
    def generation_accuracy(self) -> float:
        if not self.generation_results:
            return 0.0
        return sum(1 for r in self.generation_results if r.is_correct) / len(self.generation_results)

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "retrieval": {
                "mrr": round(self.retrieval_mrr, 4),
                "hit@1": round(self.retrieval_hit1, 4),
                "hit@3": round(self.retrieval_hit3, 4),
                "hit@5": round(self.retrieval_hit5, 4),
                "hit@10": round(self.retrieval_hit10, 4),
            },
            "generation": {
                "accuracy": round(self.generation_accuracy, 4),
                "correct": sum(1 for r in self.generation_results if r.is_correct),
                "total": len(self.generation_results),
            },
        }


class RetrievalEvaluator:
    """检索质量评估器

    评估指标:
      - MRR (Mean Reciprocal Rank)
      - Hit@k (k=1,3,5,10)
    """

    def evaluate(
        self,
        query: str,
        retrieved_docs: list[dict],
        relevant_ids: set[str],
    ) -> RetrievalEvalResult:
        result = RetrievalEvalResult(
            query=query,
            num_retrieved=len(retrieved_docs),
            num_relevant=len(relevant_ids),
        )

        if not retrieved_docs or not relevant_ids:
            return result

        for rank, doc in enumerate(retrieved_docs, start=1):
            doc_id = doc.get("doc_id", doc.get("id", ""))
            if doc_id in relevant_ids:
                if result.reciprocal_rank == 0:
                    result.reciprocal_rank = 1.0 / rank
                result.hit_at_k[rank] = True
            else:
                result.hit_at_k[rank] = False

        return result


class GenerationEvaluator:
    """生成质量评估器

    评估：
      - 答案准确率：预测选项 vs 标准答案（多选题）
      - 忠实度/相关性：LLM-as-judge
    """

    def evaluate_accuracy(
        self,
        predicted_answer: str,
        expected_answer: str,
        options: dict[str, str] | None = None,
    ) -> GenerationEvalResult:
        result = GenerationEvalResult(
            predicted_answer=predicted_answer,
            expected_answer=expected_answer,
        )

        # 直接匹配
        if predicted_answer.strip().upper() == expected_answer.strip().upper():
            result.is_correct = True
            return result

        # 模糊匹配：预测中是否包含正确答案
        if expected_answer and expected_answer in predicted_answer:
            result.is_correct = True
            return result

        # 通过选项映射匹配
        if options:
            for key, value in options.items():
                if expected_answer.strip() == key.strip():
                    if value.strip() in predicted_answer:
                        result.is_correct = True
                        return result

        return result

    async def evaluate_faithfulness(
        self,
        llm,
        answer: str,
        retrieved_docs: list[str],
    ) -> bool:
        """LLM-as-judge: 评估答案是否基于检索文档（非幻觉）"""
        prompt = f"""评估以下回答是否基于提供的文档。

回答: {answer}

参考文档:
{chr(10).join(f'- {d[:500]}' for d in retrieved_docs[:5])}

回答是否可以从参考文档中推导？
输出 JSON: {{"faithful": true/false, "reason": "理由"}}"""

        try:
            from schema.messages import Message
            resp = await llm.chat(
                messages=[Message(role="user", content=prompt)],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            import json
            data = json.loads(resp.content or '{"faithful": false}')
            return data.get("faithful", False)
        except Exception as e:
            logger.warning(f"Faithfulness evaluation failed: {e}")
            return False
