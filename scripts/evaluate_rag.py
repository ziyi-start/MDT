"""RAG 检索 + 生成评测脚本

使用 MedQA test.jsonl 对 RAG 管道进行端到端评测：
  1. 检索召回率: MRR, Hit@k (评估 Milvus 检索质量)
  2. 答案准确率: 预测 vs 标准答案 (评估 LLM 生成质量)

用法:
  python scripts/evaluate_rag.py [--num 100] [--mode retrieval|generation|both]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from config import cfg
from rag.milvus_client import MilvusManager
from rag.embedding import dummy_embed
from monitoring.rag_evaluator import RetrievalEvaluator, GenerationEvaluator, EvalReport, RetrievalEvalResult
from llm.client import AsyncLLMClient
from schema.messages import Message

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TEST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "textbooks", "zh_raw",
    "data_clean", "questions", "Mainland", "test.jsonl",
)

RETRIEVAL_PROMPT = """你是一位医学考试助手。请根据以下检索到的医学教科书内容，回答单选题。

检索到的参考内容：
{context}

问题：{question}

选项：
{options}

请只输出正确答案的字母（A/B/C/D/E），不要输出解释。"""


def load_test_questions(path: str, num: int = 0) -> list[dict]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
            if num and len(questions) >= num:
                break
    logger.info(f"已加载 {len(questions)} 道测试题")
    return questions


def _keyword_relevance(query: str, doc_content: str) -> float:
    """关键词匹配判断检索文档与问题的相关度"""
    import re
    q_terms = set(re.findall(r'[\u4e00-\u9fff]{2,}', query))
    if not q_terms:
        return 0.0
    d_terms = set(re.findall(r'[\u4e00-\u9fff]{2,}', doc_content))
    overlap = q_terms & d_terms
    return len(overlap) / len(q_terms)


async def evaluate_retrieval(
    milvus: MilvusManager,
    questions: list[dict],
    top_k: int = 10,
):
    """评估检索召回率 (仅需要 Milvus，不需要 LLM)"""
    logger.info(f"=== 检索召回率评测 (top_k={top_k}) ===")
    t0 = time.time()
    report = EvalReport()

    for i, q in enumerate(questions):
        query = q["question"]
        vec = await dummy_embed(query)
        hits = milvus.search(cfg.milvus.collections.kb, vec, limit=top_k)

        result = RetrievalEvalResult(query=query[:80], num_retrieved=len(hits))
        for rank, hit in enumerate(hits, start=1):
            content = hit.get("content", "")
            if _keyword_relevance(query, content) >= 0.1:
                if result.reciprocal_rank == 0:
                    result.reciprocal_rank = 1.0 / rank
                result.hit_at_k[rank] = True
                result.num_relevant += 1
            else:
                result.hit_at_k[rank] = False

        report.total_queries += 1
        report.retrieval_results.append(result)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            d = report.to_dict()
            logger.info(
                f"  进度: {i + 1}/{len(questions)} | "
                f"Hit@1={d['retrieval']['hit@1']:.3f} "
                f"Hit@5={d['retrieval']['hit@5']:.3f} | {elapsed:.0f}s"
            )

    d = report.to_dict()
    print("\n" + "=" * 50)
    print("检索召回率报告")
    print("=" * 50)
    print(f"题目数: {d['total_queries']}")
    print(f"MRR:     {d['retrieval']['mrr']:.4f}")
    print(f"Hit@1:   {d['retrieval']['hit@1']:.2%}")
    print(f"Hit@3:   {d['retrieval']['hit@3']:.2%}")
    print(f"Hit@5:   {d['retrieval']['hit@5']:.2%}")
    print(f"Hit@10:  {d['retrieval']['hit@10']:.2%}")

    logger.info(f"检索评测完成, 耗时 {time.time() - t0:.1f}s")
    return report


async def evaluate_generation(
    llm: AsyncLLMClient,
    milvus: MilvusManager,
    questions: list[dict],
    top_k: int = 5,
):
    """评估生成准确率 (LLM 回答单选题)"""
    logger.info(f"=== 生成准确率评测 (top_k={top_k}) ===")
    t0 = time.time()
    gen_eval = GenerationEvaluator()
    report = EvalReport()

    for i, q in enumerate(questions):
        query = q["question"]
        answer = q.get("answer", "")
        answer_idx = q.get("answer_idx", "")
        options = q.get("options", {})

        # 1. 检索
        vec = await dummy_embed(query)
        hits = milvus.search(cfg.milvus.collections.kb, vec, limit=top_k)

        # 2. 构造 prompt
        context = "\n\n".join(h.get("content", "")[:500] for h in hits[:top_k])
        options_text = "\n".join(f"{k}. {v}" for k, v in options.items()) if isinstance(options, dict) else str(options)

        prompt = RETRIEVAL_PROMPT.format(
            context=context[:3000],
            question=query,
            options=options_text,
        )

        # 3. LLM 生成
        try:
            resp = await llm.chat(
                messages=[Message(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=50,
            )
            predicted = (resp.content or "").strip()
        except Exception as e:
            logger.warning(f"LLM 调用失败 [{i}]: {e}")
            predicted = ""

        # 4. 评判
        result = gen_eval.evaluate_accuracy(
            predicted_answer=predicted,
            expected_answer=answer_idx or answer,
            options=options if isinstance(options, list) else None,
        )
        result.query = query[:80]

        report.total_queries += 1
        report.generation_results.append(result)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            acc = report.generation_accuracy
            logger.info(f"  进度: {i + 1}/{len(questions)} | 准确率: {acc:.3f} | {elapsed:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"生成评测完成, 耗时 {elapsed:.1f}s ({elapsed / 60:.1f} 分钟)")

    # 打印报告
    d = report.to_dict()
    print("\n" + "=" * 50)
    print("生成准确率报告")
    print("=" * 50)
    print(f"题目数: {d['total_queries']}")
    print(f"准确率: {d['generation']['accuracy']:.2%} ({d['generation']['correct']}/{d['generation']['total']})")

    # Export JSON
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "eval_report_gen.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    logger.info(f"报告已保存: {out_path}")

    return report


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=0, help="评测题目数 (0=全部)")
    parser.add_argument("--top_k", type=int, default=5, help="检索返回数")
    parser.add_argument("--mode", choices=["retrieval", "generation", "both"], default="generation")
    args = parser.parse_args()

    # 加载题目
    questions = load_test_questions(TEST_PATH, args.num)

    # 连接 Milvus
    milvus = MilvusManager(uri=cfg.milvus.uri)
    milvus.connect()

    if args.mode in ("generation", "both"):
        # 初始化 LLM
        llm = AsyncLLMClient()
        await evaluate_generation(llm, milvus, questions, args.top_k)

    if args.mode in ("retrieval", "both"):
        await evaluate_retrieval(milvus, questions, args.top_k)


if __name__ == "__main__":
    asyncio.run(main())
