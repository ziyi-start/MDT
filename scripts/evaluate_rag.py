"""RAG 评测脚本（改进版）

修复了 5 个流程问题:
  1. 批量编码: 所有 query 一次性编码，模型只加载一次
  2. CPU 高效: 批量编码在 CPU 上比逐个编码快 20-50 倍
  3. 断点续评: 每道题即时写 partial 文件, --resume 跳过已完成题
  4. 中间质量: 生成评测同步显示检索 MRR/Hit@k
  5. 回归基线: --save-baseline / --compare-baseline

用法:
  生成评测:  python scripts/evaluate_rag.py --num 100
  检索评测:  python scripts/evaluate_rag.py --mode retrieval --num 100
  断点续评:  python scripts/evaluate_rag.py --num 100 --resume
  API 模式:  python scripts/evaluate_rag.py --api-url http://localhost:8000 --num 100
  保存基线:  python scripts/evaluate_rag.py --num 50 --save-baseline data/baseline.json
  对比基线:  python scripts/evaluate_rag.py --num 50 --compare-baseline data/baseline.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from config import cfg
from rag.milvus_client import MilvusManager
from monitoring.rag_evaluator import GenerationEvaluator, EvalReport, RetrievalEvalResult
from llm.client import AsyncLLMClient
from schema.messages import Message

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TEST_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "textbooks", "zh_raw",
    "data_clean", "questions", "Mainland", "test.jsonl",
)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PARTIAL_RESULTS_PATH = os.path.join(DATA_DIR, "eval_partial.jsonl")

RETRIEVAL_PROMPT = """你是一位医学考试助手。请根据以下检索到的医学教科书内容，回答单选题。

检索到的参考内容：
{context}

问题：{question}

选项：
{options}

请只输出正确答案的字母（A/B/C/D/E），不要输出解释。"""


# ============================================================
# 加载题目
# ============================================================

def load_test_questions(path: str, num: int = 0) -> list[dict]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            questions.append(json.loads(line))
            if num and len(questions) >= num:
                break
    logger.info(f"已加载 {len(questions)} 道测试题")
    return questions


# ============================================================
# 批量编码 (修复 #1 & #2)
# ============================================================

def _load_bge_model():
    """延迟加载 BGE 模型，确保只加载一次"""
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"加载 BGE Embedding 模型 (device={device})...")
        model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device=device)
        logger.info(f"BGE 模型加载成功，向量维度: {model.get_embedding_dimension()}")
        return model, device
    except Exception as e:
        logger.error(f"BGE 模型加载失败: {e}")
        raise


def batch_embed_all(queries: list[str], batch_size: int = 64) -> list[list[float]]:
    """批量编码所有 query，一次前向传播完成"""
    model, device = _load_bge_model()
    t0 = time.time()
    vectors = model.encode(
        queries,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(queries) > 10,
    )
    elapsed = time.time() - t0
    logger.info(
        f"批量编码完成: {len(queries)} 条, {elapsed:.1f}s "
        f"({elapsed / max(len(queries), 1) * 1000:.1f}ms/条, device={device})"
    )
    return [v.tolist() for v in vectors]


# ============================================================
# 断点续评 (修复 #3)
# ============================================================

def load_partial_results(path: str) -> dict[int, dict]:
    """加载已完成的部分结果，返回 {idx: result_dict}"""
    done = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done[r["idx"]] = r
                except json.JSONDecodeError:
                    continue
        if done:
            logger.info(f"已加载 {len(done)} 条已完成结果 (--resume)")
    return done


def save_partial_result(path: str, result: dict):
    """追加一条部分结果到 JSONL"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ============================================================
# 关键词相关度 (用于无标注检索评估)
# ============================================================

def _keyword_relevance(query: str, doc_content: str) -> float:
    q_terms = set(re.findall(r'[\u4e00-\u9fff]{2,}', query))
    if not q_terms:
        return 0.0
    d_terms = set(re.findall(r'[\u4e00-\u9fff]{2,}', doc_content))
    overlap = q_terms & d_terms
    return len(overlap) / len(q_terms)


# ============================================================
# 检索评测
# ============================================================

async def evaluate_retrieval(
    milvus: MilvusManager,
    questions: list[dict],
    top_k: int = 10,
    batch_size: int = 64,
    resume: bool = False,
):
    logger.info(f"=== 检索召回率评测 (top_k={top_k}) ===")
    report = EvalReport()

    # 断点续评
    partial = load_partial_results(PARTIAL_RESULTS_PATH) if resume else {}

    # 批量编码 query (修复 #1) — resume 全部完成时跳过
    todo = [i for i in range(len(questions)) if i not in partial]
    if todo:
        queries = [questions[i]["question"] for i in todo]
        encoded = batch_embed_all(queries, batch_size)
        all_vectors = {todo[j]: encoded[j] for j in range(len(todo))}
    else:
        all_vectors = {}

    t0 = time.time()
    for i, q in enumerate(questions):
        if i in partial:
            result = RetrievalEvalResult(
                query=partial[i].get("query", "")[:80],
                num_retrieved=partial[i].get("num_retrieved", 0),
            )
            result.reciprocal_rank = partial[i].get("mrr_score", 0.0)
            result.num_relevant = partial[i].get("num_relevant", 0)
            for k, v in partial[i].get("hit_at_k", {}).items():
                result.hit_at_k[int(k)] = v
            report.retrieval_results.append(result)
            report.total_queries += 1
            continue

        query = q["question"]
        vec = all_vectors[i]
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

        # 即时保存 (修复 #3)
        save_partial_result(PARTIAL_RESULTS_PATH, {
            "idx": i,
            "query": query[:80],
            "num_retrieved": len(hits),
            "num_relevant": result.num_relevant,
            "mrr_score": result.reciprocal_rank,
            "hit_at_k": {str(k): v for k, v in result.hit_at_k.items()},
        })

        if (i + 1) % 50 == 0:
            d = report.to_dict()
            logger.info(
                f"  进度: {i + 1}/{len(questions)} | "
                f"Hit@1={d['retrieval']['hit@1']:.3f} "
                f"Hit@5={d['retrieval']['hit@5']:.3f} | {time.time() - t0:.0f}s"
            )

    print_retrieval_report(report)
    logger.info(f"检索评测完成, 耗时 {time.time() - t0:.1f}s")
    return report


# ============================================================
# 生成评测
# ============================================================

async def evaluate_generation(
    llm: AsyncLLMClient,
    milvus: MilvusManager,
    questions: list[dict],
    top_k: int = 5,
    batch_size: int = 64,
    resume: bool = False,
    faithfulness: bool = False,
):
    logger.info(f"=== 生成准确率评测 (top_k={top_k}) ===")
    gen_eval = GenerationEvaluator()
    report = EvalReport()

    partial = load_partial_results(PARTIAL_RESULTS_PATH) if resume else {}

    # 批量编码 query (修复 #1) — resume 全部完成时跳过
    todo = [i for i in range(len(questions)) if i not in partial]
    if todo:
        queries = [questions[i]["question"] for i in todo]
        encoded = batch_embed_all(queries, batch_size)
        all_vectors = {todo[j]: encoded[j] for j in range(len(todo))}
    else:
        all_vectors = {}

    t0 = time.time()
    for i, q in enumerate(questions):
        if i in partial:
            r = partial[i]
            result = gen_eval.evaluate_accuracy(
                predicted_answer=r.get("predicted", ""),
                expected_answer=r.get("expected", ""),
            )
            result.query = r.get("query", "")[:80]
            report.total_queries += 1
            report.generation_results.append(result)
            # 同时记录检索指标
            ret_result = RetrievalEvalResult(query=r.get("query", "")[:80])
            ret_result.reciprocal_rank = r.get("retrieval_mrr", 0.0)
            ret_result.num_relevant = r.get("num_relevant", 0)
            for k, v in r.get("hit_at_k", {}).items():
                ret_result.hit_at_k[int(k)] = v
            report.retrieval_results.append(ret_result)
            continue

        query = q["question"]
        answer = q.get("answer", "")
        answer_idx = q.get("answer_idx", "")
        options = q.get("options", {})

        # 检索
        vec = all_vectors[i]
        hits = milvus.search(cfg.milvus.collections.kb, vec, limit=top_k)

        # 记录检索质量 (修复 #4)
        ret_result = RetrievalEvalResult(query=query[:80], num_retrieved=len(hits))
        for rank, hit in enumerate(hits, start=1):
            content = hit.get("content", "")
            if _keyword_relevance(query, content) >= 0.1:
                if ret_result.reciprocal_rank == 0:
                    ret_result.reciprocal_rank = 1.0 / rank
                ret_result.hit_at_k[rank] = True
                ret_result.num_relevant += 1
            else:
                ret_result.hit_at_k[rank] = False

        # 构造 prompt
        context = "\n\n".join(h.get("content", "")[:500] for h in hits[:top_k])
        options_text = "\n".join(f"{k}. {v}" for k, v in options.items()) if isinstance(options, dict) else str(options)
        prompt = RETRIEVAL_PROMPT.format(context=context[:3000], question=query, options=options_text)

        # LLM 生成
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

        # 评判
        gen_result = gen_eval.evaluate_accuracy(
            predicted_answer=predicted,
            expected_answer=answer_idx or answer,
            options=options if isinstance(options, list) else None,
        )
        gen_result.query = query[:80]

        # Faithfulness (修复 #4)
        if faithfulness and gen_result.is_correct:
            try:
                doc_texts = [h.get("content", "") for h in hits[:top_k]]
                gen_result.faithful = await gen_eval.evaluate_faithfulness(llm, predicted, doc_texts)
            except Exception:
                gen_result.faithful = None

        report.total_queries += 1
        report.generation_results.append(gen_result)
        report.retrieval_results.append(ret_result)

        # 即时保存 (修复 #3)
        hit_at_k = {str(k): v for k, v in ret_result.hit_at_k.items()}
        save_partial_result(PARTIAL_RESULTS_PATH, {
            "idx": i,
            "query": query[:80],
            "predicted": predicted,
            "expected": answer_idx or answer,
            "is_correct": gen_result.is_correct,
            "faithful": gen_result.faithful,
            "retrieval_mrr": ret_result.reciprocal_rank,
            "num_relevant": ret_result.num_relevant,
            "hit_at_k": hit_at_k,
        })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            acc = report.generation_accuracy
            d = report.to_dict()
            logger.info(
                f"  进度: {i + 1}/{len(questions)} | "
                f"Acc={acc:.3f} | "
                f"Hit@3={d['retrieval']['hit@3']:.3f} | "
                f"{elapsed:.0f}s"
            )

    elapsed = time.time() - t0
    logger.info(f"生成评测完成, 耗时 {elapsed:.1f}s ({elapsed / 60:.1f} 分钟)")

    print_retrieval_report(report)
    print_generation_report(report)

    # Export JSON
    out_path = os.path.join(DATA_DIR, "eval_report_gen.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    logger.info(f"报告已保存: {out_path}")

    return report


# ============================================================
# 输出报告
# ============================================================

def print_retrieval_report(report: EvalReport):
    d = report.to_dict()
    print("\n" + "=" * 50)
    print("检索质量报告")
    print("=" * 50)
    print(f"题目数: {d['total_queries']}")
    print(f"MRR:     {d['retrieval']['mrr']:.4f}")
    print(f"Hit@1:   {d['retrieval']['hit@1']:.2%}")
    print(f"Hit@3:   {d['retrieval']['hit@3']:.2%}")
    print(f"Hit@5:   {d['retrieval']['hit@5']:.2%}")
    print(f"Hit@10:  {d['retrieval']['hit@10']:.2%}")


def print_generation_report(report: EvalReport):
    d = report.to_dict()
    print("\n" + "=" * 50)
    print("生成准确率报告")
    print("=" * 50)
    print(f"题目数: {d['total_queries']}")
    print(f"准确率: {d['generation']['accuracy']:.2%} "
          f"({d['generation']['correct']}/{d['generation']['total']})")

    # Faithfulness (修复 #4)
    faithful_results = [r for r in report.generation_results if r.faithful is not None]
    if faithful_results:
        faithful_count = sum(1 for r in faithful_results if r.faithful)
        print(f"忠实度: {faithful_count}/{len(faithful_results)} ({faithful_count / len(faithful_results):.1%})")


# ============================================================
# API 模式 (修复 #1: 无需本地加载模型)
# ============================================================

async def evaluate_via_api(api_url: str, questions: list[dict], resume: bool = False):
    """通过运行中的 API 服务器进行评测，无需本地加载任何模型"""
    import httpx

    logger.info(f"=== API 模式评测 (服务地址: {api_url}) ===")
    partial = load_partial_results(PARTIAL_RESULTS_PATH) if resume else {}

    correct = 0
    total = 0
    t0 = time.time()

    async with httpx.AsyncClient(timeout=120.0) as client:
        for i, q in enumerate(questions):
            if i in partial:
                total += 1
                if partial[i].get("is_correct", False):
                    correct += 1
                continue

            try:
                resp = await client.post(
                    f"{api_url}/api/query",
                    json={"query": q["question"]},
                )
                resp.raise_for_status()
                data = resp.json()
                predicted = (data.get("answer", "") or "").strip()
            except Exception as e:
                logger.warning(f"API 调用失败 [{i}]: {e}")
                predicted = ""

            expected = q.get("answer_idx", q.get("answer", ""))
            is_correct = predicted and expected and predicted.strip().upper() == expected.strip().upper()
            if not is_correct and expected and predicted:
                is_correct = expected in predicted  # 模糊匹配

            if is_correct:
                correct += 1
            total += 1

            save_partial_result(PARTIAL_RESULTS_PATH, {
                "idx": i,
                "query": q["question"][:80],
                "predicted": predicted,
                "expected": expected,
                "is_correct": is_correct,
            })

            acc = correct / total if total else 0
            if (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                logger.info(f"  进度: {i + 1}/{len(questions)} | Acc={acc:.3f} | {elapsed:.0f}s")

    elapsed = time.time() - t0
    acc = correct / total if total else 0
    print("\n" + "=" * 50)
    print("API 模式评测报告")
    print("=" * 50)
    print(f"题目数: {total}")
    print(f"准确率: {acc:.2%} ({correct}/{total})")
    print(f"耗时:   {elapsed:.1f}s ({elapsed / 60:.1f} 分钟)")

    return {"accuracy": acc, "correct": correct, "total": total, "elapsed_s": elapsed}


# ============================================================
# 回归基线 (修复 #5)
# ============================================================

def save_baseline(partial_path: str, baseline_path: str):
    """从 partial 文件提取精简基线（自动去重，保留每题的最终结果）"""
    if not os.path.exists(partial_path):
        logger.error(f"Partial 文件不存在: {partial_path}")
        return
    seen = {}
    with open(partial_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            seen[r["idx"]] = {
                "idx": r["idx"],
                "query": r.get("query", ""),
                "predicted": r.get("predicted", ""),
                "expected": r.get("expected", ""),
                "is_correct": r.get("is_correct", False),
            }
    baseline = [seen[k] for k in sorted(seen)]
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, ensure_ascii=False, indent=2)
    correct = sum(1 for b in baseline if b["is_correct"])
    acc = correct / len(baseline) if baseline else 0
    logger.info(f"基线已保存: {baseline_path} ({correct}/{len(baseline)} 正确, Acc={acc:.1%})")


def compare_baseline(baseline_path: str, partial_path: str):
    """对比基线，报告差异"""
    if not os.path.exists(baseline_path):
        logger.error(f"基线文件不存在: {baseline_path}")
        return
    if not os.path.exists(partial_path):
        logger.error(f"当前结果文件不存在: {partial_path}")
        return

    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = {b["idx"]: b for b in json.load(f)}

    with open(partial_path, "r", encoding="utf-8") as f:
        current = {r["idx"]: r for r in (json.loads(line) for line in f)}

    # 统计
    base_correct = sum(1 for b in baseline.values() if b["is_correct"])
    curr_correct = sum(1 for c in current.values() if c.get("is_correct", False))
    base_acc = base_correct / len(baseline) if baseline else 0
    curr_acc = curr_correct / len(current) if current else 0

    # 找变化
    same = set(baseline.keys()) & set(current.keys())
    regressions = []
    improvements = []
    for idx in same:
        b = baseline[idx]["is_correct"]
        c = current[idx].get("is_correct", False)
        if b and not c:
            regressions.append(idx)
        elif not b and c:
            improvements.append(idx)

    print("\n" + "=" * 50)
    print("回归基线对比")
    print("=" * 50)
    print(f"题目数:     {len(same)}")
    print(f"基线准确率: {base_acc:.1%} ({base_correct}/{len(baseline)})")
    print(f"当前准确率: {curr_acc:.1%} ({curr_correct}/{len(current)})")
    print(f"变化:       {curr_acc - base_acc:+.1%}")
    print(f"进步:       {len(improvements)} 题")
    print(f"退步:       {len(regressions)} 题")

    if regressions:
        print(f"\n退步题目 (原来正确, 现在错误):")
        for idx in sorted(regressions)[:10]:
            q = baseline[idx].get("query", "")[:80]
            print(f"  [{idx}] {q}")
            print(f"       基线: {baseline[idx]['predicted']} | 当前: {current[idx].get('predicted', '?')}")
        if len(regressions) > 10:
            print(f"  ... 共 {len(regressions)} 题")

    if improvements:
        print(f"\n进步题目 (原来错误, 现在正确):")
        for idx in sorted(improvements)[:5]:
            q = baseline[idx].get("query", "")[:80]
            print(f"  [{idx}] {q}")


# ============================================================
# 主入口
# ============================================================

async def main():
    parser = argparse.ArgumentParser(description="RAG 评测脚本 (改进版)")
    parser.add_argument("--num", type=int, default=0, help="评测题目数 (0=全部)")
    parser.add_argument("--top_k", type=int, default=5, help="检索返回数")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding 批量大小")
    parser.add_argument("--mode", choices=["retrieval", "generation", "both"], default="generation")
    parser.add_argument("--resume", action="store_true", help="从上次中断处继续 (断点续评)")
    parser.add_argument("--faithfulness", action="store_true", help="启用 LLM 忠实度评估")
    parser.add_argument("--api-url", type=str, default="", help="API 模式: 调用运行中的服务器进行评估")
    parser.add_argument("--save-baseline", type=str, default="", help="将当前结果保存为回归基线")
    parser.add_argument("--compare-baseline", type=str, default="", help="与指定基线文件对比")
    parser.add_argument("--no-save", action="store_true", help="不保存 partial 结果")
    global PARTIAL_RESULTS_PATH
    args = parser.parse_args()

    if args.no_save:
        PARTIAL_RESULTS_PATH = os.path.join(DATA_DIR, "eval_partial_temp.jsonl")

    # 回归基线对比 (修复 #5)
    if args.compare_baseline:
        compare_baseline(args.compare_baseline, PARTIAL_RESULTS_PATH)
        return

    if args.save_baseline:
        save_baseline(PARTIAL_RESULTS_PATH, args.save_baseline)
        return

    # 如果指定了 --resume 但 partial 文件不存在，重置
    if args.resume and not os.path.exists(PARTIAL_RESULTS_PATH):
        logger.info("--resume 指定但无已完成结果，从头开始")

    questions = load_test_questions(TEST_PATH, args.num)

    # API 模式 (修复 #1: 无需本地模型)
    if args.api_url:
        await evaluate_via_api(args.api_url, questions, resume=args.resume)
        return

    milvus = MilvusManager(uri=cfg.milvus.uri)
    milvus.connect()

    if args.mode in ("generation", "both"):
        llm = AsyncLLMClient()
        await evaluate_generation(
            llm, milvus, questions,
            top_k=args.top_k,
            batch_size=args.batch_size,
            resume=args.resume,
            faithfulness=args.faithfulness,
        )

    if args.mode in ("retrieval", "both"):
        await evaluate_retrieval(
            milvus, questions,
            top_k=args.top_k,
            batch_size=args.batch_size,
            resume=args.resume,
        )


if __name__ == "__main__":
    asyncio.run(main())
