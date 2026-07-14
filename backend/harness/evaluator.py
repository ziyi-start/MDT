"""Harness 评估引擎 - 7维确定性评分

核心设计理念（来自 Harness Engineering）:
  "宁要可复现的粗糙分，不要会漂移的精准分"
  评测的唯一目的是驱动迭代 —— 只有多次跑分完全一致，才能回答"这次改规范到底变好还是变坏"

评分维度（参考 SWE-bench / AgentBench / CMMI 融合）:
  1. 流程完整性 (22%) - 该走的流程节点是否都走了
  2. 输出质量   (15%) - 回答是否有实质内容而非模板套话
  3. 答案正确性 (22%) - 答案是否能通过客观验证
  4. 效率       (10%) - 耗时和 token 消耗
  5. 安全合规   (8%)  - 是否违反医疗安全规则
  6. 迭代能力   (5%)  - 失败后能否自我修复
  7. 接口验收   (18%) - 最终答案经得起验证吗

关键设计决策:
  - 零 LLM 调用: 所有评分基于确定性规则和已有指标
  - 可复现: 相同输入 + 相同配置 → 完全相同评分
  - 与基线对比: 支持版本间 regression 检测
"""
from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

from schema.models import MedicalResponse

logger = logging.getLogger(__name__)


# ============================================================
# 7维评分定义
# ============================================================

@dataclass
class DimensionScore:
    name: str
    weight: float
    score: float = 0.0
    reason: str = ""

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


DIMENSION_DEFINITIONS = [
    {"name": "process_completeness", "label": "流程完整性", "weight": 0.22},
    {"name": "output_quality",       "label": "输出质量",   "weight": 0.15},
    {"name": "correctness",          "label": "答案正确性", "weight": 0.22},
    {"name": "efficiency",           "label": "效率",       "weight": 0.10},
    {"name": "safety_compliance",    "label": "安全合规",   "weight": 0.08},
    {"name": "iteration_capability", "label": "迭代能力",   "weight": 0.05},
    {"name": "interface_acceptance", "label": "接口验收",   "weight": 0.18},
]


# ============================================================
# 评估报告
# ============================================================

@dataclass
class EvaluationReport:
    config_hash: str = ""
    timestamp: float = 0.0
    total_score: float = 0.0
    dimensions: list[DimensionScore] = field(default_factory=list)
    num_queries: int = 0
    num_safe_fallback: int = 0
    num_escalated: int = 0
    avg_latency_ms: float = 0.0
    regression: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
            "total_score": round(self.total_score, 4),
            "dimensions": [
                {"name": d.name, "label": d.label if hasattr(d, 'label') else d.name,
                 "score": round(d.score, 4), "weight": d.weight, "weighted": round(d.weighted_score, 4),
                 "reason": d.reason}
                for d in self.dimensions
            ],
            "num_queries": self.num_queries,
            "num_safe_fallback": self.num_safe_fallback,
            "num_escalated": self.num_escalated,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "regression": self.regression,
        }


# ============================================================
# Harness 评估器
# ============================================================

class HarnessEvaluator:
    """7维确定性评分评估器

    用法:
        evaluator = HarnessEvaluator()
        evaluator.add_result(response, latency_ms)
        report = evaluator.evaluate()
    """

    def __init__(self, config_hash: str = ""):
        self.config_hash = config_hash
        self._responses: list[tuple[MedicalResponse, float]] = []
        self._start_time = time.time()

    def add_result(self, response: MedicalResponse, latency_ms: float):
        self._responses.append((response, latency_ms))

    def evaluate(self, baseline_report: Optional[dict] = None) -> EvaluationReport:
        if not self._responses:
            return EvaluationReport(config_hash=self.config_hash, timestamp=time.time())

        n = len(self._responses)
        responses = [r for r, _ in self._responses]
        latencies = [l for _, l in self._responses]

        num_safe_fallback = sum(1 for r in responses if r.is_safe_fallback)
        num_escalated = sum(1 for r in responses if r.route_path == "mdt_escalated")
        avg_latency = sum(latencies) / n if latencies else 0.0

        dimensions = self._score_dimensions(responses, latencies)

        total = sum(d.weighted_score for d in dimensions)

        report = EvaluationReport(
            config_hash=self.config_hash,
            timestamp=time.time(),
            total_score=total,
            dimensions=dimensions,
            num_queries=n,
            num_safe_fallback=num_safe_fallback,
            num_escalated=num_escalated,
            avg_latency_ms=avg_latency,
        )

        if baseline_report:
            report.regression = self._compare_regression(report.to_dict(), baseline_report)

        return report

    def _score_dimensions(
        self, responses: list[MedicalResponse], latencies: list[float]
    ) -> list[DimensionScore]:
        scores = []

        # 1. 流程完整性 (22%)
        scores.append(self._score_process_completeness(responses))

        # 2. 输出质量 (15%)
        scores.append(self._score_output_quality(responses))

        # 3. 答案正确性 (22%)
        scores.append(self._score_correctness(responses))

        # 4. 效率 (10%)
        scores.append(self._score_efficiency(latencies))

        # 5. 安全合规 (8%)
        scores.append(self._score_safety_compliance(responses))

        # 6. 迭代能力 (5%)
        scores.append(self._score_iteration_capability(responses))

        # 7. 接口验收 (18%)
        scores.append(self._score_interface_acceptance(responses, latencies))

        return scores

    # ---- 各维度评分 ----

    def _score_process_completeness(self, responses: list[MedicalResponse]) -> DimensionScore:
        """流程完整性: 检查是否走完了完整的处理链路

        检查点:
        - 是否包含了路由信息 (route_path)
        - MDT 路径是否有 departments
        - 是否有来源引用 (sources)
        - 是否有置信度评分
        """
        if not responses:
            return DimensionScore(name="process_completeness", weight=0.22, score=0.0, reason="无请求数据")

        checks_passed = 0
        total_checks = len(responses) * 4

        for r in responses:
            if r.route_path:
                checks_passed += 1
            if r.route_path in ("mdt", "mdt_escalated") and r.departments:
                checks_passed += 1
            elif r.route_path in ("simple_rag", "safe_fallback"):
                checks_passed += 1
            if r.sources:
                checks_passed += 1
            if r.confidence is not None and r.confidence > 0:
                checks_passed += 1

        score = checks_passed / max(total_checks, 1)
        reason = f"{checks_passed}/{total_checks} 流程节点完整"
        return DimensionScore(name="process_completeness", weight=0.22, score=score, reason=reason)

    def _score_output_quality(self, responses: list[MedicalResponse]) -> DimensionScore:
        """输出质量: 回答是否有实质内容

        检查:
        - 回答长度 > 50 字符
        - 非模板套话 (不含"抱歉""无法回答"等空泛短语)
        - 有具体引用来源
        """
        if not responses:
            return DimensionScore(name="output_quality", weight=0.15, score=0.0, reason="无请求数据")

        # 空泛短语检测
        hollow_phrases = ["抱歉", "无法回答", "请稍后重试", "建议线下就医"]
        checks_passed = 0
        total_checks = len(responses) * 3

        for r in responses:
            answer = r.answer or ""
            # 长度检查
            if len(answer) > 50:
                checks_passed += 1
            # 非空泛
            if not any(p in answer for p in hollow_phrases):
                checks_passed += 1
            # 有来源
            if r.sources:
                checks_passed += 1

        score = checks_passed / max(total_checks, 1)
        reason = f"{checks_passed}/{total_checks} 质量检查通过"
        return DimensionScore(name="output_quality", weight=0.15, score=score, reason=reason)

    def _score_correctness(self, responses: list[MedicalResponse]) -> DimensionScore:
        """答案正确性: 答案是否可靠

        基于置信度评分和退避率评估:
        - 高置信度 = 答案可靠
        - 低退避率 = 系统能回答问题
        """
        if not responses:
            return DimensionScore(name="correctness", weight=0.22, score=0.0, reason="无请求数据")

        n = len(responses)

        # 平均置信度
        avg_confidence = sum(r.confidence for r in responses) / n if n > 0 else 0.0

        # 退避率越低越好 (退避率 > 30% 扣分)
        fallback_rate = sum(1 for r in responses if r.is_safe_fallback) / n
        fallback_penalty = max(0.0, 1.0 - fallback_rate * 3)

        score = avg_confidence * 0.6 + fallback_penalty * 0.4
        score = max(0.0, min(1.0, score))

        reason = f"avg_confidence={avg_confidence:.3f}, fallback_rate={fallback_rate:.3f}"
        return DimensionScore(name="correctness", weight=0.22, score=score, reason=reason)

    def _score_efficiency(self, latencies: list[float]) -> DimensionScore:
        """效率: 耗时

        基准: 目标延迟 < 10s (10000ms)
        - 平均延迟低于 5s → 1.0
        - 平均延迟 5-10s → 0.5-1.0
        - 平均延迟 > 10s  → 惩罚
        """
        if not latencies:
            return DimensionScore(name="efficiency", weight=0.10, score=0.0, reason="无延迟数据")

        avg_latency = sum(latencies) / len(latencies)

        if avg_latency <= 5000:
            score = 1.0
            reason = f"avg={avg_latency:.0f}ms (优秀)"
        elif avg_latency <= 10000:
            score = 1.0 - (avg_latency - 5000) / 10000
            reason = f"avg={avg_latency:.0f}ms (可接受)"
        elif avg_latency <= 30000:
            score = max(0.0, 0.5 - (avg_latency - 10000) / 40000)
            reason = f"avg={avg_latency:.0f}ms (偏慢)"
        else:
            score = 0.1
            reason = f"avg={avg_latency:.0f}ms (超时)"

        return DimensionScore(name="efficiency", weight=0.10, score=score, reason=reason)

    def _score_safety_compliance(self, responses: list[MedicalResponse]) -> DimensionScore:
        """安全合规: 是否违反医疗安全规则

        检查:
        - 安全退避是否被正确触发 (该退避时退避了)
        - 非安全路径是否正确处理
        """
        if not responses:
            return DimensionScore(name="safety_compliance", weight=0.08, score=0.0, reason="无请求数据")

        n = len(responses)
        checks_passed = 0
        total_checks = n

        for r in responses:
            if r.is_safe_fallback:
                if r.confidence == 0.0:
                    checks_passed += 1
            else:
                if r.confidence >= 0.1:
                    checks_passed += 1

        score = checks_passed / max(total_checks, 1)
        return DimensionScore(name="safety_compliance", weight=0.08, score=score,
                              reason=f"安全合规 {checks_passed}/{total_checks}")

    def _score_iteration_capability(self, responses: list[MedicalResponse]) -> DimensionScore:
        """迭代能力: 系统在失败后能否自我修复

        当前实现: 检测是否有从 simple_rag 升级到 mdt 的路径
        即系统能在置信度不足时自动升级处理方式
        """
        if not responses:
            return DimensionScore(name="iteration_capability", weight=0.05, score=0.0, reason="无请求数据")

        n = len(responses)
        escalated = sum(1 for r in responses if r.route_path == "mdt_escalated")

        if n == 0:
            return DimensionScore(name="iteration_capability", weight=0.05, score=0.0, reason="无请求数据")

        escalation_rate = escalated / n if n > 0 else 0.0
        score = min(1.0, escalation_rate * 5)
        reason = f"升级率 {escalation_rate:.2f} ({escalated}/{n})"
        return DimensionScore(name="iteration_capability", weight=0.05, score=score, reason=reason)

    def _score_interface_acceptance(self, responses: list[MedicalResponse], latencies: list[float]) -> DimensionScore:
        """接口验收: 最终答案是否经得起验证

        检查:
        - MDT 路径是否通过了 Decision Maker 评估
        - 验证结果是否一致 (高置信度 = 通过验收)
        """
        if not responses:
            return DimensionScore(name="interface_acceptance", weight=0.18, score=0.0, reason="无请求数据")

        n = len(responses)
        mdt_responses = [r for r in responses if r.route_path in ("mdt", "mdt_escalated")]

        if not mdt_responses:
            # 没有 MDT 调用时: 基于 SimpleRAG 的置信度评分
            avg_confidence = sum(r.confidence for r in responses) / n if n > 0 else 0.0
            score = avg_confidence
            reason = f"simple_rag 路径, avg_confidence={avg_confidence:.3f}"
            return DimensionScore(name="interface_acceptance", weight=0.18, score=score, reason=reason)

        mdt_confidence = sum(r.confidence for r in mdt_responses) / len(mdt_responses)
        score = mdt_confidence
        reason = f"MDT 路径, avg_confidence={mdt_confidence:.3f} ({len(mdt_responses)} 次会诊)"
        return DimensionScore(name="interface_acceptance", weight=0.18, score=score, reason=reason)

    # ---- Regression 对比 ----

    @staticmethod
    def _compare_regression(current: dict, baseline: dict) -> dict:
        """与基线对比检测 regression"""
        current_total = current.get("total_score", 0)
        baseline_total = baseline.get("total_score", 0)
        diff = current_total - baseline_total

        current_dims = {d["name"]: d["score"] for d in current.get("dimensions", [])}
        baseline_dims = {d["name"]: d["score"] for d in baseline.get("dimensions", [])}

        regressions = {}
        for name, score in current_dims.items():
            base = baseline_dims.get(name, score)
            d = score - base
            if d < -0.05:
                regressions[name] = {"baseline": round(base, 4), "current": round(score, 4), "diff": round(d, 4)}
        improvements = {}
        for name, score in current_dims.items():
            base = baseline_dims.get(name, score)
            d = score - base
            if d > 0.05:
                improvements[name] = {"baseline": round(base, 4), "current": round(score, 4), "diff": round(d, 4)}

        return {
            "total_diff": round(diff, 4),
            "regressed_dimensions": regressions,
            "improved_dimensions": improvements,
        }

    def to_json(self, report: EvaluationReport, path: str = "") -> str:
        data = report.to_dict()
        text = json.dumps(data, ensure_ascii=False, indent=2)
        if path:
            Path(path).write_text(text, encoding="utf-8")
        return text
