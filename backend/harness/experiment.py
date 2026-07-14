"""实验/AB测试框架 - 系统化配置对比

设计理念（来自 Agent Harness 综述）:
  "好 Harness 不只是会加控制，还要知道什么时候删控制"
  "Agent 工程不是越复杂越好 —— 需要实验来验证每个控制是否必要"

功能:
  - 配置快照: 记录每次实验的完整配置
  - A/B 对比: 运行两组不同配置并比较评估结果
  - 回归检测: 自动标记退化维度
  - 实验追踪: 保存实验历史
"""
from __future__ import annotations

import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "experiments"


# ============================================================
# 实验数据模型
# ============================================================

@dataclass
class ExperimentRun:
    run_id: str
    experiment_name: str
    timestamp: str
    config_snapshot: dict
    results: Optional[dict] = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Experiment:
    name: str
    control_config: dict
    treatment_config: dict
    control_results: Optional[dict] = None
    treatment_results: Optional[dict] = None
    comparison: Optional[dict] = None
    created_at: str = ""
    completed_at: str = ""


# ============================================================
# 实验追踪器
# ============================================================

class ExperimentTracker:
    """实验追踪器 - 管理 A/B 实验生命周期"""

    def __init__(self, experiments_dir: str = ""):
        self._dir = Path(experiments_dir) if experiments_dir else EXPERIMENTS_DIR
        self._dir.mkdir(parents=True, exist_ok=True)

    # ---- 配置快照 ----

    @staticmethod
    def snapshot_config(config: dict) -> tuple[str, dict]:
        """对配置做快照并计算 hash"""
        # 移除敏感字段
        safe = config.copy()
        if "api_key" in safe:
            safe["api_key"] = "***"
        if "llm" in safe and isinstance(safe["llm"], dict):
            safe["llm"] = {k: v for k, v in safe["llm"].items() if k != "api_key"}
            safe["llm"]["api_key"] = "***"

        raw = json.dumps(safe, sort_keys=True, ensure_ascii=False)
        config_hash = hashlib.sha256(raw.encode()).hexdigest()[:12]
        return config_hash, safe

    # ---- 实验运行 ----

    async def run_experiment(
        self,
        name: str,
        control_orchestrator,
        treatment_orchestrator,
        queries: list[str],
        user_id: str = "experiment_user",
    ) -> Experiment:
        """运行 A/B 实验: 用两组编排器处理相同问题

        参数:
            name: 实验名称
            control_orchestrator: 对照组编排器
            treatment_orchestrator: 实验组编排器
            queries: 测试问题列表
            user_id: 用户 ID

        返回:
            Experiment 包含两组结果和对比
        """
        from schema.models import MedicalQuery
        from harness.evaluator import HarnessEvaluator

        created_at = datetime.now().isoformat()

        # 对照组
        control_eval = HarnessEvaluator(config_hash="control")
        for q in queries:
            resp = await control_orchestrator.process(MedicalQuery(query=q, user_id=user_id))
            control_eval.add_result(resp, getattr(resp, "_latency_ms", 1000))
        control_report = control_eval.evaluate()

        # 实验组
        treatment_eval = HarnessEvaluator(config_hash="treatment")
        for q in queries:
            resp = await treatment_orchestrator.process(MedicalQuery(query=q, user_id=user_id))
            treatment_eval.add_result(resp, getattr(resp, "_latency_ms", 1000))
        treatment_report = treatment_eval.evaluate()

        comparison = self._compare_experiments(
            control_report.to_dict(), treatment_report.to_dict()
        )

        experiment = Experiment(
            name=name,
            control_config={},
            treatment_config={},
            control_results=control_report.to_dict(),
            treatment_results=treatment_report.to_dict(),
            comparison=comparison,
            created_at=created_at,
            completed_at=datetime.now().isoformat(),
        )

        self._save(experiment)
        return experiment

    @staticmethod
    def _compare_experiments(control: dict, treatment: dict) -> dict:
        """比较两组实验结果"""
        control_total = control.get("total_score", 0)
        treatment_total = treatment.get("total_score", 0)

        control_dims = {d["name"]: d for d in control.get("dimensions", [])}
        treatment_dims = {d["name"]: d for d in treatment.get("dimensions", [])}

        dim_diffs = {}
        for name, c_dim in control_dims.items():
            t_dim = treatment_dims.get(name, {})
            diff = t_dim.get("score", 0) - c_dim.get("score", 0)
            dim_diffs[name] = {
                "control": c_dim.get("score", 0),
                "treatment": t_dim.get("score", 0),
                "diff": round(diff, 4),
                "regression": diff < -0.05,
                "improvement": diff > 0.05,
            }

        return {
            "total_control": round(control_total, 4),
            "total_treatment": round(treatment_total, 4),
            "total_diff": round(treatment_total - control_total, 4),
            "dimensions": dim_diffs,
            "control_fallback_rate": control.get("num_safe_fallback", 0) / max(control.get("num_queries", 1), 1),
            "treatment_fallback_rate": treatment.get("num_safe_fallback", 0) / max(treatment.get("num_queries", 1), 1),
        }

    # ---- 持久化 ----

    def _save(self, experiment: Experiment):
        path = self._dir / f"{experiment.name}_{experiment.created_at[:10]}.json"
        data = {
            "name": experiment.name,
            "created_at": experiment.created_at,
            "completed_at": experiment.completed_at,
            "comparison": experiment.comparison,
            "control_results": experiment.control_results,
            "treatment_results": experiment.treatment_results,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"实验结果已保存: {path}")

    def list_experiments(self) -> list[dict]:
        """列出所有历史实验"""
        if not self._dir.exists():
            return []
        experiments = []
        for f in sorted(self._dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                experiments.append({
                    "name": data.get("name", f.stem),
                    "date": data.get("created_at", ""),
                    "total_diff": data.get("comparison", {}).get("total_diff", 0),
                })
            except Exception:
                pass
        return experiments

    def load_experiment(self, name: str) -> Optional[dict]:
        """按名称加载最近的一次实验"""
        candidates = sorted(self._dir.glob(f"{name}_*.json"), reverse=True)
        if not candidates:
            return None
        try:
            return json.loads(candidates[0].read_text(encoding="utf-8"))
        except Exception:
            return None
