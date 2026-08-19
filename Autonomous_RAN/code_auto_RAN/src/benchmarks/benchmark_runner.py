"""Benchmark runner comparing multi-agent system vs baselines."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.common.utils import load_config, project_root, save_json
from src.simulation.baselines import BASELINES
from src.simulation.ran_simulator import RANSimulator


class BenchmarkRunner:
    def __init__(self, seed: int = 42):
        self.sim = RANSimulator(seed=seed)
        self.kpi_cfg = load_config("kpis.json")
        self.output_dir = project_root() / "outputs" / "reports"
        self.plots_dir = project_root() / "outputs" / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def run_all(self, steps: int = 80) -> dict:
        results = []
        for name in BASELINES:
            r = self.sim.run_baseline(name, steps=steps)
            results.append(self._to_dict(r))
        ma = self.sim.run_multi_agent(steps=steps)
        results.append(self._to_dict(ma))

        df = pd.DataFrame(results)
        report_path = self.output_dir / "benchmark_report.json"
        save_json({
            "benchmarks": results,
            "target_kpis": self.kpi_cfg["target_kpis"],
            "improvement_vs_pf": self._compute_improvement(results, "proportional_fair", "multi_agent_autonomous"),
        }, report_path)
        df.to_csv(self.output_dir / "benchmark_results.csv", index=False)
        self._plot(results)
        return {"report": str(report_path), "results": results}

    def _to_dict(self, r) -> dict:
        return {
            "scheduler": r.scheduler,
            "avg_throughput_mbps": round(r.avg_throughput_mbps, 2),
            "avg_latency_ms": round(r.avg_latency_ms, 2),
            "fairness_index": round(r.fairness_index, 3),
            "energy_w": round(r.energy_w, 2),
            "handover_success_rate": round(r.handover_success_rate, 3),
            "security_detection_rate": round(r.security_detection_rate, 3),
            "qoe_score": round(r.qoe_score, 2),
            "steps": r.steps,
        }

    def _compute_improvement(self, results: list[dict], baseline: str, target: str) -> dict:
        base = next(r for r in results if r["scheduler"] == baseline)
        tgt = next(r for r in results if r["scheduler"] == target)
        return {
            "throughput_pct": round((tgt["avg_throughput_mbps"] / base["avg_throughput_mbps"] - 1) * 100, 1),
            "latency_pct": round((1 - tgt["avg_latency_ms"] / base["avg_latency_ms"]) * 100, 1),
            "energy_pct": round((1 - tgt["energy_w"] / base["energy_w"]) * 100, 1),
            "qoe_pct": round((tgt["qoe_score"] / base["qoe_score"] - 1) * 100, 1),
        }

    def _plot(self, results: list[dict]) -> None:
        schedulers = [r["scheduler"] for r in results]
        metrics = ["avg_throughput_mbps", "avg_latency_ms", "fairness_index", "qoe_score"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("Autonomous RAN Benchmark Comparison", fontsize=14)
        for ax, metric in zip(axes.flat, metrics):
            vals = [r[metric] for r in results]
            ax.bar(range(len(schedulers)), vals, color="steelblue")
            ax.set_xticks(range(len(schedulers)))
            ax.set_xticklabels(schedulers, rotation=45, ha="right", fontsize=7)
            ax.set_title(metric.replace("_", " ").title())
        plt.tight_layout()
        plot_path = self.plots_dir / "benchmark_comparison.png"
        plt.savefig(plot_path, dpi=120)
        plt.close()
