"""Per-agent visualization: histograms, CDFs, classification, output stats."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

from src.agents.base_agent import AgentObservation
from src.common.utils import load_json, project_root, save_json

PALETTE = sns.color_palette("husl", 12)


class AgentVisualizationSuite:
    AGENT_DATA_MAP = {
        "scheduler": ("ran_kpi_dataset.csv", ["cqi", "sinr_db", "throughput_mbps", "latency_ms"]),
        "resource": ("energy_metrics.csv", ["cell_utilization", "power_consumption_w", "traffic_demand_mbps"]),
        "mobility": ("mobility_traces.csv", ["velocity_mps", "rsrp_dbm", "handover_pending"]),
        "security": ("security_events.csv", ["packet_rate_pps", "spectrum_anomaly_score", "is_attack"]),
        "energy": ("energy_metrics.csv", ["power_consumption_w", "cell_utilization", "sleep_state"]),
        "carbon": ("energy_metrics.csv", ["carbon_intensity_gco2_kwh", "renewable_pct", "power_consumption_w"]),
        "ran_sleep": ("energy_metrics.csv", ["sleep_state", "cell_utilization", "power_consumption_w"]),
        "renewable_energy": ("energy_metrics.csv", ["renewable_pct", "power_consumption_w"]),
        "edge_inference": ("ran_kpi_dataset.csv", ["latency_ms", "throughput_mbps"]),
        "green_slice": ("slice_utilization.csv", ["prb_utilization", "sla_compliance"]),
        "traffic": ("ran_kpi_dataset.csv", ["buffer_occupancy", "throughput_mbps", "latency_ms", "packet_loss"]),
        "qos": ("ran_kpi_dataset.csv", ["latency_ms", "throughput_mbps", "slice"]),
        "slice": ("slice_utilization.csv", ["prb_utilization", "sla_compliance", "latency_p99_ms"]),
        "qoe": ("ran_kpi_dataset.csv", ["throughput_mbps", "latency_ms", "packet_loss"]),
        "channel_estimation": ("ran_kpi_dataset.csv", ["sinr_db", "cqi", "rsrp_dbm"]),
        "beamforming": ("ran_kpi_dataset.csv", ["sinr_db", "cqi", "rsrp_dbm"]),
        "csi": ("ran_kpi_dataset.csv", ["cqi", "sinr_db", "mcs", "throughput_mbps"]),
        "air_interface": ("ran_kpi_dataset.csv", ["sinr_db", "cqi", "mcs"]),
        "digital_twin": ("energy_metrics.csv", ["cell_utilization", "power_consumption_w"]),
        "spectrum": ("security_events.csv", ["spectrum_anomaly_score", "packet_rate_pps"]),
        "self_healing": ("security_events.csv", ["auth_failures", "spectrum_anomaly_score"]),
        "agent_optimizer": ("ran_kpi_dataset.csv", ["throughput_mbps", "latency_ms"]),
        "coordination": ("ran_kpi_dataset.csv", ["throughput_mbps", "latency_ms", "buffer_occupancy"]),
    }

    def __init__(self):
        self.data_dir = project_root() / "data" / "datasets"
        self.plots_dir = project_root() / "outputs" / "plots" / "agents"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.generated: list[str] = []

    def generate_all(self, validation_report: dict | None = None) -> dict:
        for agent in self.AGENT_DATA_MAP:
            self._plot_agent_histogram_cdf(agent)
            self._plot_agent_output_simulation(agent)
        if validation_report:
            self._plot_validation_summary(validation_report)
        self._plot_super_agent_overview()
        manifest = {"agent_plots": self.generated, "count": len(self.generated)}
        save_json(manifest, self.plots_dir / "agent_plots_manifest.json")
        return manifest

    def _save(self, fig, name: str) -> None:
        path = self.plots_dir / name
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        self.generated.append(f"agents/{name}")

    def _load_agent_data(self, agent: str) -> pd.DataFrame | None:
        fname, _ = self.AGENT_DATA_MAP[agent]
        path = self.data_dir / fname
        return pd.read_csv(path) if path.exists() else None

    def _plot_agent_histogram_cdf(self, agent: str) -> None:
        df = self._load_agent_data(agent)
        if df is None:
            return
        _, cols = self.AGENT_DATA_MAP[agent]
        cols = [c for c in cols if c in df.columns][:3]
        if not cols:
            return

        fig, axes = plt.subplots(2, len(cols), figsize=(4 * len(cols), 7))
        if len(cols) == 1:
            axes = axes.reshape(2, 1)
        fig.suptitle(f"{agent.replace('_',' ').title()} Agent — Data Distributions", fontweight="bold")

        for i, col in enumerate(cols):
            data = df[col].dropna()
            axes[0, i].hist(data, bins=35, color=PALETTE[i], edgecolor="white", alpha=0.85)
            axes[0, i].set_title(f"Histogram: {col}")
            sorted_d = np.sort(data)
            cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
            axes[1, i].plot(sorted_d, cdf, color=PALETTE[i], linewidth=2)
            axes[1, i].set_title(f"CDF: {col}")
            axes[1, i].set_ylabel("CDF")
        self._save(fig, f"{agent}_hist_cdf.png")

    def _plot_agent_output_simulation(self, agent: str) -> None:
        from src.orchestration.super_agent_controller import SuperAgentController
        ctrl = SuperAgentController()
        if agent not in ctrl.agents:
            return

        outputs = []
        df = self._load_agent_data(agent)
        if df is None:
            return
        sample = df.sample(min(50, len(df)), random_state=42)
        for _, row in sample.iterrows():
            features = {k: float(v) for k, v in row.items() if isinstance(v, (int, float, np.integer, np.floating))}
            obs = AgentObservation(timestamp=0, features=features, context={"slice": row.get("slice", "eMBB")})
            act = ctrl.agents[agent].predict(obs)
            outputs.append({**act.parameters, "confidence": act.confidence})

        if not outputs:
            return
        out_df = pd.DataFrame(outputs)
        numeric_cols = out_df.select_dtypes(include=[np.number]).columns[:4]
        if len(numeric_cols) == 0:
            return

        fig, axes = plt.subplots(1, len(numeric_cols), figsize=(4 * len(numeric_cols), 4))
        if len(numeric_cols) == 1:
            axes = [axes]
        fig.suptitle(f"{agent.replace('_',' ').title()} Agent — Model Output Distribution", fontweight="bold")
        for ax, col in zip(axes, numeric_cols):
            ax.hist(out_df[col].dropna(), bins=20, color=PALETTE[5], edgecolor="white")
            ax.set_title(col)
        self._save(fig, f"{agent}_output_dist.png")

    def _plot_validation_summary(self, report: dict) -> None:
        agents = list(report.get("validation", {}).keys())
        metrics = []
        for a, m in report.get("validation", {}).items():
            score = m.get("accuracy") or m.get("r2_score") or m.get("detection_accuracy") or 0
            metrics.append(score)
        fig, ax = plt.subplots(figsize=(12, 5))
        bars = ax.bar(agents, metrics, color=PALETTE[:len(agents)])
        ax.set_ylim(0, 1.1)
        ax.set_title("Agent Validation Scores (Train/Test)", fontweight="bold")
        ax.set_ylabel("Score")
        plt.xticks(rotation=35, ha="right")
        for bar, s in zip(bars, metrics):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{s:.3f}", ha="center", fontsize=8)
        self._save(fig, "validation_all_agents.png")

    def _plot_super_agent_overview(self) -> None:
        from src.agents.super_agent import SuperAgent
        sa = SuperAgent()
        fig, ax = plt.subplots(figsize=(10, 5))
        agents = list(sa.AGENT_WEIGHTS.keys())
        weights = list(sa.AGENT_WEIGHTS.values())
        ax.barh(agents, weights, color=PALETTE[:len(agents)])
        ax.set_xlabel("Control Weight")
        ax.set_title("Super Agent — Agent Control Weights", fontweight="bold")
        self._save(fig, "super_agent_weights.png")


def generate_agent_plots(validation_report: dict | None = None) -> dict:
    return AgentVisualizationSuite().generate_all(validation_report)
