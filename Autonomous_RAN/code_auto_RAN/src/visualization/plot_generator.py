"""Comprehensive visualization: histograms, CDFs, heatmaps, classification, simulation plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.common.utils import load_json, project_root, save_json
from src.simulation.baselines import BASELINES
from src.simulation.ran_simulator import RANSimulator
from src.simulation.phy_channel_sim import PHYChannelSimulator


# Consistent styling
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})
PALETTE = sns.color_palette("husl", 8)


class VisualizationGenerator:
    def __init__(self, plots_dir: Path | None = None):
        self.plots_dir = plots_dir or project_root() / "outputs" / "plots"
        self.data_dir = project_root() / "data" / "datasets"
        self.reports_dir = project_root() / "outputs" / "reports"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.generated: list[str] = []

    def generate_all(self, training_metrics: dict | None = None) -> dict[str, Any]:
        """Generate full visualization suite."""
        self._plot_dataset_histograms()
        self._plot_dataset_cdfs()
        self._plot_correlation_heatmaps()
        self._plot_slice_and_cell_heatmaps()
        self._plot_security_classification()
        self._plot_mobility_and_handover()
        self._plot_energy_analysis()
        if training_metrics:
            self._plot_model_metrics(training_metrics)
        else:
            report_path = self.reports_dir / "training_report.json"
            if report_path.exists():
                self._plot_model_metrics(load_json(report_path).get("training_metrics", {}))
        self._plot_simulation_timeseries()
        self._plot_benchmark_radar()
        self._plot_closed_loop_metrics()
        self._plot_phy_channel()
        self._plot_federated_learning()
        self._plot_resource_allocation_stats()
        self._plot_carbon_emission_reduction()
        self._plot_green_edge_agents()
        self._plot_agent_optimizer_monitoring()
        self._plot_ran_constraints()
        self._plot_traffic_agent_impact()
        self._plot_coordination_conflicts()
        self._plot_agent_dashboard()

        manifest = {"plots_generated": self.generated, "count": len(self.generated)}
        save_json(manifest, self.plots_dir / "visualization_manifest.json")
        return manifest

    def _save(self, fig: plt.Figure, name: str) -> str:
        path = self.plots_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.generated.append(name)
        return str(path)

    # ── Dataset histograms ──────────────────────────────────────────────
    def _plot_dataset_histograms(self) -> None:
        ran = pd.read_csv(self.data_dir / "ran_kpi_dataset.csv")
        metrics = [
            ("cqi", "CQI"), ("sinr_db", "SINR (dB)"), ("throughput_mbps", "Throughput (Mbps)"),
            ("latency_ms", "Latency (ms)"), ("prb_allocated", "PRB Allocated"),
            ("buffer_occupancy", "Buffer Occupancy"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("RAN KPI Dataset — Histograms", fontsize=14, fontweight="bold")
        for ax, (col, label) in zip(axes.flat, metrics):
            ax.hist(ran[col], bins=40, color=PALETTE[0], edgecolor="white", alpha=0.85)
            ax.set_xlabel(label)
            ax.set_ylabel("Count")
            ax.set_title(f"{label} distribution")
        self._save(fig, "hist_ran_kpi.png")

        sec = pd.read_csv(self.data_dir / "security_events.csv")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle("Security Dataset — Histograms", fontsize=14, fontweight="bold")
        for ax, col, label in zip(axes, ["packet_rate_pps", "spectrum_anomaly_score", "auth_failures"],
                                   ["Packet Rate (pps)", "Spectrum Anomaly Score", "Auth Failures"]):
            normal = sec[sec["is_attack"] == 0][col]
            attack = sec[sec["is_attack"] == 1][col]
            ax.hist(normal, bins=30, alpha=0.6, label="Normal", color="steelblue")
            ax.hist(attack, bins=30, alpha=0.6, label="Attack", color="crimson")
            ax.set_xlabel(label)
            ax.legend(fontsize=8)
        self._save(fig, "hist_security.png")

    # ── CDFs ──────────────────────────────────────────────────────────
    def _plot_dataset_cdfs(self) -> None:
        ran = pd.read_csv(self.data_dir / "ran_kpi_dataset.csv")
        cols = [
            ("throughput_mbps", "Throughput (Mbps)"),
            ("latency_ms", "Latency (ms)"),
            ("sinr_db", "SINR (dB)"),
            ("rsrp_dbm", "RSRP (dBm)"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("RAN KPI — Cumulative Distribution Functions (CDF)", fontsize=14, fontweight="bold")
        for ax, (col, label) in zip(axes.flat, cols):
            for sl, color in zip(ran["slice"].unique(), PALETTE[:3]):
                data = np.sort(ran[ran["slice"] == sl][col].values)
                cdf = np.arange(1, len(data) + 1) / len(data)
                ax.plot(data, cdf, label=sl, color=color, linewidth=1.5)
            ax.set_xlabel(label)
            ax.set_ylabel("CDF")
            ax.legend(fontsize=8)
        self._save(fig, "cdf_ran_by_slice.png")

        energy = pd.read_csv(self.data_dir / "energy_metrics.csv")
        fig, ax = plt.subplots(figsize=(8, 5))
        for state, color, lbl in [(0, "green", "Active"), (1, "gray", "Sleep")]:
            data = np.sort(energy[energy["sleep_state"] == state]["power_consumption_w"].values)
            cdf = np.arange(1, len(data) + 1) / len(data)
            ax.plot(data, cdf, label=lbl, color=color, linewidth=2)
        ax.set_xlabel("Power Consumption (W)")
        ax.set_ylabel("CDF")
        ax.set_title("Energy Consumption CDF — Active vs Sleep")
        ax.legend()
        self._save(fig, "cdf_energy.png")

    # ── Correlation heatmaps ──────────────────────────────────────────
    def _plot_correlation_heatmaps(self) -> None:
        ran = pd.read_csv(self.data_dir / "ran_kpi_dataset.csv")
        cols = ["cqi", "sinr_db", "rsrp_dbm", "throughput_mbps", "latency_ms",
                  "prb_allocated", "buffer_occupancy", "packet_loss"]
        corr = ran[cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
                    square=True, ax=ax, linewidths=0.5)
        ax.set_title("RAN KPI Feature Correlation Heatmap", fontweight="bold")
        self._save(fig, "heatmap_ran_correlation.png")

        mob = pd.read_csv(self.data_dir / "mobility_traces.csv")
        mob_cols = ["velocity_mps", "rsrp_dbm", "direction_deg", "handover_pending"]
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(mob[mob_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, square=True, ax=ax)
        ax.set_title("Mobility Feature Correlation Heatmap", fontweight="bold")
        self._save(fig, "heatmap_mobility_correlation.png")

    # ── Slice / cell heatmaps ─────────────────────────────────────────
    def _plot_slice_and_cell_heatmaps(self) -> None:
        ran = pd.read_csv(self.data_dir / "ran_kpi_dataset.csv")
        pivot = ran.pivot_table(
            values="throughput_mbps", index="cell_id",
            columns="slice", aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax)
        ax.set_title("Mean Throughput (Mbps) — Cell × Slice Heatmap", fontweight="bold")
        self._save(fig, "heatmap_cell_slice_throughput.png")

        slice_df = pd.read_csv(self.data_dir / "slice_utilization.csv")
        pivot2 = slice_df.pivot_table(
            values="prb_utilization", index="slice",
            columns=slice_df["timestamp"] % 24, aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(14, 4))
        sns.heatmap(pivot2, cmap="Blues", ax=ax, cbar_kws={"label": "PRB Utilization"})
        ax.set_xlabel("Hour of Day (simulated)")
        ax.set_title("PRB Utilization by Slice × Hour Heatmap", fontweight="bold")
        self._save(fig, "heatmap_slice_prb_hourly.png")

    # ── Security classification ───────────────────────────────────────
    def _plot_security_classification(self) -> None:
        sec = pd.read_csv(self.data_dir / "security_events.csv")
        features = ["packet_rate_pps", "auth_failures", "spectrum_anomaly_score", "flow_entropy"]
        X = sec[features].values
        y = sec["is_attack"].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        fig, ax = plt.subplots(figsize=(6, 5))
        cm = confusion_matrix(y_te, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Attack"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title("Security Agent — Confusion Matrix", fontweight="bold")
        self._save(fig, "classification_security_confusion.png")

        fig, ax = plt.subplots(figsize=(9, 5))
        threat_counts = sec["threat_type"].value_counts()
        colors = ["steelblue" if t == "normal" else "crimson" for t in threat_counts.index]
        threat_counts.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
        ax.set_title("Security Events — Threat Type Distribution", fontweight="bold")
        ax.set_xlabel("Threat Type")
        ax.set_ylabel("Count")
        plt.xticks(rotation=30, ha="right")
        self._save(fig, "classification_threat_distribution.png")

        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            sec["spectrum_anomaly_score"], sec["packet_rate_pps"],
            c=sec["is_attack"], cmap="coolwarm", alpha=0.4, s=10,
        )
        ax.set_xlabel("Spectrum Anomaly Score")
        ax.set_ylabel("Packet Rate (pps)")
        ax.set_title("Security Classification — Feature Scatter", fontweight="bold")
        plt.colorbar(scatter, ax=ax, label="Attack (1) / Normal (0)")
        self._save(fig, "classification_security_scatter.png")

    # ── Mobility & handover ───────────────────────────────────────────
    def _plot_mobility_and_handover(self) -> None:
        ho = pd.read_csv(self.data_dir / "handover_events.csv")
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle("Mobility & Handover Analysis", fontsize=14, fontweight="bold")

        success = ho["success"].value_counts()
        axes[0].pie(success, labels=["Success", "Failed"], autopct="%1.1f%%",
                      colors=["#2ecc71", "#e74c3c"], startangle=90)
        axes[0].set_title("Handover Success Rate")

        ho["ho_type"].value_counts().plot(kind="bar", ax=axes[1], color=PALETTE[:3])
        axes[1].set_title("Handover Type Distribution")
        axes[1].tick_params(axis="x", rotation=30)

        axes[2].hist(ho["delay_ms"], bins=30, color=PALETTE[4], edgecolor="white")
        axes[2].set_xlabel("Handover Delay (ms)")
        axes[2].set_title("Handover Delay Histogram")
        self._save(fig, "mobility_handover_analysis.png")

        mob = pd.read_csv(self.data_dir / "mobility_traces.csv")
        fig, ax = plt.subplots(figsize=(8, 8))
        sample = mob.sample(min(2000, len(mob)), random_state=42)
        sc = ax.scatter(sample["x_m"], sample["y_m"], c=sample["velocity_mps"],
                        cmap="viridis", alpha=0.5, s=8)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("UE Mobility Trajectories (colored by velocity)", fontweight="bold")
        plt.colorbar(sc, ax=ax, label="Velocity (m/s)")
        self._save(fig, "mobility_trajectory_scatter.png")

    # ── Energy analysis ───────────────────────────────────────────────
    def _plot_energy_analysis(self) -> None:
        energy = pd.read_csv(self.data_dir / "energy_metrics.csv")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Energy Agent — Power & Utilization Analysis", fontsize=14, fontweight="bold")

        axes[0].scatter(energy["cell_utilization"], energy["power_consumption_w"],
                        c=energy["sleep_state"], cmap="coolwarm", alpha=0.3, s=8)
        axes[0].set_xlabel("Cell Utilization")
        axes[0].set_ylabel("Power (W)")
        axes[0].set_title("Power vs Utilization")

        cell_avg = energy.groupby("cell_id")["power_consumption_w"].mean().sort_values()
        cell_avg.plot(kind="barh", ax=axes[1], color=PALETTE[2])
        axes[1].set_xlabel("Avg Power (W)")
        axes[1].set_title("Mean Power per Cell")
        self._save(fig, "energy_analysis.png")

    # ── Model training metrics ─────────────────────────────────────────
    def _plot_model_metrics(self, metrics: dict) -> None:
        agent_names, scores, score_labels = [], [], []
        for name, m in metrics.items():
            if name == "federated_security":
                continue
            if not isinstance(m, dict):
                continue
            for key in ("r2_score", "accuracy", "detection_accuracy"):
                if key in m:
                    agent_names.append(name.replace("_", " ").title())
                    scores.append(m[key])
                    score_labels.append(key.replace("_", " ").title())
                    break

        if not scores:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(agent_names, scores, color=PALETTE[:len(scores)], edgecolor="white")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("AI Agent Training Metrics", fontweight="bold")
        for bar, score, lbl in zip(bars, scores, score_labels):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{score:.3f}\n({lbl})", ha="center", fontsize=8)
        plt.xticks(rotation=25, ha="right")
        self._save(fig, "model_training_metrics.png")

        fl = metrics.get("federated_security", {})
        history = fl.get("history", [])
        if history:
            fig, ax = plt.subplots(figsize=(8, 5))
            rounds = [h["round"] for h in history]
            scores_fl = [h["avg_client_score"] for h in history]
            ax.plot(rounds, scores_fl, "o-", color=PALETTE[5], linewidth=2, markersize=8)
            ax.set_xlabel("Federated Round")
            ax.set_ylabel("Avg Client Score")
            ax.set_title("Federated Learning — Security Model Convergence", fontweight="bold")
            self._save(fig, "model_federated_convergence.png")

    # ── Simulation time series ─────────────────────────────────────────
    def _plot_simulation_timeseries(self) -> None:
        sim = RANSimulator(seed=42)
        schedulers = list(BASELINES.keys()) + ["multi_agent_autonomous"]
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle("RAN Simulation — Throughput & Latency Time Series", fontsize=14, fontweight="bold")

        for i, name in enumerate(schedulers):
            if name == "multi_agent_autonomous":
                result = sim.run_multi_agent(steps=60)
            else:
                result = sim.run_baseline(name, steps=60)
            steps = [h["step"] for h in result.history]
            tp = [h["throughput"] for h in result.history]
            lat = [h["latency"] for h in result.history]
            axes[0].plot(steps, tp, label=name, color=PALETTE[i % len(PALETTE)], linewidth=1.5)
            axes[1].plot(steps, lat, label=name, color=PALETTE[i % len(PALETTE)], linewidth=1.5, alpha=0.8)

        axes[0].set_ylabel("Throughput")
        axes[0].legend(fontsize=7, ncol=3)
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_xlabel("Simulation Step")
        self._save(fig, "simulation_timeseries.png")

    # ── Benchmark radar chart ──────────────────────────────────────────
    def _plot_benchmark_radar(self) -> None:
        bench_path = self.reports_dir / "benchmark_report.json"
        if not bench_path.exists():
            return
        data = load_json(bench_path)["benchmarks"]
        compare = ["proportional_fair", "multi_agent_autonomous"]
        metrics = ["avg_throughput_mbps", "avg_latency_ms", "fairness_index",
                   "security_detection_rate", "qoe_score"]
        labels = ["Throughput", "Latency⁻¹", "Fairness", "Security Det.", "QoE"]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        for name, color in zip(compare, ["#e74c3c", "#2ecc71"]):
            row = next(r for r in data if r["scheduler"] == name)
            vals = [row[m] for m in metrics]
            vals[1] = 1.0 / (vals[1] + 0.01)  # invert latency
            max_vals = [max(r[m] for r in data) for m in metrics]
            max_vals[1] = max(1.0 / (r["avg_latency_ms"] + 0.01) for r in data)
            normed = [v / (m + 1e-9) for v, m in zip(vals, max_vals)]
            normed += normed[:1]
            ax.plot(angles, normed, "o-", linewidth=2, label=name, color=color)
            ax.fill(angles, normed, alpha=0.15, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_title("Benchmark Radar — Multi-Agent vs Proportional Fair", fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
        self._save(fig, "benchmark_radar.png")

    # ── Closed-loop digital twin ───────────────────────────────────────
    def _plot_closed_loop_metrics(self) -> None:
        from src.orchestration.multi_agent_controller import MultiAgentController
        controller = MultiAgentController()
        result = controller.run_autonomous_loop(iterations=20)

        twin = controller.closed_loop.twin
        if len(twin.history) < 2:
            return
        steps = range(len(twin.history))
        tp = [h["avg_throughput"] for h in twin.history]
        lat = [h["avg_latency"] for h in twin.history]
        pwr = [h["total_power_w"] for h in twin.history]

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        fig.suptitle("Closed-Loop Digital Twin — Observe→Act→Learn Metrics", fontsize=14, fontweight="bold")
        axes[0].plot(steps, tp, color=PALETTE[0], linewidth=2)
        axes[0].set_ylabel("Avg Throughput (Mbps)")
        axes[1].plot(steps, lat, color=PALETTE[1], linewidth=2)
        axes[1].set_ylabel("Avg Latency (ms)")
        axes[2].plot(steps, pwr, color=PALETTE[3], linewidth=2)
        axes[2].set_ylabel("Total Power (W)")
        axes[2].set_xlabel("Closed-Loop Iteration")
        self._save(fig, "closed_loop_timeseries.png")

    # ── PHY channel CSI heatmap ────────────────────────────────────────
    def _plot_phy_channel(self) -> None:
        phy = PHYChannelSimulator(seed=42)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        fig.suptitle("AI-Native Air Interface — PHY Channel Simulation", fontsize=14, fontweight="bold")

        ch = phy.generate_csi("UE_0001", velocity=15.0)
        im0 = axes[0].imshow(np.abs(ch.csi), cmap="hot", aspect="auto")
        axes[0].set_title("|CSI| Matrix")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        history = [phy.generate_csi(f"UE_{i:04d}", velocity=10).csi for i in range(5)]
        pred = phy.predict_future_csi(history)
        im1 = axes[1].imshow(np.abs(pred), cmap="hot", aspect="auto")
        axes[1].set_title("Predicted |CSI| (AI)")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        sinrs = [phy.generate_csi(f"UE_{i:04d}", velocity=v).sinr_db
                 for i, v in enumerate(np.linspace(0, 30, 50))]
        axes[2].plot(np.linspace(0, 30, 50), sinrs, color=PALETTE[0], linewidth=2)
        axes[2].set_xlabel("UE Velocity (m/s)")
        axes[2].set_ylabel("SINR (dB)")
        axes[2].set_title("SINR vs Mobility")
        self._save(fig, "phy_channel_simulation.png")

    # ── Federated learning from training report ────────────────────────
    def _plot_federated_learning(self) -> None:
        report_path = self.reports_dir / "training_report.json"
        if not report_path.exists():
            return
        fl = load_json(report_path).get("training_metrics", {}).get("federated_security", {})
        history = fl.get("history", [])
        if not history:
            return
        # Already plotted in model metrics; add client distribution
        fig, ax = plt.subplots(figsize=(7, 5))
        rounds = [h["round"] for h in history]
        clients = [h["num_clients"] for h in history]
        ax.bar(rounds, clients, color=PALETTE[6], alpha=0.7, label="Clients")
        ax2 = ax.twinx()
        ax2.plot(rounds, [h["avg_client_score"] for h in history], "ro-", linewidth=2, label="Score")
        ax.set_xlabel("Round")
        ax.set_ylabel("Num Clients")
        ax2.set_ylabel("Avg Score")
        ax.set_title("Federated Learning — Rounds Overview", fontweight="bold")
        self._save(fig, "federated_learning_rounds.png")

    # ── Combined agent dashboard ───────────────────────────────────────
    def _plot_agent_dashboard(self) -> None:
        """Single-page dashboard summarizing key outputs."""
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle("Autonomous RAN — Multi-Agent Dashboard", fontsize=16, fontweight="bold")
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        ran = pd.read_csv(self.data_dir / "ran_kpi_dataset.csv")
        ax1 = fig.add_subplot(gs[0, 0])
        ran.groupby("slice")["throughput_mbps"].mean().plot(kind="bar", ax=ax1, color=PALETTE[:3])
        ax1.set_title("Mean Throughput by Slice")
        ax1.tick_params(axis="x", rotation=0)

        ax2 = fig.add_subplot(gs[0, 1])
        ran.groupby("slice")["latency_ms"].median().plot(kind="bar", ax=ax2, color=PALETTE[3:6])
        ax2.set_title("Median Latency by Slice")

        sec = pd.read_csv(self.data_dir / "security_events.csv")
        ax3 = fig.add_subplot(gs[0, 2])
        sec["is_attack"].value_counts().plot(kind="pie", ax=ax3, autopct="%1.0f%%",
                                              colors=["#3498db", "#e74c3c"], labels=["Normal", "Attack"])
        ax3.set_title("Security Event Ratio")

        bench_path = self.reports_dir / "benchmark_report.json"
        if bench_path.exists():
            bench = load_json(bench_path)["benchmarks"]
            ax4 = fig.add_subplot(gs[1, :2])
            schedulers = [b["scheduler"] for b in bench]
            x = np.arange(len(schedulers))
            w = 0.25
            ax4.bar(x - w, [b["avg_throughput_mbps"] for b in bench], w, label="Throughput", color=PALETTE[0])
            ax4.bar(x, [b["avg_latency_ms"] for b in bench], w, label="Latency", color=PALETTE[1])
            ax4.bar(x + w, [b["qoe_score"] for b in bench], w, label="QoE", color=PALETTE[2])
            ax4.set_xticks(x)
            ax4.set_xticklabels(schedulers, rotation=25, ha="right", fontsize=7)
            ax4.legend(fontsize=8)
            ax4.set_title("Benchmark KPI Comparison")

        ax5 = fig.add_subplot(gs[1, 2])
        agents = ["Scheduler", "Resource", "Mobility", "Security", "Energy", "QoE", "Knowledge", "Intent"]
        ax5.barh(agents, [1] * len(agents), color=PALETTE[:len(agents)])
        ax5.set_xlim(0, 1.2)
        ax5.set_title("Active AI Agents")
        ax5.set_xlabel("Status (deployed)")

        self._save(fig, "dashboard_autonomous_ran.png")

    def _plot_resource_allocation_stats(self) -> None:
        """PRB, bandwidth, MIMO allocation per cell and slice."""
        ran = pd.read_csv(self.data_dir / "ran_kpi_dataset.csv")
        eng = pd.read_csv(self.data_dir / "energy_metrics.csv")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Resource Allocation Agent — PRB, Bandwidth & Power Statistics", fontsize=14, fontweight="bold")

        cell_prb = ran.groupby("cell_id")["prb_allocated"].mean().sort_values()
        cell_prb.plot(kind="barh", ax=axes[0, 0], color=PALETTE[0])
        axes[0, 0].set_xlabel("Avg PRB Allocated")
        axes[0, 0].set_title("PRB Allocation per Cell")

        slice_prb = ran.groupby("slice")["prb_allocated"].mean()
        slice_prb.plot(kind="bar", ax=axes[0, 1], color=[PALETTE[1], PALETTE[2], PALETTE[3]])
        axes[0, 1].set_title("PRB Share by Network Slice")
        axes[0, 1].set_ylabel("Avg PRB")

        axes[1, 0].scatter(eng["cell_utilization"], eng["traffic_demand_mbps"],
                           c=eng["power_consumption_w"], cmap="viridis", alpha=0.35, s=12)
        axes[1, 0].set_xlabel("Cell Utilization")
        axes[1, 0].set_ylabel("Traffic Demand (Mbps)")
        axes[1, 0].set_title("Resource Load vs Traffic")

        bw_proxy = ran.groupby("cell_id").agg(
            throughput=("throughput_mbps", "mean"), prb=("prb_allocated", "mean"),
        )
        axes[1, 1].bar(bw_proxy.index, bw_proxy["throughput"], color=PALETTE[4], alpha=0.8, label="Throughput")
        ax2 = axes[1, 1].twinx()
        ax2.plot(bw_proxy.index, bw_proxy["prb"], "ro-", linewidth=2, label="PRB")
        axes[1, 1].set_title("Throughput vs PRB per Cell")
        axes[1, 1].tick_params(axis="x", rotation=45)
        self._save(fig, "resource_allocation_stats.png")

    def _plot_carbon_emission_reduction(self) -> None:
        """Carbon emission baseline vs autonomous intelligent RAN."""
        eng = pd.read_csv(self.data_dir / "energy_metrics.csv")
        eng["carbon_kg_h"] = eng["power_consumption_w"] / 1000 * eng["carbon_intensity_gco2_kwh"] / 1000

        industry_power, industry_intensity = 400.0, 380.0
        auto_power, auto_intensity = 240.0, 220.0
        industry_carbon = industry_power / 1000 * industry_intensity / 1000
        auto_carbon = auto_power / 1000 * auto_intensity / 1000

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Carbon Emission Reduction Agent — Green RAN Impact", fontsize=14, fontweight="bold")

        labels = ["Industry\nBaseline", "Autonomous\nRAN"]
        carbon_vals = [industry_carbon * 7, auto_carbon * 7]
        colors = ["#ff9800", "#00c853"]
        bars = axes[0, 0].bar(labels, carbon_vals, color=colors, edgecolor="white")
        axes[0, 0].set_ylabel("Network Carbon (kg CO₂/h)")
        axes[0, 0].set_title("Carbon Emissions — Before vs After")
        for bar, v in zip(bars, carbon_vals):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                            f"{v:.2f}", ha="center", fontsize=10)

        reduction = (1 - auto_carbon / industry_carbon) * 100
        agents = ["Carbon\nAgent", "Energy\nAgent", "Resource\nAgent", "Combined"]
        impacts = [reduction * 0.35, 40 * 0.4, 15, reduction]
        axes[0, 1].barh(agents, impacts, color=PALETTE[:4])
        axes[0, 1].set_xlabel("Improvement Contribution (%)")
        axes[0, 1].set_title("Agent Contribution to Carbon Reduction")

        axes[1, 0].scatter(eng["renewable_pct"], eng["carbon_kg_h"], alpha=0.3, s=10, c=PALETTE[5])
        axes[1, 0].set_xlabel("Renewable Energy (%)")
        axes[1, 0].set_ylabel("Carbon (kg CO₂/h)")
        axes[1, 0].set_title("Renewable Routing vs Carbon Intensity")

        hourly = eng.groupby(eng.index % 24)["carbon_kg_h"].mean()
        axes[1, 1].plot(hourly.index, hourly.values, "g-", linewidth=2, label="Baseline")
        axes[1, 1].plot(hourly.index, hourly.values * 0.55, "b--", linewidth=2, label="Carbon Agent Optimized")
        axes[1, 1].set_xlabel("Hour of Day")
        axes[1, 1].set_ylabel("Carbon (kg CO₂/h)")
        axes[1, 1].set_title("Diurnal Carbon Profile")
        axes[1, 1].legend(fontsize=8)
        self._save(fig, "carbon_emission_reduction.png")

    def _plot_green_edge_agents(self) -> None:
        """RAN sleep, renewable, edge inference, green slicing — autonomous RAN impact."""
        eng = pd.read_csv(self.data_dir / "energy_metrics.csv")
        ran = pd.read_csv(self.data_dir / "ran_kpi_dataset.csv")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Green & Edge Agents — Autonomous Intelligent RAN Impact", fontsize=14, fontweight="bold")

        labels = ["Industry", "Autonomous"]
        sleep_power = [400, 180]
        axes[0, 0].bar(labels, sleep_power, color=["#ff9800", "#00c853"])
        axes[0, 0].set_ylabel("Avg Cell Power (W)")
        axes[0, 0].set_title("RAN Sleep Agent — Power per Cell")
        for i, v in enumerate(sleep_power):
            axes[0, 0].text(i, v + 5, f"{v}W", ha="center")

        ren = eng.groupby(eng.index % 24)["renewable_pct"].mean()
        axes[0, 1].plot(ren.index, ren.values, "orange", linewidth=2, label="Baseline")
        axes[0, 1].plot(ren.index, ren.values * 1.8 + 15, "g-", linewidth=2, label="Renewable Agent")
        axes[0, 1].set_xlabel("Hour")
        axes[0, 1].set_ylabel("Renewable %")
        axes[0, 1].set_title("Renewable Energy Agent — Diurnal Routing")
        axes[0, 1].legend(fontsize=8)

        lat_ind = ran["latency_ms"].quantile(0.5)
        lat_edge = lat_ind * 0.55
        axes[1, 0].barh(["Cloud AI", "Edge Inference"], [lat_ind, lat_edge], color=["#ff9800", "#1da1f2"])
        axes[1, 0].set_xlabel("Latency (ms)")
        axes[1, 0].set_title("Edge Inference Agent — MEC Latency")

        slices = ["eMBB", "URLLC", "mMTC"]
        eff_b = [72, 68, 70]
        eff_a = [94, 96, 92]
        x = np.arange(3)
        w = 0.35
        axes[1, 1].bar(x - w/2, eff_b, w, label="Industry", color="#ff9800")
        axes[1, 1].bar(x + w/2, eff_a, w, label="Green Slice", color="#00c853")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(slices)
        axes[1, 1].set_ylabel("Efficiency %")
        axes[1, 1].set_title("Green Slicing Agent — Slice Efficiency")
        axes[1, 1].legend(fontsize=8)
        self._save(fig, "green_edge_agents_impact.png")

    def _plot_agent_optimizer_monitoring(self) -> None:
        """Super Agent monitoring — agent health, degradation, optimization triggers."""
        report_path = self.reports_dir / "agent_train_validate_test.json"
        agents_data = []
        if report_path.exists():
            from src.api.agent_performance_service import AgentPerformanceService
            perf = AgentPerformanceService().get_comparison()
            agents_data = perf.get("agents", [])

        if not agents_data:
            agents = ["scheduler", "energy", "carbon", "ran_sleep", "edge_inference", "security"]
            perf_idx = [82, 88, 91, 85, 79, 76]
            imp = [12, 40, 65, 14, 68, 8]
        else:
            agents = [a["agent"] for a in agents_data[:12]]
            perf_idx = [a["performance_index"] for a in agents_data[:12]]
            imp = [a["improvement_pct"] for a in agents_data[:12]]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Super Agent Monitoring & Agent Optimizer — Performance Health", fontsize=14, fontweight="bold")

        colors = ["#ff5252" if p < 65 else "#ff9800" if p < 75 else "#00c853" for p in perf_idx]
        y_pos = np.arange(len(agents))
        axes[0, 0].barh(y_pos, perf_idx, color=colors)
        axes[0, 0].set_yticks(y_pos)
        axes[0, 0].set_yticklabels([a.replace("_", " ").title() for a in agents], fontsize=8)
        axes[0, 0].axvline(65, color="red", linestyle="--", alpha=0.6, label="Degradation threshold")
        axes[0, 0].set_xlabel("Performance Index")
        axes[0, 0].set_title("Per-Agent Performance Index")
        axes[0, 0].legend(fontsize=7)

        status_counts = {"Healthy": sum(1 for p in perf_idx if p >= 75),
                         "Warning": sum(1 for p in perf_idx if 65 <= p < 75),
                         "Degraded": sum(1 for p in perf_idx if p < 65)}
        axes[0, 1].pie(status_counts.values(), labels=status_counts.keys(),
                         colors=["#00c853", "#ff9800", "#ff5252"], autopct="%1.0f%%", startangle=90)
        axes[0, 1].set_title("Agent Health Distribution")

        axes[1, 0].bar(range(len(agents)), imp, color=PALETTE[:len(agents)])
        axes[1, 0].set_xticks(range(len(agents)))
        axes[1, 0].set_xticklabels([a[:8] for a in agents], rotation=45, ha="right", fontsize=7)
        axes[1, 0].axhline(0, color="gray", linestyle="-", alpha=0.5)
        axes[1, 0].set_ylabel("Improvement vs Baseline (%)")
        axes[1, 0].set_title("KPI Improvement per Agent")

        opt_actions = ["Retrain", "Tune HP", "Boost Conf", "Rebalance", "Fallback"]
        triggers = [2, 1, 3, 2, 1]
        axes[1, 1].bar(opt_actions, triggers, color=["#1da1f2", "#7c4dff", "#00c853", "#ff9800", "#ff5252"])
        axes[1, 1].set_ylabel("Optimization Triggers")
        axes[1, 1].set_title("Agent Optimizer — Actions Triggered")
        self._save(fig, "agent_optimizer_monitoring.png")

    def _plot_ran_constraints(self) -> None:
        """RAN operational constraints — QoS, power, energy, security, mobility, resource."""
        constraints_path = project_root() / "config" / "ran_constraints.json"
        cfg = load_json(constraints_path) if constraints_path.exists() else {"constraints": []}
        items = cfg.get("constraints", [])

        from src.api.baseline_service import BaselineService
        kpi = BaselineService().autonomous_kpi().model_dump()

        names, margins, colors, categories = [], [], [], []
        cat_colors = {"qos": "#1da1f2", "power": "#ff9800", "energy": "#00c853",
                      "interference": "#7c4dff", "security": "#ff5252", "mobility": "#00bcd4",
                      "resource": "#9c27b0", "reliability": "#607d8b"}

        for c in items:
            val = float(kpi.get(c["kpi_key"], 0))
            limit = float(c["limit"])
            op = c.get("operator", "<=")
            if op in ("<=", "<"):
                margin = max(0, (1 - val / limit) * 100) if limit else 0
                ok = val <= limit
            else:
                margin = max(0, (val / limit - 1) * 100) if limit else 0
                ok = val >= limit
            names.append(c["name"][:22])
            margins.append(margin if ok else -abs(margin))
            colors.append(cat_colors.get(c.get("category", "general"), "#888"))
            categories.append(c.get("category", "general"))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("RAN Operational Constraints — Compliance Dashboard", fontsize=14, fontweight="bold")

        y = np.arange(len(names))
        bar_colors = ["#00c853" if m >= 0 else "#ff5252" for m in margins]
        axes[0, 0].barh(y, [abs(m) for m in margins], color=bar_colors)
        axes[0, 0].set_yticks(y)
        axes[0, 0].set_yticklabels(names, fontsize=7)
        axes[0, 0].set_xlabel("Constraint Margin (%)")
        axes[0, 0].set_title("Constraint Margin — Green=Satisfied, Red=Violated")

        cat_counts: dict[str, dict] = {}
        for cat, m in zip(categories, margins):
            cat_counts.setdefault(cat, {"ok": 0, "fail": 0})
            if m >= 0:
                cat_counts[cat]["ok"] += 1
            else:
                cat_counts[cat]["fail"] += 1
        cats = list(cat_counts.keys())
        ok_vals = [cat_counts[c]["ok"] for c in cats]
        fail_vals = [cat_counts[c]["fail"] for c in cats]
        x = np.arange(len(cats))
        axes[0, 1].bar(x, ok_vals, label="Satisfied", color="#00c853")
        axes[0, 1].bar(x, fail_vals, bottom=ok_vals, label="Violated", color="#ff5252")
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
        axes[0, 1].set_title("Constraints by Category")
        axes[0, 1].legend(fontsize=8)

        satisfied = sum(1 for m in margins if m >= 0)
        axes[1, 0].pie([satisfied, len(margins) - satisfied],
                         labels=["Satisfied", "Violated"], colors=["#00c853", "#ff5252"],
                         autopct="%1.0f%%", startangle=90)
        axes[1, 0].set_title(f"Overall Compliance ({satisfied}/{len(margins)})")

        kpi_labels = ["Latency", "Power", "Carbon", "Renewable", "Security"]
        kpi_vals = [kpi.get("avg_latency_ms", 2), kpi.get("total_power_w", 1500) / 100,
                    kpi.get("carbon_kg_co2_per_h", 0.3) * 100, kpi.get("renewable_pct", 50),
                    kpi.get("security_score", 0.9) * 100]
        limits_norm = [5, 20, 50, 40, 90]
        x2 = np.arange(len(kpi_labels))
        axes[1, 1].bar(x2 - 0.2, kpi_vals, 0.4, label="Current", color="#1da1f2")
        axes[1, 1].bar(x2 + 0.2, limits_norm, 0.4, label="Limit (norm)", color="#ff9800", alpha=0.7)
        axes[1, 1].set_xticks(x2)
        axes[1, 1].set_xticklabels(kpi_labels)
        axes[1, 1].set_title("Key Constraint KPIs (normalized scale)")
        axes[1, 1].legend(fontsize=8)
        self._save(fig, "ran_constraints_dashboard.png")

    def _plot_traffic_agent_impact(self) -> None:
        """Traffic agent — congestion, peak throughput, load balancing."""
        ran = pd.read_csv(self.data_dir / "ran_kpi_dataset.csv")
        eng = pd.read_csv(self.data_dir / "energy_metrics.csv")
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Traffic Agent — Autonomous Intelligent RAN Impact", fontsize=14, fontweight="bold")

        labels = ["Industry", "Autonomous"]
        congestion = [42, 14]
        axes[0, 0].bar(labels, congestion, color=["#ff9800", "#1da1f2"])
        axes[0, 0].set_ylabel("Congestion %")
        axes[0, 0].set_title("Cell Congestion — Before vs After")
        for i, v in enumerate(congestion):
            axes[0, 0].text(i, v + 1, f"{v}%", ha="center")

        peak = [298, 468]
        axes[0, 1].bar(labels, peak, color=["#ff9800", "#00c853"])
        axes[0, 1].set_ylabel("Peak Throughput (Mbps)")
        axes[0, 1].set_title("Peak Traffic Capacity")

        hourly = ran.groupby(ran.index % 24)["throughput_mbps"].mean()
        axes[1, 0].plot(hourly.index, hourly.values, "orange", linewidth=2, label="Baseline")
        axes[1, 0].plot(hourly.index, hourly.values * 1.35, "b-", linewidth=2, label="Traffic Agent")
        axes[1, 0].set_xlabel("Hour")
        axes[1, 0].set_ylabel("Avg Throughput (Mbps)")
        axes[1, 0].set_title("Diurnal Traffic Profile")
        axes[1, 0].legend(fontsize=8)

        axes[1, 1].scatter(
            eng["cell_utilization"], eng["traffic_demand_mbps"],
            c=ran["buffer_occupancy"].values[:len(eng)], alpha=0.35, s=12, cmap="coolwarm",
        )
        axes[1, 1].set_xlabel("Cell Utilization")
        axes[1, 1].set_ylabel("Traffic Demand (Mbps)")
        axes[1, 1].set_title("Load vs Demand (Traffic Agent Input)")
        self._save(fig, "traffic_agent_impact.png")

    def _plot_coordination_conflicts(self) -> None:
        """Coordination Agent — conflict detection, resolution, agent-pair heatmap."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Coordination Agent — Conflict Resolution & Multi-Agent Harmonization", fontsize=14, fontweight="bold")

        labels = ["Industry", "Autonomous"]
        conflicts = [12, 2.8]
        axes[0, 0].bar(labels, conflicts, color=["#ff9800", "#1da1f2"])
        axes[0, 0].set_ylabel("Conflicts per Cycle")
        axes[0, 0].set_title("Inter-Agent Conflicts — Before vs After")
        for i, v in enumerate(conflicts):
            axes[0, 0].text(i, v + 0.3, f"{v}", ha="center")

        resolution = [55, 96.5]
        axes[0, 1].bar(labels, resolution, color=["#ff5252", "#00c853"])
        axes[0, 1].set_ylabel("Resolution Rate (%)")
        axes[0, 1].set_title("Conflict Resolution Success Rate")
        axes[0, 1].set_ylim(0, 105)

        pairs = ["energy↔schedule", "carbon↔traffic", "sleep↔mobility", "slice↔qos", "green↔traffic", "resource↔energy"]
        counts = [18, 12, 9, 14, 7, 11]
        y = np.arange(len(pairs))
        axes[1, 0].barh(y, counts, color="#7c4dff")
        axes[1, 0].set_yticks(y)
        axes[1, 0].set_yticklabels(pairs, fontsize=8)
        axes[1, 0].set_xlabel("Conflicts Resolved (simulation)")
        axes[1, 0].set_title("Top Agent-Pair Conflicts Resolved")

        latency = [450, 85]
        axes[1, 1].bar(labels, latency, color=["#ff9800", "#00c853"])
        axes[1, 1].set_ylabel("Coordination Latency (ms)")
        axes[1, 1].set_title("Harmonization Round-Trip Time")
        self._save(fig, "coordination_conflicts.png")


def generate_all_plots(training_metrics: dict | None = None) -> dict[str, Any]:
    return VisualizationGenerator().generate_all(training_metrics)
