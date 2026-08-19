"""Per-agent performance vs industry baseline for dashboard."""

from __future__ import annotations

from src.api.baseline_service import BaselineService
from src.api.schemas import KPIStats
from src.common.utils import load_config, load_json, project_root


# Primary KPI each agent optimizes; used for baseline vs autonomous comparison.
AGENT_KPI_PROFILE: dict[str, dict] = {
    "scheduler": {"kpi": "avg_throughput_mbps", "label": "Throughput (Mbps)", "role": "PRB / throughput scheduling", "lower_better": False, "weight": 0.12},
    "resource": {"kpi": "avg_throughput_mbps", "label": "Throughput (Mbps)", "role": "PRB allocation & load balance", "lower_better": False, "weight": 0.08},
    "mobility": {"kpi": "handover_success_rate", "label": "HO Success", "role": "Handover & mobility robustness", "lower_better": False, "weight": 0.07},
    "security": {"kpi": "security_score", "label": "Security Detection", "role": "Threat detection & mitigation", "lower_better": False, "weight": 0.10},
    "energy": {"kpi": "total_power_w", "label": "Power (W)", "role": "Energy efficiency & sleep", "lower_better": True, "weight": 0.10},
    "qos": {"kpi": "qos_sla_compliance", "label": "QoS SLA", "role": "SLA enforcement", "lower_better": False, "weight": 0.06},
    "slice": {"kpi": "slice_efficiency", "label": "Slice Efficiency", "role": "Network slice orchestration", "lower_better": False, "weight": 0.08},
    "qoe": {"kpi": "qoe_score", "label": "QoE Score", "role": "Quality of Experience", "lower_better": False, "weight": 0.06},
    "channel_estimation": {"kpi": "csi_accuracy", "label": "CSI Accuracy", "role": "Channel state estimation", "lower_better": False, "weight": 0.05},
    "beamforming": {"kpi": "beamforming_gain_db", "label": "BF Gain (dB)", "role": "MIMO beamforming", "lower_better": False, "weight": 0.06},
    "csi": {"kpi": "csi_accuracy", "label": "CSI Accuracy", "role": "CSI feedback & compression", "lower_better": False, "weight": 0.05},
    "air_interface": {"kpi": "avg_sinr_db", "label": "SINR (dB)", "role": "PHY / MCS adaptation", "lower_better": False, "weight": 0.06},
    "digital_twin": {"kpi": "energy_efficiency", "label": "Energy Efficiency", "role": "Policy what-if validation", "lower_better": False, "weight": 0.05},
    "spectrum": {"kpi": "avg_throughput_mbps", "label": "Throughput (Mbps)", "role": "Spectrum & interference", "lower_better": False, "weight": 0.05},
    "self_healing": {"kpi": "avg_latency_ms", "label": "Latency (ms)", "role": "Fault recovery", "lower_better": True, "weight": 0.05},
    "carbon": {"kpi": "carbon_kg_co2_per_h", "label": "Carbon (kg CO2/h)", "role": "Carbon emission reduction", "lower_better": True, "weight": 0.06},
    "ran_sleep": {"kpi": "total_power_w", "label": "Power (W)", "role": "RAN cell sleep & TX path switch", "lower_better": True, "weight": 0.06},
    "renewable_energy": {"kpi": "renewable_pct", "label": "Renewable %", "role": "Renewable energy routing", "lower_better": False, "weight": 0.06},
    "edge_inference": {"kpi": "avg_latency_ms", "label": "Latency (ms)", "role": "Edge MEC inference offload", "lower_better": True, "weight": 0.06},
    "green_slice": {"kpi": "slice_efficiency", "label": "Slice Efficiency", "role": "Green energy-aware slicing", "lower_better": False, "weight": 0.06},
    "traffic": {"kpi": "traffic_congestion_pct", "label": "Congestion %", "role": "Traffic prediction & load balancing", "lower_better": True, "weight": 0.06},
    "knowledge": {"kpi": "automation_level", "label": "Automation", "role": "3GPP / Nokia CFAM knowledge", "lower_better": False, "weight": 0.03},
    "intent": {"kpi": "automation_level", "label": "Automation", "role": "Operator intent translation", "lower_better": False, "weight": 0.03},
    "agent_optimizer": {"kpi": "automation_level", "label": "Agent Health", "role": "Degraded agent re-optimization", "lower_better": False, "weight": 0.04},
    "coordination": {"kpi": "automation_level", "label": "Coordination Success", "role": "Multi-agent conflict resolution & harmonization", "lower_better": False, "weight": 0.05},
}

PLOT_CATEGORY_RULES: list[tuple[str, str, str]] = [
    ("agent", "Per-Agent Analysis", "agents/"),
    ("kpi", "KPI & Throughput", "hist_ran,cdf_ran,heatmap_ran,heatmap_cell,heatmap_slice_prb,simulation_timeseries"),
    ("security", "Security & Threat Detection", "security,classification_security,threat"),
    ("mobility", "Mobility & Handover", "mobility,handover,trajectory"),
    ("resource", "Resource Allocation & PRB", "resource_allocation,prb_allocation"),
    ("green_edge", "Green & Edge Agents (Sleep, Renewable, Edge, Green Slice)", "ran_sleep,renewable_energy,edge_inference,green_slicing,green_slice,carbon_emission"),
    ("traffic", "Traffic Agent & Load Balancing", "traffic_agent,traffic_optimization"),
    ("coordination", "Coordination & Conflict Resolution", "coordination_agent,coordination_conflicts"),
    ("monitoring", "Super Agent Monitoring & Optimization", "agent_optimizer,agent_monitoring,constraints"),
    ("model", "Model Training & Federated Learning", "model_,federated,phy_channel"),
    ("benchmark", "Benchmark & Comparison", "benchmark,dashboard_autonomous"),
    ("digital_twin", "Digital Twin & Closed Loop", "closed_loop,phy_channel"),
]


class AgentPerformanceService:
    def __init__(self, baseline_svc: BaselineService | None = None):
        self.baseline_svc = baseline_svc or BaselineService()
        self._train_report = self._load_train_report()

    @staticmethod
    def _load_train_report() -> dict:
        path = project_root() / "outputs" / "reports" / "agent_train_validate_test.json"
        return load_json(path) if path.exists() else {}

    def get_comparison(self) -> dict:
        industry = self.baseline_svc.industry_kpi()
        autonomous = self.baseline_svc.autonomous_kpi()
        cmp = self._compare(industry, autonomous)
        agents = []
        for name, profile in AGENT_KPI_PROFILE.items():
            kpi_key = profile["kpi"]
            b_val = getattr(industry, kpi_key, 0)
            a_val = getattr(autonomous, kpi_key, 0)
            imp = cmp["delta_pct"].get(kpi_key, 0)
            if profile["lower_better"]:
                imp = -imp
            val = self._validation_metrics(name)
            agents.append({
                "agent": name,
                "agent_label": name.replace("_", " ").title(),
                "role": profile["role"],
                "primary_kpi": profile["label"],
                "kpi_key": kpi_key,
                "baseline_value": self._fmt_val(kpi_key, b_val),
                "autonomous_value": self._fmt_val(kpi_key, a_val),
                "baseline_raw": b_val,
                "autonomous_raw": a_val,
                "improvement_pct": round(imp, 1),
                "vs_baseline": "improved" if imp > 0 else ("neutral" if imp == 0 else "review"),
                "validation_status": val["status"],
                "validation_metric": val["metric_label"],
                "validation_score": val["metric_value"],
                "validation_raw": val["metric_raw"],
                "confidence": val["confidence"],
                "inference_ok": val["inference_ok"],
                "performance_index": round(self._performance_index(imp, val), 1),
                "weight": profile["weight"],
            })
        agents.sort(key=lambda x: x["performance_index"], reverse=True)
        return {
            "baseline_label": self.baseline_svc.cfg["label"],
            "autonomous_label": self.baseline_svc.cfg["autonomous_intelligent_ran"]["label"],
            "agents": agents,
            "summary": {
                "total_agents": len(agents),
                "improved_count": sum(1 for a in agents if a["vs_baseline"] == "improved"),
                "validated_count": sum(1 for a in agents if a["validation_status"] == "validated"),
                "avg_improvement_pct": round(sum(a["improvement_pct"] for a in agents) / len(agents), 1),
                "avg_confidence": round(sum(a["confidence"] or 0 for a in agents) / len(agents), 3),
                "avg_performance_index": round(sum(a["performance_index"] for a in agents) / len(agents), 1),
            },
        }

    def _compare(self, before: KPIStats, after: KPIStats) -> dict:
        fields = list(KPIStats.model_fields.keys())
        delta_pct = {}
        for f in fields:
            if f == "timestamp":
                continue
            b, a = getattr(before, f), getattr(after, f)
            delta_pct[f] = round((a / b - 1) * 100, 2) if b else 0.0
        return {"delta_pct": delta_pct}

    def _validation_metrics(self, agent: str) -> dict:
        val = self._train_report.get("validation", {}).get(agent, {})
        test = self._train_report.get("testing", {}).get(agent, {})
        status = val.get("status", "n/a")
        conf = val.get("avg_confidence", test.get("avg_confidence"))
        if agent == "knowledge":
            train = self._train_report.get("training", {}).get(agent, {})
            return {"status": status, "metric_label": "KB Coverage", "metric_value": f"{train.get('kg_nodes', 175)} nodes",
                    "metric_raw": train.get("kg_nodes", 175), "confidence": 0.95, "inference_ok": True}
        if agent == "intent":
            train = self._train_report.get("training", {}).get(agent, {})
            return {"status": status, "metric_label": "Intents Supported", "metric_value": str(train.get("intents_supported", 5)),
                    "metric_raw": train.get("intents_supported", 5), "confidence": 0.92, "inference_ok": True}
        for key, label in (("detection_accuracy", "Detection Acc"), ("accuracy", "Accuracy"), ("r2_score", "R2 Fit")):
            if key in val:
                v = val[key]
                disp = f"{v * 100:.1f}%" if key != "r2_score" and v <= 1 else f"{v:.3f}"
                return {"status": status, "metric_label": label, "metric_value": disp,
                        "metric_raw": v, "confidence": conf, "inference_ok": test.get("inference_ok", True)}
        return {"status": status, "metric_label": "—", "metric_value": "—", "metric_raw": 0,
                "confidence": conf, "inference_ok": test.get("inference_ok", False)}

    @staticmethod
    def _fmt_val(kpi_key: str, v: float) -> str:
        if kpi_key in ("security_score", "qos_sla_compliance", "handover_success_rate",
                       "csi_accuracy", "slice_efficiency", "automation_level", "fairness_index"):
            return f"{v * 100:.1f}%" if v <= 1 else f"{v:.2f}"
        if kpi_key == "qoe_score":
            return f"{v:.2f}"
        return f"{v:.2f}"

    @staticmethod
    def _performance_index(improvement_pct: float, val: dict) -> float:
        imp = max(0, min(100, improvement_pct))
        raw = val.get("metric_raw") or 0
        if val.get("metric_label") == "R2 Fit":
            score = max(0, min(100, raw * 100))
        elif isinstance(raw, (int, float)) and raw <= 1:
            score = raw * 100
        elif isinstance(raw, (int, float)) and raw > 1:
            score = min(100, raw / 2)
        else:
            score = 80
        conf = (val.get("confidence") or 0.8) * 100
        return imp * 0.45 + score * 0.35 + conf * 0.20

    @staticmethod
    def categorize_plots(plots: list[dict]) -> dict:
        categories: dict[str, list] = {r[0]: [] for r in PLOT_CATEGORY_RULES}
        categories["other"] = []
        for p in plots:
            url = p.get("url", "")
            name = p.get("name", "")
            assigned = False
            for cat_id, cat_label, patterns in PLOT_CATEGORY_RULES:
                if any(pat in url or pat in name for pat in patterns.split(",")):
                    categories[cat_id].append({**p, "category_label": cat_label})
                    assigned = True
                    break
            if not assigned:
                categories["other"].append({**p, "category_label": "Other"})
        result = []
        for cat_id, cat_label, _ in PLOT_CATEGORY_RULES:
            items = categories[cat_id]
            if items:
                result.append({"id": cat_id, "label": cat_label, "plots": items, "count": len(items)})
        if categories["other"]:
            result.append({"id": "other", "label": "Other", "plots": categories["other"], "count": len(categories["other"])})
        return {"categories": result, "total": sum(c["count"] for c in result)}
