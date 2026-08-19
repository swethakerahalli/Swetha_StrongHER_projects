"""Global RAN state: digital twin, KPI baseline/history, action log."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import numpy as np

from src.api.baseline_service import BaselineService
from src.api.schemas import KPIStats, TargetKPIProgress
from src.common.utils import load_config
from src.digital_twin.ran_twin import RANDigitalTwin
from src.orchestration.super_agent_controller import SuperAgentController


class RANStateStore:
    _instance: RANStateStore | None = None

    def __init__(self):
        self.twin = RANDigitalTwin()
        self.controller = SuperAgentController()
        self.baseline_kpi: KPIStats | None = None
        self.current_kpi: KPIStats | None = None
        self.kpi_history: list[dict] = []
        self.parameter_changes: list[dict] = []
        self.action_log: list[dict] = []
        self.baseline_svc = BaselineService()
        self._init_baseline()

    @classmethod
    def get(cls) -> RANStateStore:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_baseline(self) -> None:
        self.baseline_kpi = self.baseline_svc.industry_kpi()
        self.current_kpi = self.baseline_svc.autonomous_kpi()
        self.kpi_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": "industry_baseline",
            "kpi": self.baseline_kpi.model_dump(),
        })
        self.kpi_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": "autonomous_ran",
            "kpi": self.current_kpi.model_dump(),
        })

    def get_industry_comparison(self) -> dict:
        """Industry baseline vs Autonomous Intelligent RAN (not twin initial state)."""
        before = self.baseline_svc.industry_kpi()
        live = self.compute_kpi()
        after = self.baseline_svc.autonomous_kpi(live)
        cmp = self.compare_kpi(before, after)
        return {
            "before": before,
            "after": after,
            "before_label": self.baseline_svc.cfg["label"],
            "after_label": self.baseline_svc.cfg["autonomous_intelligent_ran"]["label"],
            **cmp,
        }

    def compute_kpi(self) -> KPIStats:
        state = self.twin.observe()
        ues = list(self.twin.ues.values())
        tps = [u.throughput_mbps for u in ues]
        lats = [u.latency_ms for u in ues]
        cqis = [u.cqi for u in ues]
        sinrs = [u.sinr_db for u in ues]
        n = len(ues) or 1
        fairness = self._jain_fairness(tps)
        power = state["total_power_w"]
        tp_sum = sum(tps)
        slice_eff = self._compute_slice_efficiency()
        auto_level = min(1.0, len(self.controller.agents) / max(len(self.controller.agents), 1))
        carbon_kg = round(power / 1000 * getattr(self.twin, "carbon_intensity_gco2_kwh", 380) / 1000, 4)
        industry_carbon = round(2800 / 1000 * 380 / 1000, 4)
        prb_util = round(sum(c.load for c in self.twin.cells.values()) / max(len(self.twin.cells), 1) * 100, 1)
        congested = sum(1 for c in self.twin.cells.values() if c.load > 0.75)
        traffic_congestion = round(congested / max(len(self.twin.cells), 1) * 100, 1)
        peak_traffic = round(max(tps) if tps else 0, 2)
        kpi = KPIStats(
            avg_throughput_mbps=round(float(np.mean(tps)), 2),
            avg_latency_ms=round(float(np.mean(lats)), 2),
            avg_cqi=round(float(np.mean(cqis)), 2),
            avg_sinr_db=round(float(np.mean(sinrs)), 2),
            total_power_w=round(power, 2),
            fairness_index=round(fairness, 3),
            handover_success_rate=getattr(self.controller, "handover_success_rate", 0.98),
            security_score=getattr(self.controller, "security_score", 0.95),
            qoe_score=round(min(5.0, 5.0 - 0.01 * float(np.mean(lats))), 2),
            qos_sla_compliance=round(min(1.0, 1.0 - float(np.mean(lats)) / 100), 3),
            energy_efficiency=round(tp_sum / (power + 1), 4),
            beamforming_gain_db=getattr(self.twin, "beamforming_gain_db", 3.0),
            csi_accuracy=getattr(self.twin, "csi_accuracy", 0.92),
            slice_efficiency=round(slice_eff, 3),
            automation_level=round(auto_level, 3),
            global_utility=round(self._compute_utility(tp_sum / n, float(np.mean(lats)), power), 3),
            carbon_kg_co2_per_h=carbon_kg,
            carbon_intensity_gco2_kwh=getattr(self.twin, "carbon_intensity_gco2_kwh", 380.0),
            renewable_pct=getattr(self.twin, "renewable_pct", 15.0),
            prb_utilization_pct=prb_util,
            traffic_congestion_pct=traffic_congestion,
            peak_traffic_mbps=peak_traffic,
            timestamp=state["timestamp"],
        )
        return kpi

    def _compute_slice_efficiency(self) -> float:
        slices = {}
        for ue in self.twin.ues.values():
            slices.setdefault(ue.slice, []).append(ue.throughput_mbps)
        if not slices:
            return 0.9
        means = [float(np.mean(v)) for v in slices.values()]
        avg = float(np.mean(means))
        std = float(np.std(means))
        if std < 1e-6:
            return round(min(1.0, avg / 100.0), 3)
        cv = std / (avg + 1e-9)
        return round(max(0.0, min(1.0, 1.0 - cv)), 3)

    def _compute_utility(self, tp: float, lat: float, power: float) -> float:
        cfg = load_config("kpis.json")
        w = cfg["utility_function"]["weights"]
        sec = getattr(self.controller, "security_score", 0.95)
        return (
            w["alpha_throughput"] * tp
            - w["beta_latency"] * lat
            + w["gamma_security"] * sec * 10
            - w["epsilon_energy"] * power / 1000
        )

    def compute_target_kpis(self) -> list[TargetKPIProgress]:
        cmp = self.get_industry_comparison()
        before, after = cmp["before"], cmp["after"]
        d = cmp["delta_pct"]
        cfg = load_config("kpis.json")["target_kpis"]
        labels = {
            "throughput_improvement_pct": "Throughput Improvement",
            "latency_reduction_pct": "Latency Reduction",
            "energy_reduction_pct": "Energy Reduction",
            "security_detection_accuracy_pct": "Security Detection",
            "handover_success_rate_pct": "Handover Success",
            "qoe_improvement_pct": "QoE Improvement",
            "slice_efficiency_pct": "Slice Efficiency",
            "automation_level_pct": "Automation Level",
            "opex_reduction_pct": "OPEX Reduction",
        }
        values = {
            "throughput_improvement_pct": max(0, d.get("avg_throughput_mbps", 0)),
            "latency_reduction_pct": max(0, -d.get("avg_latency_ms", 0)),
            "energy_reduction_pct": max(0, -d.get("total_power_w", 0)),
            "security_detection_accuracy_pct": after.security_score * 100,
            "handover_success_rate_pct": after.handover_success_rate * 100,
            "qoe_improvement_pct": max(0, (after.qoe_score / max(before.qoe_score, 0.1) - 1) * 100),
            "slice_efficiency_pct": after.slice_efficiency * 100,
            "automation_level_pct": after.automation_level * 100,
            "opex_reduction_pct": max(0, -d.get("total_power_w", 0) * 0.5),
        }
        results = []
        for key, meta in cfg.items():
            current = values.get(key, 0)
            target = meta.get("target", meta.get("target_min", 0))
            tmin, tmax = meta.get("target_min"), meta.get("target_max")
            if tmax:
                target = (tmin + tmax) / 2
            progress = min(100, (current / target * 100) if target else 0)
            achieved = current >= target if not tmax else (tmin <= current <= tmax or current >= tmin)
            results.append(TargetKPIProgress(
                kpi_key=key, label=labels.get(key, key),
                current_value=round(current, 2), target_value=round(target, 2),
                target_min=tmin, target_max=tmax, unit=meta.get("unit", "percent"),
                achieved=achieved, progress_pct=round(progress, 1),
            ))
        return results

    @staticmethod
    def _jain_fairness(values: list[float]) -> float:
        if not values:
            return 1.0
        s = sum(values)
        if s == 0:
            return 0.0
        n = len(values)
        return (s ** 2) / (n * sum(v ** 2 for v in values) + 1e-9)

    def snapshot_before(self) -> KPIStats:
        return self.compute_kpi()

    def reset_baseline(self) -> KPIStats:
        self.baseline_kpi = self.baseline_svc.industry_kpi()
        self.kpi_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": "industry_baseline_reset",
            "kpi": self.baseline_kpi.model_dump(),
        })
        return self.baseline_kpi

    def record_kpi(self, label: str = "update") -> KPIStats:
        kpi = self.compute_kpi()
        self.current_kpi = kpi
        self.kpi_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "label": label,
            "kpi": kpi.model_dump(),
        })
        return kpi

    def log_parameter_change(self, source: str, target: str, before: dict, after: dict, agent: str = "") -> dict:
        entry = {
            "id": str(uuid.uuid4())[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "target": target,
            "agent": agent,
            "before": before,
            "after": after,
        }
        self.parameter_changes.append(entry)
        return entry

    def log_action(self, action_id: str, agents: list[str], kpi_before: KPIStats,
                   kpi_after: KPIStats, updates: list[dict], decision: dict) -> dict:
        entry = {
            "action_id": action_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agents": agents,
            "kpi_before": kpi_before.model_dump(),
            "kpi_after": kpi_after.model_dump(),
            "parameter_updates": updates,
            "super_agent_decision": decision,
        }
        self.action_log.append(entry)
        return entry

    def compare_kpi(self, before: KPIStats, after: KPIStats) -> dict:
        fields = list(KPIStats.model_fields.keys())
        delta, delta_pct = {}, {}
        for f in fields:
            if f == "timestamp":
                continue
            b, a = getattr(before, f), getattr(after, f)
            delta[f] = round(a - b, 4)
            delta_pct[f] = round((a / b - 1) * 100, 2) if b else 0.0
        return {"delta": delta, "delta_pct": delta_pct}
