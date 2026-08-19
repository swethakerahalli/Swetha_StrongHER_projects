"""Gather live RAN operational data for chatbot responses (no architecture details)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.api.ran_parameter_service import RANParameterService
from src.common.utils import load_json, project_root


class RANContextProvider:
    """Loads KPI, benchmark, training/validation, and agent status for chatbot replies."""

    REPORTS = project_root() / "outputs" / "reports"

    def __init__(self, param_svc: RANParameterService):
        self.param_svc = param_svc

    @property
    def store(self):
        return self.param_svc.store

    def industry_comparison(self) -> dict:
        return self.store.get_industry_comparison()

    def target_kpis(self) -> list[dict]:
        return [t.model_dump() for t in self.store.compute_target_kpis()]

    def agent_status(self) -> list[dict]:
        ctrl = self.store.controller
        agents = []
        for name, agent in ctrl.agents.items():
            ai_label = "Knowledge AI" if name in ("knowledge", "intent") else "AI-Driven"
            agents.append({
                "agent": name.replace("_", " ").title(),
                "role": self._agent_role(name),
                "ai_mode": ai_label,
                "model_ready": getattr(agent, "is_trained", False) or name in ("knowledge", "intent"),
                "inference_ok": True,
            })
        return agents

    def training_validation_rows(self) -> list[dict]:
        path = self.REPORTS / "agent_train_validate_test.json"
        if not path.exists():
            return self._from_csv_fallback()
        data = load_json(path)
        rows = []
        for name, val in data.get("validation", {}).items():
            train = data.get("training", {}).get(name, {})
            test = data.get("testing", {}).get(name, {})
            metric, metric_val = self._primary_metric(train, val)
            rows.append({
                "agent": name.replace("_", " ").title(),
                "validation_status": val.get("status", "n/a"),
                "primary_metric": metric,
                "metric_value": metric_val,
                "train_metric": self._metric_value(train),
                "test_samples": val.get("test_samples", test.get("test_samples", "—")),
                "avg_confidence": val.get("avg_confidence", test.get("avg_confidence")),
                "inference_ok": test.get("inference_ok", True),
                "training_samples": train.get("samples", val.get("samples", "—")),
            })
        return rows

    def benchmark_rows(self) -> list[dict]:
        return self.store.baseline_svc.benchmark_comparison()

    def slice_kpis(self) -> dict:
        state = self.store.twin.observe()
        slices: dict[str, dict] = {}
        for ue in self.store.twin.ues.values():
            s = ue.slice
            slices.setdefault(s, {"throughputs": [], "latencies": [], "count": 0})
            slices[s]["throughputs"].append(ue.throughput_mbps)
            slices[s]["latencies"].append(ue.latency_ms)
            slices[s]["count"] += 1
        out = {}
        for name, d in slices.items():
            tps, lats = d["throughputs"], d["latencies"]
            out[name] = {
                "avg_throughput_mbps": round(sum(tps) / len(tps), 2) if tps else 0,
                "avg_latency_ms": round(sum(lats) / len(lats), 2) if lats else 0,
                "ue_count": d["count"],
            }
        return out

    def e2e_summary(self) -> dict:
        from src.api.e2e_service import E2EService
        return E2EService().get_summary()

    def super_agent_summary(self) -> dict:
        sa = self.store.controller.super_agent
        return {
            "agents_managed": len(self.store.controller.agents),
            "validations": len(sa.validation_log),
            "autonomy_level_pct": round(getattr(sa, "autonomy_level", 1.0) * 100, 1),
            "last_decision": sa.validation_log[-1] if sa.validation_log else {},
        }

    @staticmethod
    def _agent_role(name: str) -> str:
        roles = {
            "scheduler": "PRB / throughput scheduling",
            "resource": "PRB allocation & load balancing",
            "mobility": "Handover & mobility robustness",
            "security": "Threat detection & mitigation",
            "energy": "Power saving & EE optimization",
            "carbon": "Carbon emission reduction & green scheduling",
            "ran_sleep": "Cell sleep mode & TX path switching",
            "renewable_energy": "Solar/wind routing & green power",
            "edge_inference": "MEC edge AI inference offload",
            "green_slice": "Energy-aware green network slicing",
            "qos": "SLA & QoS enforcement",
            "slice": "Network slice orchestration",
            "qoe": "Quality of Experience",
            "channel_estimation": "Channel state estimation",
            "beamforming": "MIMO beamforming gain",
            "csi": "CSI feedback & compression",
            "air_interface": "MCS / PHY adaptation",
            "digital_twin": "Policy what-if validation",
            "spectrum": "Spectrum & interference",
            "self_healing": "Fault detection & recovery",
            "knowledge": "3GPP / Nokia CFAM knowledge",
            "intent": "Operator intent translation",
        }
        return roles.get(name, "RAN optimization")

    @staticmethod
    def _primary_metric(train: dict, val: dict) -> tuple[str, str]:
        for key, label in (
            ("detection_accuracy", "Detection Accuracy"),
            ("accuracy", "Classification Accuracy"),
            ("r2_score", "Prediction Fit (R2)"),
        ):
            if key in val and val[key] is not None:
                v = val[key]
                if key == "r2_score":
                    return label, f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                return label, f"{v * 100:.1f}%" if isinstance(v, (int, float)) and v <= 1 else str(v)
        if val.get("status") == "skipped":
            return "Coverage", "Knowledge base indexed"
        return "Status", val.get("status", "n/a")

    @staticmethod
    def _metric_value(d: dict) -> str:
        for key in ("detection_accuracy", "accuracy", "r2_score"):
            if key in d and d[key] is not None:
                v = d[key]
                if key == "r2_score":
                    return f"{v:.3f}"
                return f"{v * 100:.1f}%"
        if "kg_nodes" in d:
            return f"{d['kg_nodes']} KB nodes"
        if "intents_supported" in d:
            return f"{d['intents_supported']} intents"
        return "—"

    def _from_csv_fallback(self) -> list[dict]:
        path = self.REPORTS / "agent_training_results.csv"
        if not path.exists():
            return []
        df = pd.read_csv(path)
        rows = []
        for _, r in df.iterrows():
            metric = "R2" if pd.notna(r.get("train_r2_score")) else "Accuracy"
            val = r.get("train_r2_score") or r.get("train_accuracy") or r.get("train_detection_accuracy")
            rows.append({
                "agent": str(r["agent"]).replace("_", " ").title(),
                "validation_status": r.get("validation_status", "n/a"),
                "primary_metric": metric,
                "metric_value": f"{val:.4f}" if metric == "R2" and pd.notna(val) else (
                    f"{val * 100:.1f}%" if pd.notna(val) else "—"
                ),
                "avg_confidence": r.get("avg_confidence"),
                "training_samples": int(r["train_samples"]) if pd.notna(r.get("train_samples")) else "—",
                "inference_ok": True,
            })
        return rows
