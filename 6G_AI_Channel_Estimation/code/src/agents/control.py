"""Coordinator (conflict resolution) and Super Agent (control plane) for CSI agents."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent

COORD_FEATS = ["snr_db", "nmse_ai", "pilot_overhead", "csi_pred_accuracy", "trust_score", "doppler_hz"]


class CoordinatorAgent(BaseAgent):
    """Detects and resolves conflicts among channel-estimation domain agents."""

    PRIORITY = [
        "security",
        "mitigation",
        "self_healing",
        "digital_twin",
        "channel",
        "pilot",
        "mobility",
        "beam",
        "spectrum",
        "optimization",
        "csi_prediction",
        "csi_feedback",
        "resource",
        "air_interface",
        "equalizer",
        "knowledge",
        "explainability",
        "orchestrator",
    ]

    def __init__(self, model_path=None):
        super().__init__("coordinator", model_path)
        self.conflict_log: list[dict] = []
        self.cycles = 0

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        df = data.copy()
        for col in COORD_FEATS:
            if col not in df.columns:
                df[col] = 0.5
        conflict = (
            ((df.get("nmse_ai", 0) > 0.1) & (df.get("csi_pred_accuracy", 0) > 0.93))
            | (df.get("is_attack", 0) == 1)
        ).astype(int)
        resolved = (conflict == 0) | (df.get("trust_score", 1) > 0.4)
        df["resolution_ok"] = resolved.astype(int)
        tr, va, te = (df[df["split"] == s] for s in ("train", "validation", "test"))
        self.model = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42)
        self.model.fit(tr[COORD_FEATS].fillna(0), tr["resolution_ok"])
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[COORD_FEATS].fillna(0))
            metrics[f"{name}_accuracy"] = round(float(accuracy_score(part["resolution_ok"], pred)), 4)
        metrics["conflict_rate_test"] = round(float(conflict.loc[te.index].mean()), 4)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        actions = observation.context.get("agent_actions", {})
        result = self.coordinate(actions, observation.features)
        return AgentAction(
            self.agent_id,
            "coordinate",
            {
                "conflicts": result["conflicts"],
                "resolutions": result["resolutions"],
                "strategy": result["strategy"],
                "harmonized_policy": result["harmonized_policy"],
                "n_conflicts": result["n_conflicts"],
            },
            result["confidence"],
        )

    def coordinate(self, actions: dict[str, dict], features: dict[str, float]) -> dict[str, Any]:
        params = {k: (v.get("parameters") if isinstance(v, dict) and "parameters" in v else v) for k, v in actions.items()}
        conflicts: list[dict] = []
        resolutions: list[str] = []
        policy = params.get("orchestrator", {}).get("global_policy", "hold")

        pred_intent = params.get("csi_prediction", {}).get("reduce_pilot") or params.get("pilot", {}).get("intent") == "reduce_pilots"
        high_nmse = float(features.get("nmse_ai", 0)) > 0.1
        if pred_intent and high_nmse:
            conflicts.append({"pair": ["csi_prediction/pilot", "channel"], "type": "pilot_density"})
            resolutions.append("Keep or increase DMRS/CSI-RS density because NMSE is above 0.1")
            policy = "increase_pilots"
            if "pilot" in params:
                params["pilot"]["intent"] = "increase_pilots"
                params["pilot"]["pilot_density"] = "dense"

        attack = params.get("security", {}).get("attack_type", "normal")
        mit = params.get("mitigation", {}).get("action", "none")
        if attack != "normal" and mit == "beam_switch":
            beam = params.get("beam", {}).get("beam_index")
            if beam is not None:
                conflicts.append({"pair": ["mitigation", "beam"], "type": "beam_target"})
                resolutions.append("Security wins: switch beam; freeze current beam-manager index")
                params["beam"]["beam_index"] = int((int(beam) + 3) % 8)
                params["beam"]["frozen_by"] = "coordinator"
                policy = f"mitigate:{mit}"

        hop = params.get("spectrum", {}).get("frequency_hop") or mit == "frequency_hop"
        hold_carrier = params.get("optimization") is not None and attack == "normal"
        if hop and hold_carrier and attack != "normal":
            conflicts.append({"pair": ["spectrum/mitigation", "optimization"], "type": "carrier"})
            resolutions.append("Jamming/contamination: hop carrier; defer SE maximization")
            policy = "mitigate:frequency_hop"

        ho = params.get("mobility", {}).get("handover_pending")
        if ho and high_nmse:
            conflicts.append({"pair": ["mobility", "channel"], "type": "handover_during_poor_csi"})
            resolutions.append("Allow predictive HO only after one extra DMRS burst")
            policy = "increase_pilots_then_ho"

        twin_ok = params.get("digital_twin", {}).get("safe_to_deploy", True)
        if twin_ok is False or float(features.get("nmse_ai", 0)) > 0.25:
            conflicts.append({"pair": ["digital_twin", "orchestrator"], "type": "unsafe_deploy"})
            resolutions.append("Twin veto: do not reduce pilots or change beam until fidelity recovers")
            policy = "hold"

        isolate = mit == "trust_scheduling"
        boost = params.get("resource", {}).get("prb_boost")
        if isolate and boost:
            conflicts.append({"pair": ["mitigation", "resource"], "type": "isolate_vs_prb"})
            resolutions.append("Do not boost PRBs for untrusted UE; isolate first")
            params["resource"]["prb_boost"] = False

        heal = params.get("self_healing", {}).get("fallback_estimator")
        if heal == "mmse" and params.get("channel", {}).get("method", "").startswith("ensemble"):
            conflicts.append({"pair": ["self_healing", "channel"], "type": "estimator_fallback"})
            resolutions.append("Poisoning/adversarial: fall back to MMSE until trust recovers")
            params["channel"]["method"] = "mmse_fallback"

        self.cycles += 1
        strategy = "security_first" if attack != "normal" else ("nmse_guard" if high_nmse else "se_optimize")
        rec = {
            "cycle": self.cycles,
            "conflicts": conflicts,
            "resolutions": resolutions,
            "strategy": strategy,
            "harmonized_policy": policy,
            "n_conflicts": len(conflicts),
            "confidence": round(0.95 - 0.03 * len(conflicts), 3),
            "harmonized_params": params,
        }
        self.conflict_log.append({"cycle": self.cycles, "n_conflicts": len(conflicts), "strategy": strategy, "policy": policy})
        return rec


class SuperAgent(BaseAgent):
    """Controls enablement, validation, and actuation of all CSI agents."""

    WEIGHTS = {
        "security": 0.14,
        "mitigation": 0.10,
        "self_healing": 0.08,
        "channel": 0.12,
        "pilot": 0.08,
        "digital_twin": 0.08,
        "coordinator": 0.08,
        "csi_prediction": 0.06,
        "csi_feedback": 0.05,
        "beam": 0.05,
        "mobility": 0.04,
        "spectrum": 0.04,
        "air_interface": 0.03,
        "equalizer": 0.03,
        "optimization": 0.04,
        "resource": 0.03,
        "explainability": 0.02,
        "knowledge": 0.01,
        "orchestrator": 0.02,
    }

    def __init__(self, model_path=None):
        super().__init__("super", model_path)
        self.enabled = {k: True for k in self.WEIGHTS}
        self.validation_log: list[dict] = []
        self.monitoring_log: list[dict] = []

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        df = data.copy()
        feats = ["snr_db", "nmse_ai", "trust_score", "csi_pred_accuracy"]
        for c in feats:
            if c not in df.columns:
                df[c] = 0.5
        y = ((df["nmse_ai"] < 0.2) & (df["trust_score"] > 0.3)).astype(int)
        tr, va, te = (df[df["split"] == s] for s in ("train", "validation", "test"))
        self.model = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42)
        self.model.fit(tr[feats].fillna(0), y.loc[tr.index])
        self.feature_keys = feats
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[feats].fillna(0))
            metrics[f"{name}_accuracy"] = round(float(accuracy_score(y.loc[part.index], pred)), 4)
        metrics["n_controlled_agents"] = len(self.WEIGHTS)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        result = self.control(observation.context.get("agent_actions", {}), observation.features, observation.context.get("coordination", {}))
        return AgentAction(self.agent_id, "super_control", result, result.get("confidence", 0.9))

    def control(self, actions: dict, features: dict[str, float], coordination: dict | None = None) -> dict[str, Any]:
        approved, rejected = [], []
        coord = coordination or {}
        fidelity_ok = bool(actions.get("digital_twin", {}).get("parameters", actions.get("digital_twin", {})).get("safe_to_deploy", True))
        if isinstance(actions.get("digital_twin"), dict) and "parameters" in actions["digital_twin"]:
            fidelity_ok = bool(actions["digital_twin"]["parameters"].get("safe_to_deploy", True))

        for name, payload in actions.items():
            if name in {"coordinator", "super"}:
                continue
            if not self.enabled.get(name, True):
                rejected.append({"agent": name, "reason": "disabled_by_super_agent"})
                continue
            params = payload.get("parameters", payload) if isinstance(payload, dict) else {}
            attack = params.get("attack_type") if name == "security" else None
            if name == "csi_prediction" and float(features.get("nmse_ai", 0)) > 0.15:
                rejected.append({"agent": name, "reason": "NMSE too high to cut pilots"})
                continue
            if name in {"optimization", "resource"} and not fidelity_ok:
                rejected.append({"agent": name, "reason": "twin_fidelity_gate"})
                continue
            approved.append(name)

        utility = 0.0
        for name in approved:
            utility += self.WEIGHTS.get(name, 0.02)
        nmse = float(features.get("nmse_ai", 0.05))
        utility *= float(np.clip(1.0 - nmse, 0.4, 1.0))
        if coord.get("n_conflicts", 0):
            utility *= 0.92

        decision = {
            "approved_agents": approved,
            "rejected": rejected,
            "approved_count": len(approved),
            "rejected_count": len(rejected),
            "global_utility": round(float(utility), 4),
            "enabled": dict(self.enabled),
            "policy": coord.get("harmonized_policy", "hold"),
            "conflicts_resolved": coord.get("n_conflicts", 0),
            "autonomy_level": round(len(approved) / max(len(actions), 1), 3),
            "confidence": 0.94,
        }
        self.validation_log.append(decision)
        return decision

    def set_enabled(self, agent_id: str, enabled: bool) -> None:
        self.enabled[agent_id] = enabled

    def get_status(self) -> dict[str, Any]:
        last = self.validation_log[-1] if self.validation_log else {}
        return {
            "enabled": dict(self.enabled),
            "weights": dict(self.WEIGHTS),
            "n_controlled_agents": len(self.WEIGHTS),
            "n_validations": len(self.validation_log),
            "last_decision": last,
            "monitoring": self.monitoring_log[-1] if self.monitoring_log else {},
        }

    def monitor(self, metrics_by_agent: dict[str, dict]) -> dict[str, Any]:
        degraded, warning = [], []
        for name, m in metrics_by_agent.items():
            acc = m.get("test_accuracy", m.get("test_r2", m.get("binary_test_accuracy", m.get("ensemble_test_r2"))))
            if acc is None:
                continue
            if float(acc) < 0.2:
                degraded.append(name)
            elif float(acc) < 0.5:
                warning.append(name)
        snap = {"degraded_agents": degraded, "warning_agents": warning, "healthy": len(metrics_by_agent) - len(degraded) - len(warning)}
        self.monitoring_log.append(snap)
        return snap
