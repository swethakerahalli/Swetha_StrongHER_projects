"""Beam, mobility, optimization, digital twin, explainability, orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, r2_score

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent

PHY = ["snr_db", "sinr_db", "cqi", "doppler_hz", "delay_spread_ns", "n_tx", "velocity_mps"]


class BeamAgent(BaseAgent):
    def __init__(self, model_path=None):
        super().__init__("beam", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        tr, va, te = (data[data["split"] == s] for s in ("train", "validation", "test"))
        y_tr = (tr["beam_index"] % 8).astype(int)
        self.model = RandomForestClassifier(n_estimators=70, max_depth=8, random_state=42, n_jobs=-1)
        self.model.fit(tr[PHY].fillna(0), y_tr)
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[PHY].fillna(0))
            y = (part["beam_index"] % 8).astype(int)
            metrics[f"{name}_accuracy"] = round(float(accuracy_score(y, pred)), 4)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        X = self._to_array(observation.features, PHY)
        beam = int(self.model.predict(X)[0]) if self.is_trained else int(observation.features.get("beam_index", 0)) % 8
        return AgentAction(self.agent_id, "beam_select", {"beam_index": beam}, confidence=0.91)


class MobilityAgent(BaseAgent):
    FEATS = ["velocity_mps", "rsrp_dbm", "snr_db", "neighbor_rsrp_dbm"]

    def __init__(self, model_path=None):
        super().__init__("mobility", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        tr, va, te = (data[data["split"] == s] for s in ("train", "validation", "test"))
        self.model = GradientBoostingClassifier(n_estimators=70, max_depth=3, random_state=42)
        self.model.fit(tr[self.FEATS].fillna(0), tr["handover_pending"])
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[self.FEATS].fillna(0))
            metrics[f"{name}_accuracy"] = round(float(accuracy_score(part["handover_pending"], pred)), 4)
            metrics[f"{name}_ho_success"] = round(float(part["handover_success"].mean()), 4)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        X = self._to_array(observation.features, self.FEATS)
        pending = int(self.model.predict(X)[0]) if self.is_trained else 0
        return AgentAction(self.agent_id, "mobility", {"handover_pending": pending, "predictive_ho": bool(pending)}, confidence=0.9)


class OptimizationAgent(BaseAgent):
    FEATS = ["snr_db", "nmse_ai", "pilot_overhead", "cqi", "n_tx"]

    def __init__(self, model_path=None):
        super().__init__("optimization", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        tr, va, te = (data[data["split"] == s] for s in ("train", "validation", "test"))
        y = tr["se_ai"]
        self.model = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=42)
        self.model.fit(tr[self.FEATS].fillna(0), y)
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[self.FEATS].fillna(0))
            metrics[f"{name}_r2"] = round(float(r2_score(part["se_ai"], pred)), 4)
            metrics[f"{name}_se_gain_vs_mmse"] = round(float((part["se_ai"].mean() / part["se_mmse"].mean() - 1) * 100), 2)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        X = self._to_array(observation.features, self.FEATS)
        se = float(self.model.predict(X)[0]) if self.is_trained else 4.0
        return AgentAction(self.agent_id, "optimize", {"predicted_se_bps_hz": round(se, 3)}, confidence=0.88)


class DigitalTwinAgent(BaseAgent):
    FEATS = ["snr_db", "nmse_ai", "trust_score", "load"]

    def __init__(self, model_path=None):
        super().__init__("digital_twin", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        n = len(data)
        split = np.array(["train"] * n, dtype=object)
        n_tr = int(0.7 * n)
        n_va = int(0.15 * n)
        split[n_tr:n_tr + n_va] = "validation"
        split[n_tr + n_va:] = "test"
        data = data.copy()
        data["split"] = split
        tr, va, te = (data[data["split"] == s] for s in ("train", "validation", "test"))
        self.model = GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=42)
        self.model.fit(tr[self.FEATS].fillna(0), tr["twin_fidelity"])
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[self.FEATS].fillna(0))
            metrics[f"{name}_r2"] = round(float(r2_score(part["twin_fidelity"], pred)), 4)
            metrics[f"{name}_mean_fidelity"] = round(float(np.mean(pred)), 4)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        X = self._to_array(observation.features, self.FEATS)
        fid = float(self.model.predict(X)[0]) if self.is_trained else 0.9
        return AgentAction(self.agent_id, "twin_validate", {"fidelity": round(fid, 4), "safe_to_deploy": fid > 0.85}, confidence=fid)


class ExplainabilityAgent(BaseAgent):
    def __init__(self, model_path=None):
        super().__init__("explainability", model_path)
        self.importances = {
            k.replace("importance_", ""): v
            for k, v in self.metrics.items()
            if k.startswith("importance_")
        }

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        from sklearn.ensemble import RandomForestRegressor

        feats = ["snr_db", "doppler_hz", "delay_spread_ns", "n_tx", "attack_severity", "cqi"]
        tr = data[data["split"] == "train"].sample(min(4000, len(data[data["split"] == "train"])), random_state=42)
        model = RandomForestRegressor(n_estimators=30, max_depth=6, random_state=42, n_jobs=-1)
        X = tr[feats].fillna(0)
        y = tr["nmse_ai"]
        model.fit(X, y)
        result = permutation_importance(model, X, y, n_repeats=1, random_state=42, n_jobs=-1)
        self.importances = {f: round(float(v), 5) for f, v in zip(feats, result.importances_mean)}
        self.model = model
        self.metrics = {f"importance_{k}": v for k, v in self.importances.items()}
        self.is_trained = True
        return self.metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        ranked = sorted(self.importances.items(), key=lambda kv: -kv[1])[:3]
        return AgentAction(self.agent_id, "explain", {"top_features": ranked}, confidence=0.86)


class OrchestratorAgent(BaseAgent):
    def __init__(self, model_path=None):
        super().__init__("orchestrator", model_path)
        self.history = []

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        te = data[data["split"] == "test"]
        self.metrics = {
            "test_mean_nmse_ai": round(float(te["nmse_ai"].mean()), 6),
            "test_mean_nmse_mmse": round(float(te["nmse_mmse"].mean()), 6),
            "nmse_improvement_pct": round(float((1 - te["nmse_ai"].mean() / te["nmse_mmse"].mean()) * 100), 2),
            "attack_rate": round(float(te["is_attack"].mean()), 4),
        }
        self.is_trained = True
        self.model = self.metrics
        return self.metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        actions = observation.context.get("agent_actions", {})
        attack = actions.get("security", {}).get("attack_type", "normal")
        mitigation = actions.get("mitigation", {}).get("action", "none")
        nmse_val = float(observation.features.get("nmse_ai", 0.05))
        policy = "hold"
        if attack != "normal":
            policy = f"mitigate:{mitigation}"
        elif nmse_val > 0.2:
            policy = "increase_pilots"
        elif observation.features.get("csi_pred_accuracy", 0.9) > 0.94:
            policy = "reduce_pilots"
        self.history.append(policy)
        return AgentAction(self.agent_id, "orchestrate", {"global_policy": policy, "attack": attack}, confidence=0.93)
