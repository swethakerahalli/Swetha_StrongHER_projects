"""PHY agents that complete the 6G channel-estimation stack (TS 38.211 / 38.214 / TR 38.843)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent

PHY = ["snr_db", "nmse_ai", "pilot_overhead", "doppler_hz", "cqi", "n_tx", "delay_spread_ns"]


def _splits(data: pd.DataFrame):
    return (data[data["split"] == s] for s in ("train", "validation", "test"))


class PilotAgent(BaseAgent):
    """DMRS/CSI-RS pilot density: increase when NMSE is poor, reduce when CSI prediction is confident."""

    def __init__(self, model_path=None):
        super().__init__("pilot", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        tr, va, te = _splits(data)
        y = (tr["nmse_ai"] > 0.08).astype(int)
        self.model = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42)
        self.model.fit(tr[PHY].fillna(0), y)
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[PHY].fillna(0))
            lab = (part["nmse_ai"] > 0.08).astype(int)
            metrics[f"{name}_accuracy"] = round(float(accuracy_score(lab, pred)), 4)
            metrics[f"{name}_mean_pilot_overhead"] = round(float(part["pilot_overhead"].mean()), 4)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        need_more = bool(self.model.predict(self._to_array(f, PHY))[0]) if self.is_trained else f.get("nmse_ai", 0) > 0.08
        density = "dense" if need_more else "sparse"
        return AgentAction(
            self.agent_id,
            "pilot_control",
            {
                "pilot_density": density,
                "intent": "increase_pilots" if need_more else "reduce_pilots",
                "dmrs_ports": 4 if need_more else 2,
            },
            confidence=0.9,
        )


class EqualizerAgent(BaseAgent):
    """MMSE/ZF equalizer selection from estimated CSI quality."""

    def __init__(self, model_path=None):
        super().__init__("equalizer", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        tr, va, te = _splits(data)
        self.model = GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=42)
        self.model.fit(tr[PHY].fillna(0), tr["ber_ai"])
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[PHY].fillna(0))
            metrics[f"{name}_r2"] = round(float(r2_score(part["ber_ai"], pred)), 4)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        ber_hat = float(self.model.predict(self._to_array(f, PHY))[0]) if self.is_trained else 1e-3
        mode = "mmse" if f.get("nmse_ai", 0.05) < 0.12 else "regularized_mmse"
        return AgentAction(self.agent_id, "equalize", {"mode": mode, "predicted_ber": round(ber_hat, 6)}, 0.88)


class AirInterfaceAgent(BaseAgent):
    """Reference-signal configuration: DMRS, CSI-RS, SRS (TS 38.211 / 38.214)."""

    def __init__(self, model_path=None):
        super().__init__("air_interface", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        tr, va, te = _splits(data)
        y = np.clip((tr["cqi"] / 5).astype(int), 0, 3)
        self.model = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42)
        self.model.fit(tr[PHY].fillna(0), y)
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[PHY].fillna(0))
            lab = np.clip((part["cqi"] / 5).astype(int), 0, 3)
            metrics[f"{name}_accuracy"] = round(float(accuracy_score(lab, pred)), 4)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        cfg = int(self.model.predict(self._to_array(f, PHY))[0]) if self.is_trained else 1
        period_ms = [80, 40, 20, 10][cfg]
        return AgentAction(
            self.agent_id,
            "air_interface",
            {"csi_rs_period_ms": period_ms, "srs_enabled": f.get("velocity_mps", 0) > 8, "ptrs_on": f.get("fc_ghz", 3) > 24},
            0.87,
        )


class CsiFeedbackAgent(BaseAgent):
    """CSI report periodicity / compression (TR 38.843 CSI prediction and overhead reduction)."""

    def __init__(self, model_path=None):
        super().__init__("csi_feedback", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        tr, va, te = _splits(data)
        self.model = GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=42)
        self.model.fit(tr[PHY].fillna(0), tr["csi_pred_accuracy"])
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[PHY].fillna(0))
            metrics[f"{name}_r2"] = round(float(r2_score(part["csi_pred_accuracy"], pred)), 4)
            metrics[f"{name}_mean_acc"] = round(float(np.mean(pred)), 4)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        acc = float(self.model.predict(self._to_array(f, PHY))[0]) if self.is_trained else 0.9
        return AgentAction(
            self.agent_id,
            "csi_feedback",
            {"report_period_ms": 40 if acc > 0.93 else 10, "compress": acc > 0.9, "predicted_accuracy": round(acc, 4)},
            min(0.99, acc),
        )


class SpectrumAgent(BaseAgent):
    """Carrier / frequency-hop recommendation under jamming and THz blockage."""

    def __init__(self, model_path=None):
        super().__init__("spectrum", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        tr, va, te = _splits(data)
        y = ((tr["is_attack"] == 1) | (tr["snr_db"] < 2)).astype(int)
        self.model = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42)
        self.feature_keys = PHY + ["is_attack"]
        self.model.fit(tr[self.feature_keys].fillna(0), y)
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[self.feature_keys].fillna(0))
            lab = ((part["is_attack"] == 1) | (part["snr_db"] < 2)).astype(int)
            metrics[f"{name}_accuracy"] = round(float(accuracy_score(lab, pred)), 4)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        hop = bool(self.model.predict(self._to_array(f, getattr(self, "feature_keys", PHY + ["is_attack"])))[0]) if self.is_trained else False
        return AgentAction(self.agent_id, "spectrum", {"frequency_hop": hop, "intent": "frequency_hop" if hop else "hold_carrier"}, 0.86)


class SelfHealingAgent(BaseAgent):
    """Recovery after CSI attacks: fallback estimator and model rollback."""

    def __init__(self, model_path=None):
        super().__init__("self_healing", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        tr, va, te = _splits(data)
        y = tr["is_attack"]
        feats = ["nmse_ai", "anomaly_score", "trust_score", "snr_db"]
        self.feature_keys = feats
        self.model = GradientBoostingClassifier(n_estimators=40, max_depth=3, random_state=42)
        self.model.fit(tr[feats].fillna(0), y)
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[feats].fillna(0))
            metrics[f"{name}_accuracy"] = round(float(accuracy_score(part["is_attack"], pred)), 4)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        heal = bool(self.model.predict(self._to_array(f, self.feature_keys))[0]) if self.is_trained else f.get("is_attack", 0) == 1
        return AgentAction(
            self.agent_id,
            "self_heal",
            {"fallback_estimator": "mmse" if heal else "ai_ensemble", "rollback_model": heal, "active": heal},
            0.9,
        )


class ResourceAgent(BaseAgent):
    """PRB / pilot resource vs spectral efficiency."""

    def __init__(self, model_path=None):
        super().__init__("resource", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        tr, va, te = _splits(data)
        self.model = GradientBoostingRegressor(n_estimators=40, max_depth=3, random_state=42)
        self.model.fit(tr[PHY].fillna(0), tr["se_ai"])
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[PHY].fillna(0))
            metrics[f"{name}_r2"] = round(float(r2_score(part["se_ai"], pred)), 4)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        se = float(self.model.predict(self._to_array(f, PHY))[0]) if self.is_trained else 4.0
        return AgentAction(self.agent_id, "resource", {"predicted_se": round(se, 3), "prb_boost": se < 3.5}, 0.85)


class KnowledgeAgent(BaseAgent):
    """Maps radio context to 3GPP / Nokia procedure (DMRS, CSI-RS, CDL profile)."""

    def __init__(self, model_path=None):
        super().__init__("knowledge", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        te = data[data["split"] == "test"]
        self.metrics = {
            "test_profiles": int(te["channel_profile"].nunique()),
            "test_scenarios": int(te["scenario"].nunique()),
            "knowledge_sources": 6,
        }
        self.is_trained = True
        self.model = self.metrics
        return self.metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        refs = ["TR-38.901", "TS-38.211"]
        if f.get("fc_ghz", 3) > 24:
            refs.append("TS-38.101-2")
        if f.get("is_attack", 0):
            refs.append("TR-38.843")
        return AgentAction(self.agent_id, "knowledge", {"references": refs, "estimator_hint": "mmse_if_los_else_ai"}, 0.8)
