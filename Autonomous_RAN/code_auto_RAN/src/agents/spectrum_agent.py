"""Spectrum Agent — AI-driven spectrum allocation and interference mitigation."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.agents.ai_predictor import ml_classify_proba, scale_to_range
from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class SpectrumAgent(BaseAgent):
    FEATURE_KEYS = [
        "spectrum_anomaly_score", "cell_utilization", "interference_db",
        "traffic_demand_mbps", "sinr_db",
    ]

    def __init__(self, model_path=None):
        super().__init__("spectrum_agent", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        df = data.copy()
        if "interference_db" not in df.columns:
            df["interference_db"] = df.get("spectrum_anomaly_score", pd.Series(0.1, index=df.index)) * 10
        if "spectrum_anomaly_score" not in df.columns:
            df["spectrum_anomaly_score"] = 0.1
        df["needs_reallocation"] = (
            (df["spectrum_anomaly_score"] > 0.5) | (df.get("cell_utilization", 0.5) > 0.8)
        ).astype(int)
        cols = [c for c in self.FEATURE_KEYS if c in df.columns]
        X = df[cols].fillna(0).values
        y = df["needs_reallocation"].values
        self.model = GradientBoostingClassifier(n_estimators=40, random_state=42)
        self._train_cols = cols
        self.model.fit(X, y)
        self.is_trained = True
        return {"accuracy": float(self.model.score(X, y)), "samples": len(df)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        realloc_prob = ml_classify_proba(self.model, f, getattr(self, "_train_cols", self.FEATURE_KEYS)) if self.is_trained else 0.3
        bandwidth_mhz = scale_to_range(f.get("traffic_demand_mbps", 50) / 10, 20, 100)
        carrier = "n78" if f.get("sinr_db", 10) > 5 else "n41"
        spectrum_action = "reallocate" if realloc_prob > 0.5 else "maintain"

        return AgentAction(
            agent_id=self.agent_id,
            action_type="spectrum",
            parameters={
                "spectrum_action": spectrum_action,
                "bandwidth_mhz": round(bandwidth_mhz, 1),
                "carrier_band": carrier,
                "interference_mitigation": round(1 - f.get("spectrum_anomaly_score", 0.1), 3),
                "reallocation_probability": round(realloc_prob, 3),
                "dynamic_spectrum_sharing": realloc_prob > 0.4,
            },
            confidence=0.89,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "agent_id": self.agent_id,
                     "train_cols": getattr(self, "_train_cols", self.FEATURE_KEYS)}, path)

    def load(self, path: Path) -> None:
        payload = joblib.load(path)
        self.model = payload["model"]
        self._train_cols = payload.get("train_cols", self.FEATURE_KEYS)
        self.is_trained = True
