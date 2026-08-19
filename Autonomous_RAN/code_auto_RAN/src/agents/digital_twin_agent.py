"""Digital Twin Agent — AI-driven what-if validation and policy simulation."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.agents.ai_predictor import ml_predict, scale_to_range
from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class DigitalTwinAgent(BaseAgent):
    FEATURE_KEYS = [
        "avg_throughput_mbps", "avg_latency_ms", "total_power_w",
        "cell_utilization", "traffic_demand_mbps",
    ]

    def __init__(self, model_path=None):
        super().__init__("digital_twin_agent", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        df = data.copy()
        if "power_consumption_w" in df.columns:
            df["total_power_w"] = df["power_consumption_w"]
        if "throughput_mbps" in df.columns and "avg_throughput_mbps" not in df.columns:
            df["avg_throughput_mbps"] = df["throughput_mbps"]
        cols = [c for c in self.FEATURE_KEYS if c in df.columns]
        if len(cols) < 3:
            self.is_trained = True
            return {"mode": "twin_surrogate", "samples": 0}
        X = df[cols].fillna(0).values
        y = df["cell_utilization"].values if "cell_utilization" in df.columns else X[:, 0]
        self.model = GradientBoostingRegressor(n_estimators=40, random_state=42)
        self._train_cols = cols
        self.model.fit(X, y)
        self.is_trained = True
        return {"r2_score": float(self.model.score(X, y)), "samples": len(df)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        if self.is_trained and self.model is not None:
            cols = getattr(self, "_train_cols", self.FEATURE_KEYS)
            fidelity = scale_to_range(ml_predict(self.model, f, cols, default=0.5), 0.7, 0.99)
        else:
            fidelity = 0.95
        policy_safe = fidelity > 0.85
        predicted_tp = f.get("avg_throughput_mbps", 20) * (1 + fidelity * 0.1)
        predicted_lat = f.get("avg_latency_ms", 5) * (1 - fidelity * 0.05)

        return AgentAction(
            agent_id=self.agent_id,
            action_type="digital_twin",
            parameters={
                "fidelity_score": round(fidelity, 3),
                "policy_validation": "approved" if policy_safe else "review",
                "predicted_throughput_mbps": round(predicted_tp, 2),
                "predicted_latency_ms": round(predicted_lat, 2),
                "what_if_recommended": policy_safe,
                "twin_action": "deploy_policy" if policy_safe else "simulate_only",
            },
            confidence=round(fidelity, 2),
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
