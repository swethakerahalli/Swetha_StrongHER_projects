"""CSI Agent — CSI feedback compression and prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor

from src.agents.ai_predictor import ml_predict
from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent
from src.simulation.phy_channel_sim import PHYChannelSimulator


class CSIAgent(BaseAgent):
    FEATURE_KEYS = ["cqi", "sinr_db", "rsrp_dbm", "mcs"]

    def __init__(self, model_path=None):
        super().__init__("csi_agent", model_path)
        self.phy = PHYChannelSimulator()
        self.pca = None
        if model_path and Path(model_path).exists():
            self.load(model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        X = data[self.FEATURE_KEYS].fillna(0).values
        y = data["throughput_mbps"].values if "throughput_mbps" in data.columns else data["cqi"].values
        self.model = GradientBoostingRegressor(n_estimators=30, random_state=42)
        self.model.fit(X, y)
        self.pca = PCA(n_components=2)
        self.pca.fit(X)
        self.is_trained = True
        return {"r2_score": float(self.model.score(X, y)), "samples": len(data),
                "csi_compression_ratio": round(X.shape[1] / 2, 2)}

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "agent_id": self.agent_id, "pca": self.pca}, path)

    def load(self, path) -> None:
        payload = joblib.load(path)
        self.model = payload["model"]
        self.pca = payload.get("pca")
        self.is_trained = True

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        predicted_tp = ml_predict(self.model, f, self.FEATURE_KEYS) if self.is_trained else f.get("throughput_mbps", 10)
        history = [self.phy.generate_csi("UE_CSI", velocity=5).csi for _ in range(3)]
        pred_csi = self.phy.predict_future_csi(history)
        if self.pca and self.is_trained:
            compressed = self.pca.transform(self._to_array(f, self.FEATURE_KEYS))
            compression_ratio = len(self.FEATURE_KEYS) / max(compressed.shape[1], 1)
        else:
            compression_ratio = 2.0
        csi_accuracy = round(min(1.0, predicted_tp / 100), 3) if self.is_trained else round(f.get("cqi", 8) / 15, 3)

        return AgentAction(
            agent_id=self.agent_id,
            action_type="csi",
            parameters={
                "csi_prediction_available": True,
                "compression_ratio": round(compression_ratio, 2),
                "predicted_csi_power": round(float(np.abs(pred_csi).mean()), 4),
                "predicted_throughput_mbps": round(predicted_tp, 2),
                "feedback_bits_saved_pct": round((1 - 1 / compression_ratio) * 100, 1) if compression_ratio > 1 else 0,
                "csi_accuracy": csi_accuracy,
            },
            confidence=0.84 if self.is_trained else 0.7,
        )
