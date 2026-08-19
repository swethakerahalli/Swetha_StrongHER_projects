"""Beamforming Agent — MRT/MMSE beam selection and weight optimization."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.agents.ai_predictor import ml_predict, scale_to_range
from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent
from src.simulation.phy_channel_sim import PHYChannelSimulator


class BeamformingAgent(BaseAgent):
    FEATURE_KEYS = ["sinr_db", "cqi", "rsrp_dbm", "cell_utilization"]

    def __init__(self, model_path=None):
        super().__init__("beamforming_agent", model_path)
        self.phy = PHYChannelSimulator()

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        cols = [c for c in self.FEATURE_KEYS if c in data.columns]
        X = data[cols].fillna(0).values
        y = data["sinr_db"].values if "sinr_db" in data.columns else data[cols[0]].values
        self.model = RandomForestRegressor(n_estimators=30, random_state=42)
        self._train_cols = cols
        self.model.fit(X, y)
        self.is_trained = True
        return {"r2_score": float(self.model.score(X, y)), "samples": len(data)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        cols = getattr(self, "_train_cols", self.FEATURE_KEYS)
        predicted_sinr = ml_predict(self.model, f, cols) if self.is_trained else f.get("sinr_db", 10)
        ch = self.phy.generate_csi("UE_BF", velocity=f.get("velocity_mps", 0))
        weights = self.phy.beamforming_weights(ch.csi)
        gain_db = float(10 * np.log10(np.abs(weights).sum() ** 2 + 1e-9)) + (predicted_sinr - f.get("sinr_db", 10)) * 0.1

        mode = "MRT" if predicted_sinr > 10 else "MMSE"
        num_beams = int(scale_to_range(predicted_sinr / 3, 2, 8))

        return AgentAction(
            agent_id=self.agent_id,
            action_type="beamforming",
            parameters={
                "beamforming_mode": mode,
                "num_beams": num_beams,
                "beamforming_gain_db": round(gain_db, 2),
                "beam_weights_norm": round(float(np.linalg.norm(weights)), 4),
            },
            confidence=0.86,
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
