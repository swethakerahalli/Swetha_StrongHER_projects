"""Channel Estimation Agent — AI-native PHY channel estimation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from src.agents.ai_predictor import ml_predict
from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent
from src.simulation.phy_channel_sim import PHYChannelSimulator


class ChannelEstimationAgent(BaseAgent):
    FEATURE_KEYS = ["sinr_db", "cqi", "velocity_mps", "rsrp_dbm"]

    def __init__(self, model_path=None):
        super().__init__("channel_estimation_agent", model_path)
        self.phy = PHYChannelSimulator()

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        mob = data if "velocity_mps" in data.columns else None
        if mob is None:
            self.is_trained = True
            return {"mode": "phy_surrogate", "samples": 0}
        X = mob[self.FEATURE_KEYS[:3]].fillna(0).values if "velocity_mps" in mob else mob[["sinr_db", "cqi"]].values
        y = mob["rsrp_dbm"].values if "rsrp_dbm" in mob else mob["sinr_db"].values
        self.model = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"r2_score": float(self.model.score(X, y)), "samples": len(data)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        keys = [k for k in self.FEATURE_KEYS if k in f or True]
        if self.is_trained:
            predicted_rsrp = ml_predict(self.model, f, keys[:3] if len(keys) >= 3 else self.FEATURE_KEYS[:3])
            pilots = np.array([f.get("sinr_db", 10), f.get("cqi", 8), predicted_rsrp])
        else:
            pilots = np.array([f.get("sinr_db", 10), f.get("cqi", 8), f.get("rsrp_dbm", -95)])
        h_est = self.phy.neural_channel_estimate(pilots, interference=0.1)
        nmse = float(np.mean(np.abs(h_est - pilots) ** 2) / (np.mean(np.abs(pilots) ** 2) + 1e-9))
        csi_accuracy = round(max(0, 1 - nmse), 3)

        return AgentAction(
            agent_id=self.agent_id,
            action_type="channel_estimation",
            parameters={
                "estimated_channel_power": round(float(np.abs(h_est).mean()), 4),
                "nmse": round(nmse, 4),
                "estimation_method": "ai_mlp" if self.is_trained else "neural_ls",
                "csi_accuracy": csi_accuracy,
            },
            confidence=0.88 if self.is_trained else 0.75,
        )
