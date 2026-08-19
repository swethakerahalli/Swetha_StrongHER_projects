"""Resource Allocation Agent - spectrum, power, MIMO optimization."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.agents.ai_predictor import ml_predict, scale_to_range
from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class ResourceAgent(BaseAgent):
    FEATURE_KEYS = ["cell_utilization", "traffic_demand_mbps", "power_consumption_w", "renewable_pct"]

    def __init__(self, model_path=None):
        super().__init__("resource_agent", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        X = data[self.FEATURE_KEYS].values
        y = data["traffic_demand_mbps"].values / (data["power_consumption_w"].values + 1)
        self.model = RandomForestRegressor(n_estimators=40, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"r2_score": float(self.model.score(X, y)), "samples": len(data)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        efficiency = ml_predict(self.model, f, self.FEATURE_KEYS) if self.is_trained else 0.5
        power_dbm = scale_to_range(46 - efficiency * 20, 10, 46)
        bandwidth_mhz = int(scale_to_range(efficiency * 100, 20, 100))
        mimo_streams = int(scale_to_range(efficiency * 8, 1, 8))
        ca_enabled = efficiency > 0.6

        return AgentAction(
            agent_id=self.agent_id,
            action_type="resource_allocation",
            parameters={
                "power_dbm": round(power_dbm, 1),
                "bandwidth_mhz": bandwidth_mhz,
                "mimo_streams": mimo_streams,
                "carrier_aggregation": ca_enabled,
                "spectral_efficiency": round(efficiency, 4),
            },
            confidence=0.85,
        )
