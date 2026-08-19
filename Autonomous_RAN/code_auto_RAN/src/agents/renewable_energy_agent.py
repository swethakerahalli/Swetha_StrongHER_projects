"""Renewable Energy Agent — solar/wind routing and green power scheduling."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.agents.ai_predictor import ml_predict
from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class RenewableEnergyAgent(BaseAgent):
    FEATURE_KEYS = ["renewable_pct", "power_consumption_w", "cell_utilization", "traffic_demand_mbps"]

    def __init__(self, model_path=None):
        super().__init__("renewable_energy_agent", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        X = data[self.FEATURE_KEYS].values
        y = data["renewable_pct"].values
        self.model = GradientBoostingRegressor(n_estimators=35, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"r2_score": float(self.model.score(X, y)), "samples": len(data)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        current = f.get("renewable_pct", 15)

        if self.is_trained:
            target = min(90, max(current, ml_predict(self.model, f, self.FEATURE_KEYS) * 100))
        else:
            target = min(85, current + 25 + (20 if f.get("cell_utilization", 0.5) < 0.4 else 5))

        routing_pct = round(target, 1)
        carbon_reduction = round((routing_pct - current) * 0.4, 1)

        return AgentAction(
            agent_id=self.agent_id,
            action_type="renewable_energy",
            parameters={
                "renewable_routing_pct": routing_pct,
                "green_power_priority": True,
                "carbon_offset_pct": carbon_reduction,
                "battery_buffer_pct": round(min(80, routing_pct * 0.6), 1),
            },
            confidence=0.87,
        )
