"""Energy Agent - power optimization and sleep mode control."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class EnergyAgent(BaseAgent):
    FEATURE_KEYS = ["power_consumption_w", "cell_utilization", "traffic_demand_mbps", "renewable_pct"]

    def __init__(self, model_path=None):
        super().__init__("energy_agent", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        X = data[self.FEATURE_KEYS].values
        y = data["sleep_state"].values
        self.model = GradientBoostingClassifier(n_estimators=30, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"accuracy": float(self.model.score(X, y)), "samples": len(data)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        util = f.get("cell_utilization", 0.5)
        renewable = f.get("renewable_pct", 0)

        if self.is_trained:
            x = self._to_array(f, self.FEATURE_KEYS)
            sleep_prob = float(self.model.predict_proba(x)[0][1])
        else:
            sleep_prob = 0.7 if util < 0.15 else 0.1

        enable_sleep = sleep_prob > 0.5
        power_scale = 0.1 if enable_sleep else max(0.3, util)
        carbon_aware = renewable > 20

        return AgentAction(
            agent_id=self.agent_id,
            action_type="energy",
            parameters={
                "sleep_mode": enable_sleep,
                "power_scale_factor": round(power_scale, 2),
                "carbon_aware_mode": carbon_aware,
                "estimated_savings_pct": round((1 - power_scale) * 100, 1) if enable_sleep else 0,
            },
            confidence=0.88,
        )
