"""RAN Sleep Agent — AI-driven cell sleep / wake and TX path switching."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class RANSleepAgent(BaseAgent):
    FEATURE_KEYS = ["cell_utilization", "traffic_demand_mbps", "power_consumption_w", "sleep_state"]

    def __init__(self, model_path=None):
        super().__init__("ran_sleep_agent", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        X = data[self.FEATURE_KEYS].values
        y = data["sleep_state"].values
        self.model = GradientBoostingClassifier(n_estimators=35, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"accuracy": float(self.model.score(X, y)), "samples": len(data)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        util = f.get("cell_utilization", 0.5)
        traffic = f.get("traffic_demand_mbps", 50)

        if self.is_trained:
            sleep_prob = float(self.model.predict_proba(self._to_array(f, self.FEATURE_KEYS))[0][1])
        else:
            sleep_prob = 0.8 if util < 0.12 and traffic < 20 else 0.15

        enable_sleep = sleep_prob > 0.55
        power_saving_pct = round((1 - (0.12 if enable_sleep else max(0.5, util))) * 100, 1)

        return AgentAction(
            agent_id=self.agent_id,
            action_type="ran_sleep",
            parameters={
                "sleep_mode": enable_sleep,
                "cells_to_sleep": 2 if enable_sleep else 0,
                "tx_path_switch": "minimal" if enable_sleep else "active",
                "power_saving_pct": power_saving_pct,
                "wake_threshold_util": 0.18,
            },
            confidence=0.89,
        )
