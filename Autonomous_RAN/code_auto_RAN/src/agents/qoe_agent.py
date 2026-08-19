"""QoE Agent - user experience prediction and optimization."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class QoEAgent(BaseAgent):
    FEATURE_KEYS = ["throughput_mbps", "latency_ms", "packet_loss"]

    def __init__(self, model_path=None):
        super().__init__("qoe_agent", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        X = data[self.FEATURE_KEYS].values
        y = 5 - 0.02 * data["latency_ms"] - 10 * data["packet_loss"] + 0.01 * data["throughput_mbps"]
        y = y.clip(1, 5)
        self.model = GradientBoostingRegressor(n_estimators=40, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"r2_score": float(self.model.score(X, y)), "samples": len(data)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        tp = f.get("throughput_mbps", 10)
        lat = f.get("latency_ms", 10)
        loss = f.get("packet_loss", 0.01)

        if self.is_trained:
            x = self._to_array(f, self.FEATURE_KEYS)
            mos = float(self.model.predict(x)[0])
        else:
            mos = max(1, min(5, 5 - 0.02 * lat - 10 * loss + 0.01 * tp))

        action = "maintain"
        if mos < 3:
            action = "boost_priority"
        elif mos > 4.5:
            action = "optimize_energy"

        return AgentAction(
            agent_id=self.agent_id,
            action_type="qoe",
            parameters={
                "qoe_score": round(mos, 2),
                "mos_estimate": round(mos, 2),
                "optimization_action": action,
            },
            confidence=0.82,
        )
