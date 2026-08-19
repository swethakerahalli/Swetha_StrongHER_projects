"""Mobility Agent - handover prediction and cell selection."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class MobilityAgent(BaseAgent):
    FEATURE_KEYS = ["velocity_mps", "rsrp_dbm", "handover_pending", "direction_deg"]

    def __init__(self, model_path=None):
        super().__init__("mobility_agent", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        X = data[self.FEATURE_KEYS].values
        y = data["handover_pending"].values
        self.model = GradientBoostingClassifier(n_estimators=40, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"accuracy": float(self.model.score(X, y)), "samples": len(data)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        velocity = f.get("velocity_mps", 0)
        rsrp = f.get("rsrp_dbm", -95)
        neighbors = observation.context.get("neighbor_cells", ["CELL_001"])

        if self.is_trained:
            x = self._to_array(f, self.FEATURE_KEYS)
            ho_prob = float(self.model.predict_proba(x)[0][1])
        else:
            ho_prob = 0.3 if velocity > 10 and rsrp < -100 else 0.05

        target = neighbors[0] if ho_prob > 0.5 else observation.context.get("current_cell", "CELL_000")
        recommend_ho = ho_prob > 0.5

        return AgentAction(
            agent_id=self.agent_id,
            action_type="mobility",
            parameters={
                "handover_recommended": recommend_ho,
                "handover_probability": round(ho_prob, 3),
                "target_cell": target,
                "mobility_state": "high" if velocity > 15 else ("medium" if velocity > 5 else "low"),
            },
            confidence=0.9 if recommend_ho else 0.7,
        )
