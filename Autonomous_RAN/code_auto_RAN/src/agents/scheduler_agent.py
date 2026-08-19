"""Scheduler Agent - PRB allocation and QoS-aware scheduling."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.agents.ai_predictor import ml_predict, scale_to_range
from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class SchedulerAgent(BaseAgent):
    FEATURE_KEYS = ["cqi", "sinr_db", "buffer_occupancy", "latency_ms", "mcs", "prb_allocated"]

    def __init__(self, model_path=None):
        super().__init__("scheduler_agent", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        X = data[self.FEATURE_KEYS].values
        y = data["throughput_mbps"].values
        self.model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        score = float(self.model.score(X, y))
        return {"r2_score": score, "samples": len(data)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        predicted_tp = ml_predict(self.model, f, self.FEATURE_KEYS) if self.is_trained else f.get("cqi", 5) * 5.0

        priority = scale_to_range(predicted_tp / 50, 0, 1)
        prb = int(scale_to_range(predicted_tp / 4, 1, 25))
        mcs = int(scale_to_range(predicted_tp / 3, 0, 28))

        return AgentAction(
            agent_id=self.agent_id,
            action_type="schedule",
            parameters={
                "prb_assignment": prb,
                "scheduling_priority": round(priority, 3),
                "mcs_recommendation": mcs,
                "predicted_throughput_mbps": round(predicted_tp, 2),
            },
            confidence=min(1.0, priority + 0.2),
        )

    def heuristic_schedule(self, ue_metrics: list[dict]) -> list[dict]:
        """Proportional-fair inspired scheduling for simulation."""
        scores = []
        for m in ue_metrics:
            cqi = m.get("cqi", 1)
            buf = m.get("buffer", 0.1)
            avg_tp = max(m.get("avg_throughput", 0.1), 0.1)
            score = (cqi / 15) * buf / avg_tp
            scores.append(score)
        total = sum(scores) or 1.0
        allocations = []
        for m, s in zip(ue_metrics, scores):
            prb = max(1, int(100 * s / total))
            allocations.append({**m, "prb": prb, "priority": s / total})
        return allocations
