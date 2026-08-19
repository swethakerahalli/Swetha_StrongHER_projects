"""Traffic Agent — AI-driven traffic prediction, load balancing, and congestion management."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.agents.ai_predictor import ml_predict, scale_to_range
from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class TrafficAgent(BaseAgent):
    FEATURE_KEYS = [
        "buffer_occupancy", "throughput_mbps", "latency_ms",
        "packet_loss", "prb_allocated",
    ]

    def __init__(self, model_path=None):
        super().__init__("traffic_agent", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        df = data.copy()
        if "cell_utilization" not in df.columns:
            df["cell_utilization"] = df.get("buffer_occupancy", 0.5)
        df["congestion_score"] = (
            df["buffer_occupancy"] * 0.5
            + df["packet_loss"] * 10
            + (df["latency_ms"] / 20).clip(0, 1) * 0.3
        )
        X = df[self.FEATURE_KEYS].values
        y = df["throughput_mbps"].values
        self.model = GradientBoostingRegressor(n_estimators=40, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"r2_score": float(self.model.score(X, y)), "samples": len(df)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        buffer = f.get("buffer_occupancy", 0.5)
        util = f.get("cell_utilization", f.get("prb_utilization", 0.6))
        tp = f.get("throughput_mbps", 30)
        lat = f.get("latency_ms", 5)

        if self.is_trained:
            pred_tp = ml_predict(self.model, f, self.FEATURE_KEYS)
            gain = max(0, (pred_tp / max(tp, 1) - 1) * 100)
        else:
            gain = 18.0 if buffer > 0.7 else 8.0

        congestion = util * 0.6 + buffer * 0.4
        reroute_pct = scale_to_range(congestion, 0.15, 0.55) if congestion > 0.55 else 0.1
        congestion_reduction = round(min(45, congestion * 50 + gain * 0.3), 1)
        peak_boost = round(gain + congestion_reduction * 0.4, 1)

        return AgentAction(
            agent_id=self.agent_id,
            action_type="traffic_optimization",
            parameters={
                "traffic_action": "load_balance" if congestion > 0.5 else "predictive_scale",
                "reroute_pct": round(reroute_pct, 2),
                "congestion_reduction_pct": congestion_reduction,
                "peak_throughput_boost_pct": peak_boost,
                "predicted_peak_mbps": round(tp * (1 + peak_boost / 100), 1),
                "target_cells": ["CELL_000", "CELL_001"] if congestion > 0.6 else [],
            },
            confidence=0.9,
        )
