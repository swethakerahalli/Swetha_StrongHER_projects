"""Edge Inference Agent — low-latency AI inference at network edge for RAN control."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.agents.ai_predictor import ml_predict, scale_to_range
from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class EdgeInferenceAgent(BaseAgent):
    FEATURE_KEYS = ["latency_ms", "throughput_mbps", "cell_utilization", "packet_rate_pps"]

    def __init__(self, model_path=None):
        super().__init__("edge_inference_agent", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        df = data.copy()
        if "packet_rate_pps" not in df.columns:
            df["packet_rate_pps"] = 200.0
        X = df[self.FEATURE_KEYS].values
        y = df["latency_ms"].values
        self.model = RandomForestRegressor(n_estimators=40, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"r2_score": float(self.model.score(X, y)), "samples": len(df)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        lat = f.get("latency_ms", 5)

        if self.is_trained:
            pred_lat = ml_predict(self.model, f, self.FEATURE_KEYS)
            edge_lat = max(0.3, pred_lat * 0.45)
        else:
            edge_lat = max(0.3, lat * 0.55)

        latency_reduction = round((1 - edge_lat / max(lat, 0.5)) * 100, 1)
        offload_pct = scale_to_range(1 - edge_lat / max(lat, 1), 0.4, 0.95)

        return AgentAction(
            agent_id=self.agent_id,
            action_type="edge_inference",
            parameters={
                "edge_latency_ms": round(edge_lat, 2),
                "inference_offload_pct": round(offload_pct, 2),
                "latency_reduction_pct": latency_reduction,
                "edge_node_id": "EDGE_MEC_01",
                "model_cache_hit_rate": 0.92,
            },
            confidence=0.9,
        )
