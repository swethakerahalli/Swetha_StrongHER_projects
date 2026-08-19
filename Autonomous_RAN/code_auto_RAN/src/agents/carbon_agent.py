"""Carbon Emission Reduction Agent — green scheduling and carbon-aware RAN optimization."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.agents.ai_predictor import ml_predict, scale_to_range
from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class CarbonAgent(BaseAgent):
    FEATURE_KEYS = [
        "power_consumption_w", "cell_utilization", "renewable_pct",
        "carbon_intensity_gco2_kwh", "traffic_demand_mbps",
    ]

    def __init__(self, model_path=None):
        super().__init__("carbon_agent", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        df = data.copy()
        if "carbon_intensity_gco2_kwh" not in df.columns:
            df["carbon_intensity_gco2_kwh"] = 350.0
        X = df[self.FEATURE_KEYS].values
        y = (df["power_consumption_w"] * df["carbon_intensity_gco2_kwh"]).values / 1e6
        self.model = GradientBoostingRegressor(n_estimators=40, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"r2_score": float(self.model.score(X, y)), "samples": len(df)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        power = f.get("power_consumption_w", 400)
        renewable = f.get("renewable_pct", 15)
        intensity = f.get("carbon_intensity_gco2_kwh", 380)
        util = f.get("cell_utilization", 0.5)

        if self.is_trained:
            pred_emission = ml_predict(self.model, f, self.FEATURE_KEYS)
            reduction = max(0.1, min(0.65, 1 - pred_emission / max(power * intensity / 1e6, 0.01)))
        else:
            reduction = min(0.55, 0.15 + renewable / 200 + (0.3 if util < 0.4 else 0.1))

        new_intensity = round(intensity * (1 - reduction * 0.45), 1)
        renewable_target = min(80, renewable + reduction * 40)
        carbon_kg_h = round(power * new_intensity / 1e6, 4)

        return AgentAction(
            agent_id=self.agent_id,
            action_type="carbon_reduction",
            parameters={
                "carbon_intensity_gco2_kwh": new_intensity,
                "renewable_routing_pct": round(renewable_target, 1),
                "carbon_reduction_pct": round(reduction * 100, 1),
                "green_scheduling_enabled": True,
                "carbon_kg_co2_per_h": carbon_kg_h,
                "estimated_co2_savings_kg_h": round(power * (intensity - new_intensity) / 1e6, 4),
            },
            confidence=0.91,
        )
