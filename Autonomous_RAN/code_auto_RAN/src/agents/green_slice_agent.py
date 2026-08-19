"""Green Slicing Agent — energy-aware network slice orchestration."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent
from src.common.utils import load_config


class GreenSliceAgent(BaseAgent):
    FEATURE_KEYS = [
        "prb_utilization", "sla_compliance", "throughput_mbps",
        "power_consumption_w", "renewable_pct",
    ]

    def __init__(self, model_path=None):
        super().__init__("green_slice_agent", model_path)
        self.slice_cfg = load_config("system_config.json")["network_slices"]

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        df = data.copy()
        if "power_consumption_w" not in df.columns:
            df["power_consumption_w"] = 400.0
        if "renewable_pct" not in df.columns:
            df["renewable_pct"] = 20.0
        df["green_rebalance"] = ((df["prb_utilization"] > 0.7) | (df["sla_compliance"] < 0.96)).astype(int)
        X = df[self.FEATURE_KEYS].values
        y = df["green_rebalance"].values
        self.model = GradientBoostingClassifier(n_estimators=35, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"accuracy": float(self.model.score(X, y)), "samples": len(df)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        sl = observation.context.get("slice", "eMBB")
        prb = f.get("prb_utilization", f.get("cell_utilization", 0.6))
        renewable = f.get("renewable_pct", 15)
        power = f.get("power_consumption_w", 400)

        if self.is_trained:
            green_prob = float(self.model.predict_proba(self._to_array(f, self.FEATURE_KEYS))[0][1])
        else:
            green_prob = 0.7 if power > 350 and renewable < 40 else 0.35

        energy_saving = round(green_prob * 22, 1) if green_prob > 0.5 else 5.0
        slice_eff_gain = round(green_prob * 18, 1)

        return AgentAction(
            agent_id=self.agent_id,
            action_type="green_slice",
            parameters={
                "slice": sl,
                "green_slice_action": "eco_rebalance" if green_prob > 0.5 else "maintain",
                "energy_saving_pct": energy_saving,
                "slice_efficiency_gain_pct": slice_eff_gain,
                "renewable_affinity": round(min(1.0, renewable / 100 + 0.3), 2),
                "power_cap_w": round(power * (1 - energy_saving / 100), 1),
            },
            confidence=0.88,
        )
