"""Network Slice Agent — isolation, PRB allocation, and SLA compliance."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent
from src.common.utils import load_config


class SliceAgent(BaseAgent):
    FEATURE_KEYS = [
        "prb_utilization", "active_ues", "sla_compliance",
        "throughput_mbps", "latency_p99_ms",
    ]
    SLICE_PRIORITY = {"URLLC": 5, "eMBB": 2, "mMTC": 1}

    def __init__(self, model_path=None):
        super().__init__("slice_agent", model_path)
        self.slice_cfg = load_config("system_config.json")["network_slices"]

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        data = data.copy()
        data["needs_rebalance"] = (
            (data["sla_compliance"] < 0.95) | (data["prb_utilization"] > 0.85)
        ).astype(int)
        X = data[self.FEATURE_KEYS].values
        y = data["needs_rebalance"].values
        self.model = GradientBoostingClassifier(n_estimators=40, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"accuracy": float(self.model.score(X, y)), "samples": len(data)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        sl = observation.context.get("slice", "eMBB")
        cfg = self.slice_cfg.get(sl, self.slice_cfg["eMBB"])
        sla = f.get("sla_compliance", 0.98)
        prb = f.get("prb_utilization", 0.6)
        lat_p99 = f.get("latency_p99_ms", 10)

        if self.is_trained:
            x = self._to_array(f, self.FEATURE_KEYS)
            rebalance_prob = float(self.model.predict_proba(x)[0][1])
        else:
            rebalance_prob = 0.75 if sla < 0.95 or prb > 0.85 else 0.2

        if rebalance_prob > 0.5:
            slice_action = "rebalance" if prb > 0.75 else "boost_priority"
            isolation = "strict" if sl == "URLLC" else "shared"
        else:
            slice_action = "maintain"
            isolation = "shared"

        prb_share = min(0.5, prb * cfg.get("priority", 2) / 10)
        if sl == "URLLC" and lat_p99 > cfg.get("latency_budget_ms", 1) * 2:
            slice_action = "boost_priority"
            prb_share = min(0.5, prb_share + 0.15)

        return AgentAction(
            agent_id=self.agent_id,
            action_type="slice",
            parameters={
                "slice": sl,
                "slice_action": slice_action,
                "prb_share_pct": round(prb_share, 3),
                "isolation_level": isolation,
                "sla_compliance": round(sla, 4),
                "sla_target": 0.99,
                "priority": self.SLICE_PRIORITY.get(sl, 2),
            },
            confidence=0.91,
        )
