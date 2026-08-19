"""QoS Agent — SLA compliance and slice-aware QoS enforcement."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent
from src.common.utils import load_config


class QoSAgent(BaseAgent):
    FEATURE_KEYS = ["latency_ms", "throughput_mbps", "packet_loss", "prb_allocated"]

    SLA = {
        "URLLC": {"latency_ms": 1, "throughput_mbps": 10},
        "eMBB": {"latency_ms": 10, "throughput_mbps": 100},
        "mMTC": {"latency_ms": 100, "throughput_mbps": 1},
    }

    def __init__(self, model_path=None):
        super().__init__("qos_agent", model_path)
        self.slice_cfg = load_config("system_config.json")["network_slices"]

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        data = data.copy()
        data["sla_violation"] = data.apply(self._sla_violation_row, axis=1).astype(int)
        X = data[self.FEATURE_KEYS].values
        y = data["sla_violation"].values
        self.model = GradientBoostingClassifier(n_estimators=40, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"accuracy": float(self.model.score(X, y)), "samples": len(data)}

    def _sla_violation_row(self, row) -> bool:
        sl = row.get("slice", "eMBB")
        sla = self.SLA.get(sl, self.SLA["eMBB"])
        return row["latency_ms"] > sla["latency_ms"] or row["throughput_mbps"] < sla["throughput_mbps"] * 0.5

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        sl = observation.context.get("slice", "eMBB")
        sla = self.SLA.get(sl, self.SLA["eMBB"])
        lat, tp = f.get("latency_ms", 5), f.get("throughput_mbps", 10)

        if self.is_trained:
            x = self._to_array(f, self.FEATURE_KEYS)
            viol_prob = float(self.model.predict_proba(x)[0][1])
        else:
            viol_prob = 0.8 if lat > sla["latency_ms"] else 0.1

        priority_boost = 2 if sl == "URLLC" else (1 if sl == "eMBB" else 0)
        action = "boost_priority" if viol_prob > 0.5 else "maintain"

        return AgentAction(
            agent_id=self.agent_id,
            action_type="qos",
            parameters={
                "slice": sl,
                "sla_violation_prob": round(viol_prob, 3),
                "qos_action": action,
                "priority_boost": priority_boost,
                "latency_budget_ms": sla["latency_ms"],
                "min_throughput_mbps": sla["throughput_mbps"],
            },
            confidence=0.9,
        )
