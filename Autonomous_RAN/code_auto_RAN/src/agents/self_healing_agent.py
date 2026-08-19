"""Self-Healing Agent — AI-driven anomaly recovery and network resilience."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.agents.ai_predictor import ml_classify_proba
from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class SelfHealingAgent(BaseAgent):
    FEATURE_KEYS = [
        "packet_rate_pps", "auth_failures", "spectrum_anomaly_score",
        "flow_entropy", "cell_utilization",
    ]

    RECOVERY_ACTIONS = {
        0: "monitor",
        1: "reroute_traffic",
        2: "isolate_cell",
        3: "restore_defaults",
        4: "federated_retrain",
    }

    def __init__(self, model_path=None):
        super().__init__("self_healing_agent", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        df = data.copy()
        if "cell_utilization" not in df.columns:
            df["cell_utilization"] = 0.5
        df["needs_healing"] = (
            (df.get("is_attack", 0) == 1)
            | (df["spectrum_anomaly_score"] > 0.7)
            | (df["auth_failures"] > 2)
        ).astype(int)
        X = df[self.FEATURE_KEYS].fillna(0).values
        y = df["needs_healing"].values
        self.model = GradientBoostingClassifier(n_estimators=40, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"accuracy": float(self.model.score(X, y)), "samples": len(df)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        heal_prob = ml_classify_proba(self.model, f, self.FEATURE_KEYS) if self.is_trained else 0.2
        severity = min(4, int(heal_prob * 5))
        recovery = self.RECOVERY_ACTIONS.get(severity, "monitor")
        auto_heal = heal_prob > 0.55

        return AgentAction(
            agent_id=self.agent_id,
            action_type="self_healing",
            parameters={
                "healing_required": auto_heal,
                "recovery_action": recovery,
                "severity_level": severity,
                "healing_probability": round(heal_prob, 3),
                "estimated_recovery_time_s": round(30 * (1 - heal_prob), 1),
                "self_healing_enabled": True,
            },
            confidence=0.9,
        )
