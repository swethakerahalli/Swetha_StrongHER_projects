"""Agent Optimizer — meta-agent that re-trains and re-tunes degraded RAN agents."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class AgentOptimizerAgent(BaseAgent):
    """Monitors peer agents and triggers optimization when performance degrades."""

    FEATURE_KEYS = [
        "performance_index", "improvement_pct", "confidence",
        "validation_score", "degradation_score",
    ]

    OPTIMIZATION_ACTIONS = [
        "retrain_model", "tune_hyperparams", "boost_confidence",
        "rebalance_weights", "fallback_policy",
    ]

    def __init__(self, model_path=None):
        super().__init__("agent_optimizer_agent", model_path)
        self.optimization_log: list[dict] = []

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        df = data.copy()
        for col in self.FEATURE_KEYS:
            if col not in df.columns:
                df[col] = 0.5
        df["needs_optimization"] = (
            (df["performance_index"] < 65)
            | (df["improvement_pct"] < 0)
            | (df["confidence"] < 0.72)
        ).astype(int)
        X = df[self.FEATURE_KEYS].values
        y = df["needs_optimization"].values
        self.model = GradientBoostingClassifier(n_estimators=30, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"accuracy": float(self.model.score(X, y)), "samples": len(df)}

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        target = observation.context.get("target_agent", "unknown")
        perf_idx = f.get("performance_index", 80)
        imp = f.get("improvement_pct", 10)
        conf = f.get("confidence", 0.85)
        deg = f.get("degradation_score", 0)

        if self.is_trained:
            prob = float(self.model.predict_proba(self._to_array(f, self.FEATURE_KEYS))[0][1])
        else:
            prob = 0.8 if perf_idx < 65 or imp < 0 or conf < 0.72 else 0.2

        if prob > 0.5:
            action = self._select_action(perf_idx, imp, conf)
            gain = round(min(25, prob * 20 + deg * 5), 1)
        else:
            action = "monitor_only"
            gain = 0

        entry = {
            "target_agent": target,
            "action": action,
            "degradation_score": round(deg, 2),
            "expected_recovery_pct": gain,
            "triggered": prob > 0.5,
        }
        self.optimization_log.append(entry)

        return AgentAction(
            agent_id=self.agent_id,
            action_type="agent_optimization",
            parameters={
                "target_agent": target,
                "optimization_action": action,
                "degradation_score": round(deg, 2),
                "expected_recovery_pct": gain,
                "retrain_epochs": 15 if action == "retrain_model" else 0,
                "hyperparam_tune": action in ("tune_hyperparams", "retrain_model"),
                "weight_adjustment": round(prob * 0.15, 3),
                "monitoring_interval_s": 30,
            },
            confidence=round(0.85 + prob * 0.1, 2),
        )

    @staticmethod
    def _select_action(perf_idx: float, improvement: float, confidence: float) -> str:
        if perf_idx < 45 or improvement < -5:
            return "retrain_model"
        if confidence < 0.65:
            return "boost_confidence"
        if improvement < 0:
            return "tune_hyperparams"
        if perf_idx < 65:
            return "rebalance_weights"
        return "fallback_policy"
