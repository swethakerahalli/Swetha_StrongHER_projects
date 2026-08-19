"""Security Agent - threat detection, prediction, and mitigation."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class SecurityAgent(BaseAgent):
    FEATURE_KEYS = ["packet_rate_pps", "auth_failures", "spectrum_anomaly_score", "flow_entropy"]

    def __init__(self, model_path=None):
        super().__init__("security_agent", model_path)
        self.detector = None
        self.classifier = None

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        X = data[self.FEATURE_KEYS].values
        y = data["is_attack"].values

        self.detector = IsolationForest(contamination=0.1, random_state=42)
        self.detector.fit(X)

        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42)
        self.classifier.fit(X, y)

        self.model = {"detector": self.detector, "classifier": self.classifier}
        self.is_trained = True
        acc = float(self.classifier.score(X, y))
        return {"detection_accuracy": acc, "samples": len(data)}

    def load(self, path) -> None:
        super().load(path)
        if isinstance(self.model, dict):
            self.detector = self.model.get("detector")
            self.classifier = self.model.get("classifier")

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        x = self._to_array(f, self.FEATURE_KEYS)

        if self.is_trained and self.detector is not None and self.classifier is not None:
            anomaly = int(self.detector.predict(x)[0] == -1)
            attack_prob = float(self.classifier.predict_proba(x)[0][1])
            threat_type = self.classifier.classes_[
                int(self.classifier.predict(x)[0])
            ] if hasattr(self.classifier, "classes_") else "unknown"
        else:
            anomaly = int(f.get("spectrum_anomaly_score", 0) > 0.6)
            attack_prob = f.get("spectrum_anomaly_score", 0)
            threat_type = "suspected_attack" if anomaly else "normal"

        is_threat = anomaly or attack_prob > 0.5
        mitigation = "none"
        if is_threat:
            if attack_prob > 0.8:
                mitigation = "quarantine_node"
            elif attack_prob > 0.6:
                mitigation = "slice_protection"
            else:
                mitigation = "rate_limit"

        return AgentAction(
            agent_id=self.agent_id,
            action_type="security",
            parameters={
                "threat_detected": is_threat,
                "threat_score": round(attack_prob, 3),
                "anomaly_flag": bool(anomaly),
                "threat_type": threat_type,
                "mitigation_action": mitigation,
            },
            confidence=0.95 if is_threat else 0.8,
        )
