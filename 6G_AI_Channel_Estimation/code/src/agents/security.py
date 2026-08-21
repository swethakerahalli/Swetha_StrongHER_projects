"""Security detection and mitigation agents."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent

SEC_FEATURES = [
    "snr_db", "anomaly_score", "pilot_correlation", "csi_consistency",
    "trust_score", "nmse_ls", "nmse_ai", "attack_severity",
]


def _cls_metrics(y_true, y_pred, y_proba, prefix: str) -> dict[str, float]:
    out = {
        f"{prefix}_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        f"{prefix}_precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        f"{prefix}_recall": round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        f"{prefix}_f1": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            out[f"{prefix}_roc_auc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
        except Exception:
            out[f"{prefix}_roc_auc"] = 0.0
    return out


class SecurityAgent(BaseAgent):
    def __init__(self, model_path=None):
        super().__init__("security", model_path)
        self.iforest = None
        self.classes_ = None

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        tr, va, te = (data[data["split"] == s] for s in ("train", "validation", "test"))
        self.model = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=42, n_jobs=-1)
        self.iforest = IsolationForest(contamination=0.2, random_state=42)
        X_tr = tr[SEC_FEATURES].fillna(0)
        y_tr = tr["attack_type"]
        self.model.fit(X_tr, y_tr)
        self.iforest.fit(X_tr[tr["is_attack"] == 0] if (tr["is_attack"] == 0).any() else X_tr)
        self.classes_ = list(self.model.classes_)
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            X = part[SEC_FEATURES].fillna(0)
            pred = self.model.predict(X)
            proba = self.model.predict_proba(X)
            metrics.update(_cls_metrics(part["attack_type"], pred, None, f"multiclass_{name}"))
            bin_true = part["is_attack"]
            bin_pred = (pred != "normal").astype(int)
            # binary proba = 1 - P(normal) if present
            if "normal" in self.classes_:
                idx = self.classes_.index("normal")
                bin_score = 1.0 - proba[:, idx]
                y_bin_proba = np.vstack([1 - bin_score, bin_score]).T
            else:
                y_bin_proba = None
            metrics.update(_cls_metrics(bin_true, bin_pred, y_bin_proba, f"binary_{name}"))
        metrics["samples_train"] = int(len(tr))
        metrics["samples_validation"] = int(len(va))
        metrics["samples_test"] = int(len(te))
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        X = self._to_array(observation.features, SEC_FEATURES)
        if self.is_trained:
            label = str(self.model.predict(X)[0])
            proba = self.model.predict_proba(X)[0]
            conf = float(np.max(proba))
        else:
            label = "normal" if observation.features.get("anomaly_score", 0) < 0.4 else "jamming"
            conf = 0.6
        return AgentAction(
            self.agent_id,
            "attack_detection",
            {"attack_type": label, "is_attack": int(label != "normal"), "p_attack": round(conf if label != "normal" else 1 - conf, 4)},
            confidence=conf,
        )


class MitigationAgent(BaseAgent):
    POLICY = {
        "pilot_contamination": "pilot_reassignment",
        "jamming": "frequency_hop",
        "csi_spoofing": "beam_switch",
        "false_csi_injection": "trust_scheduling",
        "data_poisoning": "federated_retrain",
        "adversarial": "beam_switch",
        "backdoor": "federated_retrain",
        "normal": "none",
    }

    def __init__(self, model_path=None):
        super().__init__("mitigation", model_path)

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        tr, va, te = (data[data["split"] == s] for s in ("train", "validation", "test"))
        self.model = GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=42)
        feats = [c for c in SEC_FEATURES if c in tr.columns]
        self.feature_keys = feats
        self.model.fit(tr[feats].fillna(0), tr["mitigation_action"])
        metrics = {}
        for name, part in (("train", tr), ("validation", va), ("test", te)):
            pred = self.model.predict(part[feats].fillna(0))
            metrics[f"{name}_accuracy"] = round(float(accuracy_score(part["mitigation_action"], pred)), 4)
            if "mitigation_success" in part.columns:
                metrics[f"{name}_success_rate"] = round(float(part["mitigation_success"].mean()), 4)
        self.metrics = metrics
        self.is_trained = True
        return metrics

    def predict(self, observation: AgentObservation) -> AgentAction:
        attack = str(observation.context.get("attack_type", "normal"))
        action = self.POLICY.get(attack, "none")
        return AgentAction(self.agent_id, "mitigation", {"action": action, "attack_type": attack}, confidence=0.9)
