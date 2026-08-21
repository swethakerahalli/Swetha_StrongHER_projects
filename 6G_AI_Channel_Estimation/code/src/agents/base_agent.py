"""Base agent contract for the 6G channel intelligence platform."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass
class AgentAction:
    agent_id: str
    action_type: str
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class AgentObservation:
    timestamp: int
    features: dict[str, float]
    context: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    def __init__(self, agent_id: str, model_path: Path | None = None):
        self.agent_id = agent_id
        self.model = None
        self.is_trained = False
        self.metrics: dict[str, float] = {}
        self.model_path = model_path
        if model_path and model_path.exists():
            self.load(model_path)

    @abstractmethod
    def train(self, data: Any) -> dict[str, float]:
        pass

    @abstractmethod
    def predict(self, observation: AgentObservation) -> AgentAction:
        pass

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "agent_id": self.agent_id,
                "metrics": self.metrics,
                "feature_keys": getattr(self, "feature_keys", None),
            },
            path,
        )

    def load(self, path: Path) -> None:
        payload = joblib.load(path)
        self.model = payload["model"]
        self.metrics = payload.get("metrics", {})
        keys = payload.get("feature_keys")
        if keys is not None:
            self.feature_keys = keys
        self.is_trained = True

    @staticmethod
    def _to_array(features: dict[str, float], keys: list[str]) -> np.ndarray:
        return np.array([[float(features.get(k, 0.0)) for k in keys]], dtype=float)
