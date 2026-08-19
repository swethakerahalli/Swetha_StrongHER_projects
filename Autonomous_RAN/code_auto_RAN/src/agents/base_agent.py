"""Base agent interface for Autonomous RAN multi-agent system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
from pathlib import Path


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
        joblib.dump({"model": self.model, "agent_id": self.agent_id}, path)

    def load(self, path: Path) -> None:
        payload = joblib.load(path)
        self.model = payload["model"]
        self.is_trained = True

    @staticmethod
    def _to_array(features: dict[str, float], keys: list[str]) -> np.ndarray:
        return np.array([[features.get(k, 0.0) for k in keys]], dtype=float)
