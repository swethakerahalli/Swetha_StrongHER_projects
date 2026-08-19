"""Federated Learning framework (FedAvg) for privacy-preserving agent training."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


class FederatedLearningCoordinator:
    """Coordinate distributed training across edge nodes (simulated)."""

    def __init__(self, num_clients: int = 5, rounds: int = 10):
        self.num_clients = num_clients
        self.rounds = rounds
        self.global_model: BaseEstimator | None = None
        self.round_history: list[dict] = []

    def _split_data(self, X: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        n = len(X)
        indices = np.array_split(np.random.permutation(n), self.num_clients)
        return [(X[idx], y[idx]) for idx in indices if len(idx) > 0]

    def _get_model_params(self, model: BaseEstimator) -> dict[str, np.ndarray]:
        if hasattr(model, "coef_"):
            return {"coef": model.coef_.copy(), "intercept": model.intercept_.copy()}
        if hasattr(model, "estimators_"):
            return {"n_estimators": len(model.estimators_)}
        return {}

    def train_federated(
        self,
        model_factory,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, Any]:
        clients = self._split_data(X, y)
        local_models = []

        for rnd in range(self.rounds):
            round_models = []
            round_scores = []
            for X_c, y_c in clients:
                m = model_factory()
                m.fit(X_c, y_c)
                score = float(m.score(X_c, y_c)) if hasattr(m, "score") else 0.0
                round_models.append(m)
                round_scores.append(score)
            self.global_model = round_models[0]
            for m in round_models[1:]:
                if hasattr(self.global_model, "coef_") and hasattr(m, "coef_"):
                    self.global_model.coef_ = (
                        self.global_model.coef_ + m.coef_
                    ) / 2
            local_models = round_models
            self.round_history.append({
                "round": rnd + 1,
                "avg_client_score": float(np.mean(round_scores)),
                "num_clients": len(clients),
            })

        final_score = float(self.global_model.score(X, y)) if self.global_model else 0.0
        return {
            "rounds_completed": self.rounds,
            "final_score": final_score,
            "history": self.round_history,
        }
