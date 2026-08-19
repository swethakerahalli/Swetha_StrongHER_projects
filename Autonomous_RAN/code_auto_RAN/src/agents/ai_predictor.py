"""Shared AI inference helpers for RAN agents."""

from __future__ import annotations

import numpy as np


def ml_predict(model, features: dict, keys: list[str], default: float = 0.0) -> float:
    """Run sklearn model inference; requires trained model."""
    if model is None:
        raise RuntimeError("AI model not loaded — train agent first")
    x = np.array([[float(features.get(k, default)) for k in keys]], dtype=float)
    pred = model.predict(x)
    return float(pred[0])


def ml_classify_proba(model, features: dict, keys: list[str], class_idx: int = 1) -> float:
    if model is None:
        raise RuntimeError("AI model not loaded — train agent first")
    x = np.array([[float(features.get(k, 0.0)) for k in keys]], dtype=float)
    return float(model.predict_proba(x)[0][class_idx])


def scale_to_range(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))
