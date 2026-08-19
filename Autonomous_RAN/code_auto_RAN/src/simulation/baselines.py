"""Baseline schedulers for benchmarking (RR, PF, Max-TP, SON)."""

from __future__ import annotations

import numpy as np


def round_robin(ue_metrics: list[dict], num_prbs: int = 100) -> list[dict]:
    n = len(ue_metrics) or 1
    prb_each = max(1, num_prbs // n)
    return [{**m, "prb": prb_each, "scheduler": "round_robin"} for m in ue_metrics]


def proportional_fair(ue_metrics: list[dict], num_prbs: int = 100) -> list[dict]:
    scores = []
    for m in ue_metrics:
        cqi = max(m.get("cqi", 1), 1)
        avg = max(m.get("avg_throughput", 0.1), 0.1)
        scores.append((cqi / 15) / avg)
    total = sum(scores) or 1.0
    return [
        {**m, "prb": max(1, int(num_prbs * s / total)), "scheduler": "proportional_fair"}
        for m, s in zip(ue_metrics, scores)
    ]


def max_throughput(ue_metrics: list[dict], num_prbs: int = 100) -> list[dict]:
    if not ue_metrics:
        return []
    best = max(ue_metrics, key=lambda m: m.get("cqi", 0))
    return [
        {**m, "prb": num_prbs if m is best else 1, "scheduler": "max_throughput"}
        for m in ue_metrics
    ]


def son_static(ue_metrics: list[dict], num_prbs: int = 100) -> list[dict]:
    """Static SON-like allocation based on fixed QoS weights."""
    weights = {"URLLC": 3.0, "eMBB": 2.0, "mMTC": 1.0}
    scores = [weights.get(m.get("slice", "eMBB"), 1.0) * m.get("cqi", 5) for m in ue_metrics]
    total = sum(scores) or 1.0
    return [
        {**m, "prb": max(1, int(num_prbs * s / total)), "scheduler": "son_static"}
        for m, s in zip(ue_metrics, scores)
    ]


BASELINES = {
    "round_robin": round_robin,
    "proportional_fair": proportional_fair,
    "max_throughput": max_throughput,
    "son_static": son_static,
}


def compute_throughput(allocations: list[dict], rng: np.random.Generator) -> float:
    total = 0.0
    for a in allocations:
        eff = (a.get("cqi", 5) / 15) * a.get("prb", 1)
        total += eff * rng.exponential(5)
    return total


def compute_fairness(allocations: list[dict]) -> float:
    prbs = [a.get("prb", 1) for a in allocations]
    if not prbs:
        return 1.0
    s = sum(prbs)
    n = len(prbs)
    if s == 0:
        return 0.0
    props = [p / s for p in prbs]
    jain = (sum(props) ** 2) / (n * sum(p ** 2 for p in props) + 1e-9)
    return float(jain)
