"""Train / validate / test all agents and persist models + metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.agents import AGENT_CLASSES
from src.common.utils import load_json, project_root, save_json


def load_frames() -> dict[str, pd.DataFrame]:
    data = project_root() / "data" / "datasets"
    return {
        "channel": pd.read_csv(data / "channel_estimation_dataset.csv"),
        "security": pd.read_csv(data / "security_dataset.csv"),
        "mobility": pd.read_csv(data / "mobility_dataset.csv"),
        "twin": pd.read_csv(data / "digital_twin_states.csv"),
    }


def train_all(only: list[str] | None = None, skip_existing: bool = False) -> dict:
    frames = load_frames()
    models_dir = project_root() / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    report_path = project_root() / "outputs" / "reports" / "train_val_test_report.json"
    report: dict = load_json(report_path) if report_path.exists() else {"agents": {}, "architecture": {}}
    report.setdefault("agents", {})
    names = list(AGENT_CLASSES)
    if only:
        names = [n for n in names if n in set(only)]
    elif skip_existing:
        names = [n for n in names if not (models_dir / f"{n}_agent.joblib").exists()]
    for name in names:
        cls = AGENT_CLASSES[name]
        agent = cls()
        if name in {
            "channel", "csi_prediction", "csi_feedback", "pilot", "equalizer", "air_interface",
            "beam", "optimization", "resource", "explainability", "orchestrator", "knowledge",
            "coordinator", "super", "spectrum", "self_healing",
        }:
            metrics = agent.train(frames["channel"])
        elif name in {"security", "mitigation"}:
            metrics = agent.train(frames["security"])
        elif name == "mobility":
            metrics = agent.train(frames["mobility"])
        else:
            metrics = agent.train(frames["twin"])
        agent.save(models_dir / f"{name}_agent.joblib")
        report["agents"][name] = {
            "trained": True,
            "metrics": metrics,
            "model_path": str(models_dir / f"{name}_agent.joblib"),
        }
        print(f"[ok] {name}: { {k: metrics[k] for k in list(metrics)[:4]} }")

    ch = frames["channel"]
    te = ch[ch["split"] == "test"]
    nmse_imp = (1 - te["nmse_ai"].mean() / te["nmse_mmse"].mean()) * 100
    ber_red = (1 - te["ber_ai"].mean() / te["ber_mmse"].mean()) * 100
    se_gain = (te["se_ai"].mean() / te["se_mmse"].mean() - 1) * 100
    report["architecture"] = {
        "test_nmse_ls": round(float(te["nmse_ls"].mean()), 6),
        "test_nmse_mmse": round(float(te["nmse_mmse"].mean()), 6),
        "test_nmse_ai": round(float(te["nmse_ai"].mean()), 6),
        "nmse_improvement_pct": round(float(nmse_imp), 2),
        "ber_reduction_pct": round(float(ber_red), 2),
        "spectral_efficiency_gain_pct": round(float(se_gain), 2),
        "csi_prediction_accuracy": round(float(te["csi_pred_accuracy"].mean() * 100), 2),
        "attack_rate_test": round(float(te["is_attack"].mean()), 4),
        "n_train": int((ch["split"] == "train").sum()),
        "n_validation": int((ch["split"] == "validation").sum()),
        "n_test": int((ch["split"] == "test").sum()),
    }
    save_json(report, project_root() / "outputs" / "reports" / "train_val_test_report.json")
    return report
