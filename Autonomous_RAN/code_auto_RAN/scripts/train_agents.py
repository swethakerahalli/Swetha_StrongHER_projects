#!/usr/bin/env python3
"""Train all AI agents on generated datasets."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents import (
    EnergyAgent, IntentAgent, KnowledgeAgent, MobilityAgent, QoEAgent, ResourceAgent,
    SchedulerAgent, SecurityAgent,
)
from src.common.utils import project_root, save_json
from src.federated.federated_learning import FederatedLearningCoordinator


def main():
    data_dir = project_root() / "data" / "datasets"
    models_dir = project_root() / "outputs" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    agents = {
        "scheduler": (SchedulerAgent(), pd.read_csv(data_dir / "ran_kpi_dataset.csv")),
        "resource": (ResourceAgent(), pd.read_csv(data_dir / "energy_metrics.csv")),
        "mobility": (MobilityAgent(), pd.read_csv(data_dir / "mobility_traces.csv")),
        "security": (SecurityAgent(), pd.read_csv(data_dir / "security_events.csv")),
        "energy": (EnergyAgent(), pd.read_csv(data_dir / "energy_metrics.csv")),
        "qoe": (QoEAgent(), pd.read_csv(data_dir / "ran_kpi_dataset.csv")),
        "knowledge": (KnowledgeAgent(), None),
        "intent": (IntentAgent(), None),
    }

    metrics = {}
    for name, (agent, df) in agents.items():
        print(f"Training {name} agent...")
        if df is not None:
            result = agent.train(df)
        else:
            result = agent.train({})
        if name not in ("knowledge", "intent"):
            agent.save(models_dir / f"{name}_agent.joblib")
        metrics[name] = result
        print(f"  -> {result}")

    fl = FederatedLearningCoordinator(num_clients=5, rounds=5)
    sec_df = pd.read_csv(data_dir / "security_events.csv")
    X = sec_df[["packet_rate_pps", "auth_failures", "spectrum_anomaly_score", "flow_entropy"]].values
    y = sec_df["is_attack"].values
    from sklearn.ensemble import RandomForestClassifier
    fl_result = fl.train_federated(lambda: RandomForestClassifier(n_estimators=20, random_state=42), X, y)
    metrics["federated_security"] = fl_result

    report = {"training_metrics": metrics, "models_saved": list(agents.keys())}
    save_json(report, project_root() / "outputs" / "reports" / "training_report.json")
    print(f"\nTraining complete. Report saved to outputs/reports/training_report.json")


if __name__ == "__main__":
    main()
