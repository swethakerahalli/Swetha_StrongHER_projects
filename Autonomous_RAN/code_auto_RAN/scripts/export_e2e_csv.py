#!/usr/bin/env python3
"""Export end-to-end results and dataset inventory to CSV files."""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.api.e2e_service import E2EResultsService
from src.common.utils import project_root
from src.data.dataset_generator import generate_datasets


def export_all():
    out = project_root() / "outputs" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    data_dir = project_root() / "data" / "datasets"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Regenerating datasets (CSV)...")
    generate_datasets()

    svc = E2EResultsService()
    summary = svc.get_summary()

    pd.DataFrame(summary["datasets"]).to_csv(out / "dataset_inventory.csv", index=False)
    pd.DataFrame(summary["benchmark"]["results"]).to_csv(out / "benchmark_results.csv", index=False)

    train_path = project_root() / "outputs" / "reports" / "agent_train_validate_test.json"
    if train_path.exists():
        train = json.loads(train_path.read_text(encoding="utf-8"))
        rows = []
        for agent, m in train.get("training", {}).items():
            v = train.get("validation", {}).get(agent, {})
            t = train.get("testing", {}).get(agent, {})
            rows.append({
                "agent": agent,
                "ai_type": "llm" if agent in ("knowledge", "intent") else "sklearn",
                **{f"train_{k}": v_ for k, v_ in m.items()},
                "validation_status": v.get("status"),
                "avg_confidence": v.get("avg_confidence", t.get("avg_confidence")),
            })
        pd.DataFrame(rows).to_csv(out / "agent_training_results.csv", index=False)

    api_path = out / "api_invocation_demo.json"
    if api_path.exists():
        api = json.loads(api_path.read_text(encoding="utf-8"))
        cmp = api.get("comparison", {})
        kpi_rows = []
        for key in cmp.get("before", {}):
            if key == "timestamp":
                continue
            kpi_rows.append({
                "kpi": key,
                "before": cmp["before"].get(key),
                "after": cmp["after"].get(key),
                "delta": cmp.get("delta", {}).get(key),
                "delta_pct": cmp.get("delta_pct", {}).get(key),
            })
        pd.DataFrame(kpi_rows).to_csv(out / "api_kpi_before_after.csv", index=False)

    phases = pd.DataFrame(summary["pipeline_phases"])
    phases.to_csv(out / "e2e_pipeline_phases.csv", index=False)

    demo = summary["api_demo"]
    pd.DataFrame([demo]).to_csv(out / "e2e_api_demo_summary.csv", index=False)

    print("CSV exports written to outputs/reports/:")
    for f in sorted(out.glob("*.csv")):
        print(f"  {f.name}")


if __name__ == "__main__":
    export_all()
