#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.pipeline import train_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train / validate / test CSI agents")
    parser.add_argument("--skip-existing", action="store_true", help="Skip agents that already have a joblib")
    parser.add_argument("--only", nargs="*", help="Train only these agent ids")
    args = parser.parse_args()
    report = train_all(only=args.only, skip_existing=args.skip_existing)
    print("Architecture:", report["architecture"])
    print("Agents trained in report:", len(report.get("agents", {})))
