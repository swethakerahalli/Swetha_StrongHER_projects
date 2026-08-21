#!/usr/bin/env python3
"""Full pipeline: dataset → train/val/test → plots → docs/slides."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    print(f"\n=== {script} ===")
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT)


def main():
    run("generate_datasets.py")
    run("train_validate_test.py")
    run("generate_visualizations.py")
    run("generate_docs.py")
    run("generate_slides.py")
    print("\nEnd-to-end 6G AI Channel Estimation pipeline complete.")
    print("Start dashboard: python scripts/run_api_server.py")


if __name__ == "__main__":
    main()
