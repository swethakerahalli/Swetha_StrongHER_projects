#!/usr/bin/env python3
"""Generate synthetic RAN datasets."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset_generator import generate_datasets


if __name__ == "__main__":
    import pandas as pd

    paths = generate_datasets(num_samples=80_000)
    print("Generated datasets (80,000 rows each):")
    for name, path in paths.items():
        if path.suffix == ".csv":
            n = sum(1 for _ in open(path, encoding="utf-8")) - 1
            cols = pd.read_csv(path, nrows=0).columns.tolist()
            print(f"  {name}: {path.name}  rows={n}  cols={len(cols)}")
        else:
            print(f"  {name}: {path}")
