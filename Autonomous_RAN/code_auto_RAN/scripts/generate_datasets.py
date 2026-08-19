#!/usr/bin/env python3
"""Generate synthetic RAN datasets."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset_generator import generate_datasets


if __name__ == "__main__":
    paths = generate_datasets()
    print("Generated datasets:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
