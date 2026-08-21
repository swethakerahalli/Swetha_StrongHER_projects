#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset_generator import ChannelDatasetGenerator


def main():
    paths = ChannelDatasetGenerator().generate_all()
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
