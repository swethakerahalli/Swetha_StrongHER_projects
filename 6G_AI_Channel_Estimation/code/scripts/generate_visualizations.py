#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization.plots import VisualizationSuite


if __name__ == "__main__":
    manifest = VisualizationSuite().generate_all()
    print(f"Generated {manifest['count']} plots")
