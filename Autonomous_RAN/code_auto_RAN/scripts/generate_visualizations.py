#!/usr/bin/env python3
"""Generate all visualizations: histograms, CDFs, heatmaps, classification, simulation plots."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization.plot_generator import generate_all_plots


def main():
    print("Generating visualizations...")
    manifest = generate_all_plots()
    print(f"\nGenerated {manifest['count']} plots in outputs/plots/:")
    for p in manifest["plots_generated"]:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
