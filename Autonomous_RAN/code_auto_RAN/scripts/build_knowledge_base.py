#!/usr/bin/env python3
"""Build knowledge graph and feature store."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.knowledge_base_builder import build_knowledge_base


if __name__ == "__main__":
    paths = build_knowledge_base()
    print("Knowledge base built:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
