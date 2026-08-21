"""Shared utilities."""

from __future__ import annotations

import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data, path: Path | str, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent)


def load_config(name: str) -> dict:
    return load_json(project_root() / "config" / name)
