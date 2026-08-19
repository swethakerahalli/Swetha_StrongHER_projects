"""Shared utilities and types."""

from pathlib import Path
import json


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: Path | str, indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_config(name: str) -> dict:
    return load_json(project_root() / "config" / name)
