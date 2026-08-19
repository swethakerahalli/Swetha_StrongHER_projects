#!/usr/bin/env python3
"""Start FastAPI Digital Twin RAN server."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    print("Starting Autonomous RAN API + Dashboard at http://localhost:8080")
    print("  Dashboard: http://localhost:8080/dashboard")
    print("  API docs:  http://localhost:8080/docs")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8080, reload=False)
