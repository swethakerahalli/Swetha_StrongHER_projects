#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn

if __name__ == "__main__":
    print("6G AI Channel Estimation dashboard: http://localhost:8090/dashboard")
    print("API docs: http://localhost:8090/docs")
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8090, reload=False)
