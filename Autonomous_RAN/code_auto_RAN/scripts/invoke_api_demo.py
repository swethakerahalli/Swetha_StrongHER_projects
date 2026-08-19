#!/usr/bin/env python3
"""Invoke Digital Twin RAN APIs and demonstrate before/after KPI changes."""

import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = "http://localhost:8080"


def call(method, path, body=None):
    url = f"{BASE}{path}"
    kwargs = {"timeout": 30}
    if body is not None:
        kwargs["json"] = body
    r = requests.request(method, url, **kwargs)
    r.raise_for_status()
    return r.json()


def main():
    print("=" * 60)
    print("Autonomous RAN API Invocation Demo")
    print("=" * 60)

    try:
        health = call("GET", "/api/health")
        print(f"\n[1] Health: {health}")
    except requests.ConnectionError:
        print("ERROR: API server not running. Start with: python scripts/run_api_server.py")
        sys.exit(1)

    print("\n[2] Reset baseline KPI...")
    baseline = call("POST", "/api/kpi/baseline/reset")
    print(f"    Baseline throughput: {baseline['baseline']['avg_throughput_mbps']} Mbps")

    print("\n[3] Current KPI (before agent run)...")
    before = call("GET", "/api/kpi/current")
    print(f"    Throughput: {before['avg_throughput_mbps']} Mbps, Latency: {before['avg_latency_ms']} ms")

    print("\n[4] Invoke Super Agent — run all agents...")
    result = call("POST", "/api/agents/run", {
        "intent": "Reduce energy by 20%, improve throughput, maintain URLLC QoS",
        "cell_id": "CELL_000",
    })
    print(f"    Action ID: {result['action_id']}")
    print(f"    Agents: {result['agents_invoked']}")
    print(f"    Throughput: {result['kpi_before']['avg_throughput_mbps']} -> {result['kpi_after']['avg_throughput_mbps']} Mbps")
    print(f"    Latency: {result['kpi_before']['avg_latency_ms']} -> {result['kpi_after']['avg_latency_ms']} ms")
    print(f"    Power: {result['kpi_before']['total_power_w']} -> {result['kpi_after']['total_power_w']} W")
    print(f"    Super Agent utility: {result.get('super_agent_decision', {}).get('global_utility')}")
    print(f"    Parameter updates: {len(result['parameter_updates'])}")

    print("\n[5] KPI comparison (before vs after)...")
    cmp = call("GET", "/api/kpi/comparison")
    for k, pct in cmp["delta_pct"].items():
        if k in ("avg_throughput_mbps", "avg_latency_ms", "total_power_w", "qos_sla_compliance"):
            print(f"    {k}: {pct:+.1f}%")

    print("\n[6] Update cell parameters via API...")
    cell_update = call("PUT", "/api/ran/cells/CELL_001", {"tx_power_dbm": 35, "load": 0.7})
    print(f"    Cell updated, current throughput: {cell_update['current_kpi']['avg_throughput_mbps']} Mbps")

    print("\n[7] Chatbot — KPI query...")
    chat = call("POST", "/api/chat", {"message": "Show KPI comparison and agent status"})
    print(f"    Reply: {chat['reply'][:200]}...")

    print("\n[8] Closed-loop automation (5 iterations)...")
    loop = call("POST", "/api/closed-loop/run", {"iterations": 5})
    print(f"    Throughput: {loop['kpi_before']['avg_throughput_mbps']} -> {loop['kpi_after']['avg_throughput_mbps']} Mbps")

    print("\n[9] Super Agent status...")
    sa = call("GET", "/api/super-agent/status")
    print(f"    Managed agents: {len(sa['agents_managed'])}, Validations: {sa['validations']}")

    out = ROOT / "outputs" / "reports" / "api_invocation_demo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"agent_run": result, "comparison": cmp, "super_agent": sa}, f, indent=2, default=str)
    print(f"\n[Complete] Results saved to {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
