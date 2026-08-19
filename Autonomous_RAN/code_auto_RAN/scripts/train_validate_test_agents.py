#!/usr/bin/env python3
"""Train, validate, and test all RAN agents with metrics report."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agents import (
    AgentOptimizerAgent, AirInterfaceAgent, BeamformingAgent, CarbonAgent, CoordinationAgent, CSIAgent,
    ChannelEstimationAgent, DigitalTwinAgent, EdgeInferenceAgent, EnergyAgent, GreenSliceAgent, IntentAgent,
    KnowledgeAgent, MobilityAgent, QoEAgent, QoSAgent, RANSleepAgent, RenewableEnergyAgent,
    ResourceAgent, SchedulerAgent, SecurityAgent, SelfHealingAgent, SliceAgent, SpectrumAgent,
    TrafficAgent,
)
from src.agents.base_agent import AgentObservation
from src.common.utils import project_root, save_json
from src.orchestration.super_agent_controller import SuperAgentController
from src.api.agent_performance_service import AgentPerformanceService
from src.visualization.agent_plots import generate_agent_plots


def train_agent(name, agent, df):
  if df is None or len(df) == 0:
    return agent.train({})
  if name == "channel_estimation":
    mob = pd.read_csv(project_root() / "data/datasets/mobility_traces.csv")
    merged = df[["sinr_db", "cqi", "rsrp_dbm"]].head(len(mob)).copy()
    merged["velocity_mps"] = mob["velocity_mps"].values[:len(merged)]
    return agent.train(merged)
  return agent.train(df)


def validate_agent(name, agent, df):
  if df is None or len(df) < 20:
    return {"status": "skipped"}
  try:
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)
    train_metrics = train_agent(name, agent, train_df)
    test_sample = test_df.sample(min(30, len(test_df)), random_state=1)
    predictions = []
    for _, row in test_sample.iterrows():
      features = {k: float(v) for k, v in row.items() if isinstance(v, (int, float, np.integer, np.floating))}
      context = {"slice": row["slice"]} if "slice" in row.index else {}
      obs = AgentObservation(timestamp=0, features=features, context=context)
      act = agent.predict(obs)
      predictions.append(act.confidence)
    return {
      **train_metrics,
      "test_samples": len(test_sample),
      "avg_confidence": round(float(np.mean(predictions)), 3),
      "status": "validated",
    }
  except Exception as e:
    return {"status": "error", "error": str(e)}


def main():
  data_dir = project_root() / "data/datasets"
  models_dir = project_root() / "outputs/models"
  models_dir.mkdir(parents=True, exist_ok=True)

  ran = pd.read_csv(data_dir / "ran_kpi_dataset.csv")
  mob = pd.read_csv(data_dir / "mobility_traces.csv")
  sec = pd.read_csv(data_dir / "security_events.csv")
  eng = pd.read_csv(data_dir / "energy_metrics.csv")
  slice_df = pd.read_csv(data_dir / "slice_utilization.csv")
  ran_edge = ran.copy()
  if len(sec) >= len(ran):
    ran_edge["packet_rate_pps"] = sec["packet_rate_pps"].values[:len(ran_edge)]
  else:
    ran_edge["packet_rate_pps"] = 200.0
  if "cell_utilization" not in ran_edge.columns:
    if "buffer_occupancy" in ran_edge.columns:
      ran_edge["cell_utilization"] = ran_edge["buffer_occupancy"].clip(0, 1)
    elif "prb_allocated" in ran_edge.columns:
      ran_edge["cell_utilization"] = (ran_edge["prb_allocated"] / ran_edge["prb_allocated"].max()).clip(0, 1)
    else:
      ran_edge["cell_utilization"] = 0.5
  slice_green = slice_df.copy()
  slice_green["power_consumption_w"] = eng["power_consumption_w"].mean()
  slice_green["renewable_pct"] = eng["renewable_pct"].mean()

  agents = {
    "scheduler": SchedulerAgent(),
    "resource": ResourceAgent(),
    "mobility": MobilityAgent(),
    "security": SecurityAgent(),
    "energy": EnergyAgent(),
    "carbon": CarbonAgent(),
    "ran_sleep": RANSleepAgent(),
    "renewable_energy": RenewableEnergyAgent(),
    "edge_inference": EdgeInferenceAgent(),
    "green_slice": GreenSliceAgent(),
    "traffic": TrafficAgent(),
    "qos": QoSAgent(),
    "slice": SliceAgent(),
    "qoe": QoEAgent(),
    "channel_estimation": ChannelEstimationAgent(),
    "beamforming": BeamformingAgent(),
    "csi": CSIAgent(),
    "air_interface": AirInterfaceAgent(),
    "digital_twin": DigitalTwinAgent(),
    "spectrum": SpectrumAgent(),
    "self_healing": SelfHealingAgent(),
    "knowledge": KnowledgeAgent(),
    "intent": IntentAgent(),
  }
  data_map = {
    "scheduler": ran, "resource": eng, "mobility": mob, "security": sec,
    "energy": eng, "carbon": eng, "ran_sleep": eng, "renewable_energy": eng,
    "edge_inference": ran_edge, "green_slice": slice_green, "traffic": ran,
    "qos": ran, "slice": slice_df, "qoe": ran, "channel_estimation": ran,
    "beamforming": ran, "csi": ran, "air_interface": ran,
    "digital_twin": eng, "spectrum": sec, "self_healing": sec,
    "knowledge": None, "intent": None,
  }

  perf_comparison = AgentPerformanceService().get_comparison()
  optimizer_df = pd.DataFrame([{
    "performance_index": a["performance_index"],
    "improvement_pct": a["improvement_pct"],
    "confidence": a["confidence"] or 0.8,
    "validation_score": a.get("validation_raw", 0.8),
    "degradation_score": max(0, 65 - a["performance_index"]),
  } for a in perf_comparison["agents"] if a["agent"] not in ("knowledge", "intent", "agent_optimizer", "coordination")])
  data_map["agent_optimizer"] = optimizer_df if len(optimizer_df) > 0 else None
  agents["agent_optimizer"] = AgentOptimizerAgent()

  coord_df = ran.copy()
  coord_df["num_actions"] = np.random.randint(8, 24, len(coord_df))
  coord_df["num_conflicts"] = np.random.randint(0, 8, len(coord_df))
  coord_df["avg_confidence"] = np.random.uniform(0.7, 0.98, len(coord_df))
  coord_df["total_power_w"] = eng["power_consumption_w"].values[:len(coord_df)] if len(eng) >= len(coord_df) else 2800
  data_map["coordination"] = coord_df
  agents["coordination"] = CoordinationAgent()

  training, validation, testing = {}, {}, {}

  print("=" * 60)
  print("TRAIN / VALIDATE / TEST — All RAN Agents")
  print("=" * 60)

  for name, agent in agents.items():
    df = data_map.get(name)
    print(f"\n[{name}]")
    t = train_agent(name, agent, df) if df is not None else agent.train({})
    training[name] = t
    print(f"  Train: {t}")
    if name not in ("knowledge", "intent"):
      agent.save(models_dir / f"{name}_agent.joblib")
    v = validate_agent(name, agent, df)
    validation[name] = v
    print(f"  Validate: {v}")
    testing[name] = {"inference_ok": v.get("status") in ("validated", "skipped"),
                     "avg_confidence": v.get("avg_confidence", 0)}

  print("\n[Super Agent Integration Test]")
  ctrl = SuperAgentController()
  result = ctrl.run_all_agents(
    intent="Optimize URLLC latency and reduce energy by 15%",
    kpi_before={"avg_throughput_mbps": 20, "avg_latency_ms": 8, "total_power_w": 2800},
  )
  testing["super_agent"] = {
    "approved": result["super_agent_decision"]["approved_count"],
    "rejected": result["super_agent_decision"]["rejected_count"],
    "utility": result["super_agent_decision"]["global_utility"],
  }
  print(f"  Approved: {testing['super_agent']['approved']}, Utility: {testing['super_agent']['utility']}")

  report = {"training": training, "validation": validation, "testing": testing}
  out = project_root() / "outputs/reports/agent_train_validate_test.json"
  save_json(report, out)
  print(f"\nReport: {out}")

  print("\nGenerating per-agent visualizations...")
  viz = generate_agent_plots(report)
  print(f"Generated {viz['count']} agent plots")

  print("=" * 60)


if __name__ == "__main__":
  main()
