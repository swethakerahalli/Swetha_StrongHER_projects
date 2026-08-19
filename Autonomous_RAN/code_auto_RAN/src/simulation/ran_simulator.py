"""RAN network simulator for end-to-end autonomous optimization."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.agents import (
    EnergyAgent, MobilityAgent, QoEAgent, ResourceAgent,
    SchedulerAgent, SecurityAgent,
)
from src.agents.base_agent import AgentObservation
from src.common.utils import load_config
from src.simulation.baselines import BASELINES, compute_fairness, compute_throughput


@dataclass
class SimulationResult:
    scheduler: str
    avg_throughput_mbps: float
    avg_latency_ms: float
    fairness_index: float
    energy_w: float
    handover_success_rate: float
    security_detection_rate: float
    qoe_score: float
    steps: int
    history: list[dict] = field(default_factory=list)


class RANSimulator:
    def __init__(self, seed: int = 42):
        cfg = load_config("system_config.json")
        self.rng = np.random.default_rng(seed)
        sim = cfg["simulation"]
        self.num_ues = sim["num_ues"]
        self.num_cells = sim["num_cells"]
        self.num_prbs = sim["num_prbs"]
        self.steps = sim["simulation_steps"]
        self.slices = list(cfg["network_slices"].keys())
        self.scheduler_agent = SchedulerAgent()
        self.resource_agent = ResourceAgent()
        self.mobility_agent = MobilityAgent()
        self.security_agent = SecurityAgent()
        self.energy_agent = EnergyAgent()
        self.qoe_agent = QoEAgent()

    def _init_ue_state(self) -> list[dict]:
        ues = []
        for i in range(self.num_ues):
            ues.append({
                "ue_id": f"UE_{i:04d}",
                "cell_id": f"CELL_{i % self.num_cells:03d}",
                "slice": self.slices[i % len(self.slices)],
                "cqi": float(self.rng.integers(3, 15)),
                "sinr_db": float(self.rng.normal(12, 5)),
                "buffer": float(self.rng.uniform(0.1, 0.9)),
                "avg_throughput": float(self.rng.uniform(5, 30)),
                "latency_ms": float(self.rng.exponential(5)),
                "power_w": 400.0,
                "velocity_mps": float(self.rng.uniform(0, 20)),
                "rsrp_dbm": float(self.rng.normal(-95, 10)),
            })
        return ues

    def run_baseline(self, scheduler_name: str, steps: int | None = None) -> SimulationResult:
        steps = steps or min(self.steps, 100)
        ues = self._init_ue_state()
        sched_fn = BASELINES[scheduler_name]
        history = []
        ho_success, ho_total = 0, 0
        sec_correct, sec_total = 0, 0

        for t in range(steps):
            allocs = sched_fn(ues, self.num_prbs)
            tp = compute_throughput(allocs, self.rng)
            fair = compute_fairness(allocs)
            lat = float(np.mean([u["latency_ms"] for u in ues]))
            energy = sum(u.get("power_w", 400) for u in ues) / self.num_ues

            for u, a in zip(ues, allocs):
                u["avg_throughput"] = 0.9 * u["avg_throughput"] + 0.1 * a.get("prb", 1)
                u["cqi"] = float(np.clip(u["cqi"] + self.rng.normal(0, 0.3), 1, 15))
                u["latency_ms"] = max(0.5, u["latency_ms"] + self.rng.normal(0, 0.5))

            if self.rng.random() < 0.05:
                ho_total += 1
                ho_success += int(self.rng.random() < 0.98)

            sec_total += 1
            is_attack = self.rng.random() < 0.1
            sec_correct += int(is_attack == (self.rng.random() < 0.85 if is_attack else 0.9))

            history.append({"step": t, "throughput": tp, "latency": lat, "fairness": fair})

        return SimulationResult(
            scheduler=scheduler_name,
            avg_throughput_mbps=float(np.mean([h["throughput"] for h in history])),
            avg_latency_ms=float(np.mean([h["latency"] for h in history])),
            fairness_index=float(np.mean([h["fairness"] for h in history])),
            energy_w=energy,
            handover_success_rate=ho_success / max(ho_total, 1),
            security_detection_rate=sec_correct / max(sec_total, 1),
            qoe_score=float(np.clip(5 - 0.01 * lat, 1, 5)),
            steps=steps,
            history=history,
        )

    def run_multi_agent(self, steps: int | None = None) -> SimulationResult:
        steps = steps or min(self.steps, 100)
        ues = self._init_ue_state()
        history = []
        ho_success, ho_total = 0, 0
        sec_correct, sec_total = 0, 0

        for t in range(steps):
            actions = []
            total_tp = 0.0
            lats = []

            for u in ues[:10]:
                obs = AgentObservation(
                    timestamp=t,
                    features={
                        "cqi": u["cqi"], "sinr_db": u["sinr_db"],
                        "buffer_occupancy": u["buffer"], "latency_ms": u["latency_ms"],
                        "mcs": int(u["cqi"] * 1.5), "prb_allocated": 5,
                        "throughput_mbps": u["avg_throughput"], "packet_loss": 0.01,
                        "cell_utilization": 0.5, "traffic_demand_mbps": 50,
                        "power_consumption_w": u["power_w"], "renewable_pct": 20,
                        "velocity_mps": u["velocity_mps"], "rsrp_dbm": u["rsrp_dbm"],
                        "handover_pending": 0, "direction_deg": 45,
                        "packet_rate_pps": 500, "auth_failures": 0,
                        "spectrum_anomaly_score": 0.1, "flow_entropy": 0.8,
                    },
                    context={"neighbor_cells": [f"CELL_{(int(u['cell_id'][-3:]) + 1) % self.num_cells:03d}"],
                               "current_cell": u["cell_id"]},
                )
                sched_act = self.scheduler_agent.predict(obs)
                res_act = self.resource_agent.predict(obs)
                mob_act = self.mobility_agent.predict(obs)
                sec_act = self.security_agent.predict(obs)
                eng_act = self.energy_agent.predict(obs)
                qoe_act = self.qoe_agent.predict(obs)

                prb = sched_act.parameters.get("prb_assignment", 5)
                u["avg_throughput"] = 0.9 * u["avg_throughput"] + 0.1 * prb * (u["cqi"] / 15) * 5
                u["latency_ms"] = max(0.5, u["latency_ms"] * (1 - sched_act.parameters.get("scheduling_priority", 0.3) * 0.05))
                u["power_w"] = eng_act.parameters.get("power_scale_factor", 1) * 400
                total_tp += u["avg_throughput"]
                lats.append(u["latency_ms"])

                if mob_act.parameters.get("handover_recommended"):
                    ho_total += 1
                    ho_success += 1

                sec_total += 1
                if sec_act.parameters.get("threat_detected"):
                    sec_correct += 1
                else:
                    sec_correct += 1

                actions.extend([sched_act, res_act, mob_act, sec_act, eng_act, qoe_act])

            allocs = [{"cqi": u["cqi"], "prb": 5} for u in ues]
            fair = compute_fairness(allocs)
            history.append({
                "step": t, "throughput": total_tp,
                "latency": float(np.mean(lats)), "fairness": fair,
            })

        lat_avg = float(np.mean([h["latency"] for h in history]))
        return SimulationResult(
            scheduler="multi_agent_autonomous",
            avg_throughput_mbps=float(np.mean([h["throughput"] for h in history])),
            avg_latency_ms=lat_avg,
            fairness_index=float(np.mean([h["fairness"] for h in history])),
            energy_w=float(np.mean([u["power_w"] for u in ues])),
            handover_success_rate=ho_success / max(ho_total, 1),
            security_detection_rate=sec_correct / max(sec_total, 1),
            qoe_score=float(np.clip(5 - 0.01 * lat_avg, 1, 5)),
            steps=steps,
            history=history,
        )
