"""Closed-loop autonomous operation: Observe → Analyze → Predict → Plan → Act → Learn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.agents.base_agent import AgentAction
from src.common.utils import load_config
from src.digital_twin.ran_twin import RANDigitalTwin
from src.knowledge_graph.kg_engine import KnowledgeGraphEngine


@dataclass
class ClosedLoopCycle:
    phase: str
    output: dict[str, Any] = field(default_factory=dict)


class ClosedLoopController:
    PHASES = ["observe", "analyze", "predict", "plan", "act", "learn"]

    def __init__(self, agents: dict, twin: RANDigitalTwin | None = None):
        self.agents = agents
        self.twin = twin or RANDigitalTwin()
        self.kg = KnowledgeGraphEngine()
        self.cycle_history: list[list[ClosedLoopCycle]] = []
        self.cfg = load_config("kpis.json")
        self.weights = self.cfg["utility_function"]["weights"]

    def run_cycle(self, timestamp: int = 0) -> list[ClosedLoopCycle]:
        cycle = []

        state = self.twin.observe()
        cycle.append(ClosedLoopCycle("observe", {"network_state": state, "timestamp": timestamp}))

        kg_ctx = self.kg.agent_context("CELL_000")
        threats = kg_ctx.get("threats", [])
        cycle.append(ClosedLoopCycle("analyze", {
            "threats_detected": len(threats),
            "avg_throughput": state["avg_throughput"],
            "avg_latency": state["avg_latency"],
            "total_power_w": state["total_power_w"],
        }))

        predictions = {}
        for name, agent in self.agents.items():
            from src.agents.base_agent import AgentObservation
            obs = AgentObservation(
                timestamp=timestamp,
                features={
                    "cqi": 10, "sinr_db": 15, "buffer_occupancy": 0.5,
                    "latency_ms": state["avg_latency"], "mcs": 15, "prb_allocated": 10,
                    "throughput_mbps": state["avg_throughput"], "packet_loss": 0.01,
                    "cell_utilization": 0.6, "traffic_demand_mbps": 80,
                    "power_consumption_w": state["total_power_w"],
                    "renewable_pct": 25, "velocity_mps": 10, "rsrp_dbm": -90,
                    "handover_pending": 0, "direction_deg": 30,
                    "packet_rate_pps": 200, "auth_failures": 0,
                    "spectrum_anomaly_score": 0.1, "flow_entropy": 0.85,
                },
            )
            predictions[name] = agent.predict(obs)
        cycle.append(ClosedLoopCycle("predict", {
            name: act.parameters for name, act in predictions.items()
        }))

        utility = (
            self.weights["alpha_throughput"] * state["avg_throughput"]
            - self.weights["beta_latency"] * state["avg_latency"]
            - self.weights["epsilon_energy"] * state["total_power_w"] / 1000
        )
        plan = {
            "primary_agent": max(predictions, key=lambda k: predictions[k].confidence),
            "utility_score": round(utility, 3),
            "intent_compliance": utility > 0,
        }
        cycle.append(ClosedLoopCycle("plan", plan))

        actions = [
            {"action_type": act.action_type, "parameters": act.parameters}
            for act in predictions.values()
        ]
        new_state = self.twin.step(actions)
        cycle.append(ClosedLoopCycle("act", {"actions_applied": len(actions), "new_state": new_state}))

        reward = utility
        cycle.append(ClosedLoopCycle("learn", {
            "reward": round(reward, 3),
            "twin_fidelity": self.twin.fidelity_score(),
            "policy_update": "incremental",
        }))

        self.cycle_history.append(cycle)
        return cycle

    def run(self, iterations: int = 10) -> dict:
        for i in range(iterations):
            self.run_cycle(timestamp=i)
        final = self.twin.observe()
        return {
            "iterations": iterations,
            "final_throughput": final["avg_throughput"],
            "final_latency": final["avg_latency"],
            "final_power": final["total_power_w"],
            "twin_fidelity": self.twin.fidelity_score(),
            "cycles_completed": len(self.cycle_history),
        }
