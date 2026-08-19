"""Super Agent — validates, controls, and orchestrates all RAN agents."""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import AgentAction, AgentObservation
from src.common.utils import load_config, load_json, project_root


class SuperAgent:
    """Top-level autonomous intelligence controller for Intelligent RAN."""

    AGENT_WEIGHTS = {
        "scheduler": 0.07, "resource": 0.06, "mobility": 0.06, "security": 0.08,
        "energy": 0.05, "carbon": 0.05, "ran_sleep": 0.05, "renewable_energy": 0.05,
        "edge_inference": 0.05, "green_slice": 0.04, "traffic": 0.05,
        "qos": 0.05, "slice": 0.05, "qoe": 0.05,
        "channel_estimation": 0.03, "beamforming": 0.03, "csi": 0.03,
        "air_interface": 0.06, "digital_twin": 0.05, "spectrum": 0.04,
        "self_healing": 0.04, "knowledge": 0.01, "intent": 0.01,
        "agent_optimizer": 0.04, "coordination": 0.05,
    }

    def __init__(self):
        self.cfg = load_config("kpis.json")
        self.weights = self.cfg["utility_function"]["weights"]
        self.validation_log: list[dict] = []
        self.monitoring_log: list[dict] = []
        self.optimization_log: list[dict] = []
        self.handover_success_rate = 0.98
        self.security_score = 0.95
        self._load_nokia_policy()

    def _load_nokia_policy(self) -> None:
        path = project_root() / "data" / "knowledge_base" / "nokia_cfam_references.json"
        self.nokia_policy = load_json(path) if path.exists() else {}

    def validate_and_control(
        self,
        actions: list[AgentAction],
        kpi_before: dict,
        observation: AgentObservation,
    ) -> dict[str, Any]:
        """Validate agent outputs, resolve conflicts, approve parameter changes."""
        approved, rejected, modified = [], [], []

        for act in actions:
            verdict = self._validate_action(act, kpi_before, observation)
            if verdict["approved"]:
                if verdict.get("modified_params"):
                    act.parameters.update(verdict["modified_params"])
                approved.append(act)
            else:
                rejected.append({"agent": act.agent_id, "reason": verdict["reason"]})

        global_utility = self._compute_utility(approved, kpi_before)
        decision = {
            "approved_count": len(approved),
            "rejected_count": len(rejected),
            "approved_agents": [a.agent_id for a in approved],
            "rejected": rejected,
            "global_utility": round(global_utility, 4),
            "policy_refs": ["OSS_FC_017307", "SR003080"],
            "autonomy_level": min(1.0, len(approved) / max(len(actions), 1)),
        }
        self.validation_log.append(decision)
        return {"approved_actions": approved, "decision": decision}

    def monitor_and_optimize(
        self,
        monitoring_snapshot: dict,
        optimizer_actions: list[AgentAction],
    ) -> dict[str, Any]:
        """Record per-agent health monitoring and optimization triggers from Agent Optimizer."""
        degraded = monitoring_snapshot.get("degraded_agents", [])
        warnings = monitoring_snapshot.get("warning_agents", [])
        optimizations = []
        for act in optimizer_actions:
            p = act.parameters
            optimizations.append({
                "target_agent": p.get("target_agent"),
                "action": p.get("optimization_action"),
                "expected_recovery_pct": p.get("expected_recovery_pct", 0),
                "confidence": act.confidence,
            })

        record = {
            "monitored": monitoring_snapshot.get("monitored_count", 0),
            "healthy": monitoring_snapshot.get("healthy_count", 0),
            "warnings": monitoring_snapshot.get("warning_count", 0),
            "degraded": monitoring_snapshot.get("degraded_count", 0),
            "optimization_triggered": len(optimizations) > 0,
            "optimizations": optimizations,
            "degraded_agents": [a["agent"] for a in degraded],
            "warning_agents": [a["agent"] for a in warnings],
        }
        self.monitoring_log.append(record)
        if optimizations:
            self.optimization_log.extend(optimizations)
        return record

    def _validate_action(self, act: AgentAction, kpi_before: dict, obs: AgentObservation) -> dict:
        if act.confidence < 0.5:
            return {"approved": False, "reason": "low_confidence"}

        if act.action_type == "security" and act.parameters.get("threat_detected"):
            if act.parameters.get("threat_score", 0) < 0.6:
                return {"approved": False, "reason": "insufficient_threat_evidence"}

        if act.action_type == "energy" and act.parameters.get("sleep_mode"):
            if kpi_before.get("avg_throughput_mbps", 0) > 50:
                return {
                    "approved": True,
                    "modified_params": {"sleep_mode": False, "power_scale_factor": 0.7},
                    "reason": "throughput_guard",
                }

        if act.action_type == "schedule":
            prb = act.parameters.get("prb_assignment", 5)
            if prb > 50:
                return {"approved": True, "modified_params": {"prb_assignment": 25}}

        if act.action_type == "mobility" and act.parameters.get("handover_recommended"):
            if act.parameters.get("handover_probability", 0) < 0.4:
                return {"approved": False, "reason": "low_handover_probability"}

        if act.action_type == "slice":
            prb_share = act.parameters.get("prb_share_pct", 0)
            if prb_share > 0.45:
                return {"approved": True, "modified_params": {"prb_share_pct": 0.45}}

        return {"approved": True}

    def _compute_utility(self, actions: list[AgentAction], kpi: dict) -> float:
        tp = kpi.get("avg_throughput_mbps", 10)
        lat = kpi.get("avg_latency_ms", 5)
        pwr = kpi.get("total_power_w", 1000)
        sec = self.security_score
        u = (
            self.weights["alpha_throughput"] * tp
            - self.weights["beta_latency"] * lat
            + self.weights["gamma_security"] * sec * 10
            - self.weights["epsilon_energy"] * pwr / 1000
        )
        agent_bonus = sum(self.AGENT_WEIGHTS.get(a.agent_id.replace("_agent", ""), 0.01) * a.confidence
                          for a in actions)
        return u + agent_bonus

    def get_status(self) -> dict:
        return {
            "role": "super_agent",
            "agents_managed": list(self.AGENT_WEIGHTS.keys()),
            "validations": len(self.validation_log),
            "monitoring_cycles": len(self.monitoring_log),
            "optimizations_triggered": len(self.optimization_log),
            "last_monitoring": self.monitoring_log[-1] if self.monitoring_log else None,
            "handover_success_rate": self.handover_success_rate,
            "security_score": self.security_score,
            "nokia_cfam_features": len(self.nokia_policy.get("cfam_features", [])),
        }
