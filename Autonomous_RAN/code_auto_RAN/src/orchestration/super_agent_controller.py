"""Super Agent Controller — full multi-agent stack with super-agent validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents import (
    AgentOptimizerAgent, AirInterfaceAgent, BeamformingAgent, CarbonAgent, CoordinationAgent, CSIAgent,
    ChannelEstimationAgent, DigitalTwinAgent, EdgeInferenceAgent, EnergyAgent, GreenSliceAgent, IntentAgent,
    KnowledgeAgent, MobilityAgent, QoEAgent, QoSAgent, RANSleepAgent, RenewableEnergyAgent,
    ResourceAgent, SchedulerAgent, SecurityAgent, SelfHealingAgent, SliceAgent, SpectrumAgent,
    TrafficAgent,
)
from src.api.agent_monitoring_service import AgentMonitoringService
from src.agents.base_agent import AgentAction, AgentObservation
from src.agents.super_agent import SuperAgent
from src.common.utils import load_config, load_json, project_root, save_json
from src.orchestration.closed_loop import ClosedLoopController


class SuperAgentController:
    def __init__(self):
        models_dir = project_root() / "outputs" / "models"
        self.agents = {
            "scheduler": self._load_or_create(SchedulerAgent, models_dir / "scheduler_agent.joblib"),
            "resource": self._load_or_create(ResourceAgent, models_dir / "resource_agent.joblib"),
            "mobility": self._load_or_create(MobilityAgent, models_dir / "mobility_agent.joblib"),
            "security": self._load_or_create(SecurityAgent, models_dir / "security_agent.joblib"),
            "energy": self._load_or_create(EnergyAgent, models_dir / "energy_agent.joblib"),
            "carbon": self._load_or_create(CarbonAgent, models_dir / "carbon_agent.joblib"),
            "ran_sleep": self._load_or_create(RANSleepAgent, models_dir / "ran_sleep_agent.joblib"),
            "renewable_energy": self._load_or_create(RenewableEnergyAgent, models_dir / "renewable_energy_agent.joblib"),
            "edge_inference": self._load_or_create(EdgeInferenceAgent, models_dir / "edge_inference_agent.joblib"),
            "green_slice": self._load_or_create(GreenSliceAgent, models_dir / "green_slice_agent.joblib"),
            "traffic": self._load_or_create(TrafficAgent, models_dir / "traffic_agent.joblib"),
            "qos": self._load_or_create(QoSAgent, models_dir / "qos_agent.joblib"),
            "slice": self._load_or_create(SliceAgent, models_dir / "slice_agent.joblib"),
            "qoe": self._load_or_create(QoEAgent, models_dir / "qoe_agent.joblib"),
            "channel_estimation": self._load_or_create(ChannelEstimationAgent, models_dir / "channel_estimation_agent.joblib"),
            "beamforming": self._load_or_create(BeamformingAgent, models_dir / "beamforming_agent.joblib"),
            "csi": self._load_or_create(CSIAgent, models_dir / "csi_agent.joblib"),
            "air_interface": self._load_or_create(AirInterfaceAgent, models_dir / "air_interface_agent.joblib"),
            "digital_twin": self._load_or_create(DigitalTwinAgent, models_dir / "digital_twin_agent.joblib"),
            "spectrum": self._load_or_create(SpectrumAgent, models_dir / "spectrum_agent.joblib"),
            "self_healing": self._load_or_create(SelfHealingAgent, models_dir / "self_healing_agent.joblib"),
            "knowledge": KnowledgeAgent(),
            "intent": IntentAgent(),
            "agent_optimizer": self._load_or_create(AgentOptimizerAgent, models_dir / "agent_optimizer_agent.joblib"),
            "coordination": self._load_or_create(CoordinationAgent, models_dir / "coordination_agent.joblib"),
        }
        self.meta_agents = frozenset({"agent_optimizer", "coordination"})
        self.super_agent = SuperAgent()
        self.monitoring_svc = AgentMonitoringService()
        self.cfg = load_config("agents_config.json")
        self.closed_loop = ClosedLoopController(self.agents)
        self.message_log: list[dict] = []
        self.handover_success_rate = 0.98
        self.security_score = 0.95

    @staticmethod
    def _load_or_create(cls, path: Path):
        agent = cls(path if path.exists() else None)
        return agent

    def get_agent_status(self) -> list[dict]:
        """Return AI readiness status for all managed agents."""
        status = []
        for name, agent in self.agents.items():
            ai_type = "llm" if name in ("knowledge", "intent") else "sklearn"
            status.append({
                "name": name,
                "agent_id": agent.agent_id,
                "ai_driven": True,
                "ai_type": ai_type,
                "is_trained": getattr(agent, "is_trained", True),
                "model_loaded": getattr(agent, "model", None) is not None or ai_type == "llm",
            })
        return status

    def build_observation(self, features: dict | None = None, context: dict | None = None) -> AgentObservation:
        defaults = {
            "cqi": 10, "sinr_db": 15, "rsrp_dbm": -90, "rsrq_db": -10,
            "buffer_occupancy": 0.5, "latency_ms": 5, "mcs": 15, "prb_allocated": 10,
            "throughput_mbps": 30, "packet_loss": 0.01, "cell_utilization": 0.6,
            "traffic_demand_mbps": 80, "power_consumption_w": 400, "renewable_pct": 20,
            "carbon_intensity_gco2_kwh": 380,
            "velocity_mps": 10, "handover_pending": 0, "direction_deg": 45,
            "packet_rate_pps": 200, "auth_failures": 0,
            "spectrum_anomaly_score": 0.1, "flow_entropy": 0.85,
            "prb_utilization": 0.6, "active_ues": 15, "sla_compliance": 0.97,
            "latency_p99_ms": 8, "interference_db": 3.0,
            "avg_throughput_mbps": 20, "total_power_w": 2800,
        }
        if features:
            defaults.update(features)
        return AgentObservation(timestamp=0, features=defaults, context=context or {})

    def run_all_agents(
        self,
        observation: AgentObservation | None = None,
        intent: str = "",
        agent_filter: list[str] | None = None,
        kpi_before: dict | None = None,
    ) -> dict[str, Any]:
        obs = observation or self.build_observation()
        if intent:
            obs.context["operator_intent"] = intent
            intent_act = self.agents["intent"].predict(obs)
            obs.context["parsed_intent"] = intent_act.parameters

        names = agent_filter or [n for n in self.agents if n not in self.meta_agents]
        raw_actions = []
        for name in names:
            if name not in self.agents:
                continue
            act = self.agents[name].predict(obs)
            raw_actions.append(act)
            self.message_log.append({"from": name, "action": act.action_type, "confidence": act.confidence})

        coord_obs = AgentObservation(
            timestamp=obs.timestamp, features=obs.features,
            context={**obs.context, "raw_actions": raw_actions},
        )
        coord_result = self.agents["coordination"].coordinate(raw_actions, coord_obs)
        coordinated_actions = coord_result["resolved_actions"]
        coord_act = self.agents["coordination"].predict(coord_obs)
        self.message_log.append({
            "from": "coordination", "action": "coordination",
            "conflicts": len(coord_result["conflicts"]), "confidence": coord_act.confidence,
        })

        kpi = kpi_before or {}
        result = self.super_agent.validate_and_control(coordinated_actions, kpi, obs)

        monitoring = self.monitoring_svc.monitor_all()
        optimizer_actions = []
        if monitoring.get("optimization_required"):
            for obs_data in self.monitoring_svc.build_optimizer_observations(monitoring):
                opt_obs = AgentObservation(
                    timestamp=0, features=obs_data["features"], context=obs_data["context"],
                )
                opt_act = self.agents["agent_optimizer"].predict(opt_obs)
                if opt_act.parameters.get("optimization_action") != "monitor_only":
                    optimizer_actions.append(opt_act)
        monitoring_record = self.super_agent.monitor_and_optimize(monitoring, optimizer_actions)

        return {
            "raw_actions": [{"agent": a.agent_id, "type": a.action_type, "params": a.parameters} for a in raw_actions],
            "coordinated_actions": [
                {"agent": a.agent_id, "type": a.action_type, "params": a.parameters} for a in coordinated_actions
            ],
            "coordination": {
                "strategy": coord_result["coordination_strategy"],
                "conflicts_detected": len(coord_result["conflicts"]),
                "conflicts_resolved": len(coord_result["conflicts"]),
                "resolution_success_pct": coord_result["resolution_success_pct"],
                "conflicts": coord_result["conflicts"],
            },
            "approved_actions": result["approved_actions"],
            "super_agent_decision": result["decision"],
            "agent_monitoring": monitoring,
            "optimization_actions": [
                {"agent": a.agent_id, "type": a.action_type, "params": a.parameters} for a in optimizer_actions
            ],
            "monitoring_record": monitoring_record,
        }

    def coordinate(self, observation: AgentObservation) -> list[AgentAction]:
        result = self.run_all_agents(observation=observation)
        return result["approved_actions"]

    def deploy_oran_policies(self) -> dict[str, Any]:
        oran = load_json(project_root() / "config" / "oran_config.json")
        policies = []
        for xapp in oran["oran_architecture"]["near_rt_ric"]["xapps"]:
            agent = self.agents.get(xapp["agent"])
            if agent:
                policies.append({
                    "xapp": xapp["name"], "agent_id": agent.agent_id,
                    "status": "deployed", "control_actions": xapp["control_actions"],
                })
        return {"xapps_deployed": policies, "rapps": oran["oran_architecture"]["non_rt_ric"]["rapps"],
                "super_agent": self.super_agent.get_status()}

    def run_autonomous_loop(self, iterations: int = 10) -> dict:
        return self.closed_loop.run(iterations)

    def run_monitoring_cycle(self, kpi: dict | None = None) -> dict:
        """Super Agent monitors all agents and triggers Agent Optimizer for degraded ones."""
        monitoring = self.monitoring_svc.monitor_all()
        optimizer_actions = []
        for obs_data in self.monitoring_svc.build_optimizer_observations(monitoring):
            opt_obs = AgentObservation(
                timestamp=0, features=obs_data["features"], context=obs_data["context"],
            )
            opt_act = self.agents["agent_optimizer"].predict(opt_obs)
            if opt_act.parameters.get("optimization_action") != "monitor_only":
                optimizer_actions.append(opt_act)
        record = self.super_agent.monitor_and_optimize(monitoring, optimizer_actions)
        constraints = self.monitoring_svc.get_constraints_status(kpi)
        return {
            "monitoring": monitoring,
            "optimization_actions": [
                {"agent": a.agent_id, "type": a.action_type, "params": a.parameters} for a in optimizer_actions
            ],
            "monitoring_record": record,
            "constraints": constraints,
        }
