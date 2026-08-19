"""Multi-Agent Controller - coordinates all autonomous RAN agents."""

from __future__ import annotations

from typing import Any

from src.agents import (
    EnergyAgent, IntentAgent, KnowledgeAgent, MobilityAgent, QoEAgent,
    ResourceAgent, SchedulerAgent, SecurityAgent,
)
from src.agents.base_agent import AgentAction, AgentObservation
from src.common.utils import load_config, load_json, project_root, save_json
from src.orchestration.closed_loop import ClosedLoopController


class MultiAgentController:
    def __init__(self):
        self.agents = {
            "scheduler": SchedulerAgent(),
            "resource": ResourceAgent(),
            "mobility": MobilityAgent(),
            "security": SecurityAgent(),
            "energy": EnergyAgent(),
            "qoe": QoEAgent(),
            "knowledge": KnowledgeAgent(),
            "intent": IntentAgent(),
        }
        self.llm_status = self._check_llm()
        self.cfg = load_config("agents_config.json")
        self.closed_loop = ClosedLoopController(self.agents)
        self.message_log: list[dict] = []

    def _check_llm(self) -> dict:
        from src.llm.llm_provider import LLMProvider
        llm = LLMProvider()
        return {"ollama_available": llm.is_ollama_available(), "fallback": "nokia_cached_or_rule_based"}

    def coordinate(self, observation: AgentObservation) -> list[AgentAction]:
        actions = []
        for name, agent in self.agents.items():
            act = agent.predict(observation)
            actions.append(act)
            self.message_log.append({
                "from": name, "to": "controller",
                "action": act.action_type, "confidence": act.confidence,
            })
        return self._resolve_conflicts(actions)

    def _resolve_conflicts(self, actions: list[AgentAction]) -> list[AgentAction]:
        energy = next((a for a in actions if a.action_type == "energy"), None)
        sched = next((a for a in actions if a.action_type == "schedule"), None)
        if energy and sched:
            if energy.parameters.get("sleep_mode") and sched.parameters.get("prb_assignment", 0) > 10:
                sched.parameters["prb_assignment"] = max(1, sched.parameters["prb_assignment"] // 2)
        return actions

    def deploy_oran_policies(self) -> dict[str, Any]:
        oran = load_json(project_root() / "config" / "oran_config.json")
        policies = []
        for xapp in oran["oran_architecture"]["near_rt_ric"]["xapps"]:
            agent = self.agents.get(xapp["agent"])
            if agent:
                policies.append({
                    "xapp": xapp["name"],
                    "agent_id": agent.agent_id,
                    "status": "deployed",
                    "control_actions": xapp["control_actions"],
                })
        return {"xapps_deployed": policies, "rapps": oran["oran_architecture"]["non_rt_ric"]["rapps"]}

    def run_autonomous_loop(self, iterations: int = 10) -> dict:
        return self.closed_loop.run(iterations)

    def export_state(self, path) -> None:
        save_json({
            "agents": list(self.agents.keys()),
            "messages": len(self.message_log),
            "last_actions": self.message_log[-10:] if self.message_log else [],
        }, path)
