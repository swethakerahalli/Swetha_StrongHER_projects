"""Optional Nokia agent-shim adapter.

agent-shim.git could not be cloned (scm.cci.nokia.net authentication failed).
This module exposes the same observe/act loop so a future shim client can wrap
the local agents without changing orchestration.
"""

from __future__ import annotations

from typing import Any

from src.agents import AGENT_CLASSES, AgentObservation


class AgentShimAdapter:
    """Drop-in local shim: register agents, observe features, return actions."""

    def __init__(self):
        self.agents = {name: cls() for name, cls in AGENT_CLASSES.items()}

    def list_agents(self) -> list[str]:
        return list(self.agents.keys())

    def observe_and_act(self, agent_id: str, features: dict[str, float], context: dict[str, Any] | None = None) -> dict[str, Any]:
        agent = self.agents[agent_id]
        obs = AgentObservation(timestamp=0, features=features, context=context or {})
        action = agent.predict(obs)
        return {
            "agent_id": action.agent_id,
            "action_type": action.action_type,
            "parameters": action.parameters,
            "confidence": action.confidence,
            "shim": "local-fallback",
        }
