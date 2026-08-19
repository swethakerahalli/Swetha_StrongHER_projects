"""Intent Agent - operator intent to policy translation using LLM."""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent
from src.llm.llm_provider import LLMProvider


class IntentAgent(BaseAgent):
    def __init__(self, model_path=None):
        super().__init__("intent_agent", model_path)
        self.llm = LLMProvider()
        self.active_intents: list[dict] = []

    def train(self, data: Any) -> dict[str, float]:
        self.is_trained = True
        return {"intents_supported": 5}

    def predict(self, observation: AgentObservation) -> AgentAction:
        intent_text = observation.context.get(
            "operator_intent",
            "Maintain URLLC latency below 5ms and reduce energy consumption by 20%",
        )
        parsed = self.llm.parse_intent(intent_text)
        self.active_intents.append(parsed)

        kpi_targets = {}
        intent_l = intent_text.lower()
        if "energy" in intent_l or "carbon" in intent_l:
            kpi_targets["energy_reduction_pct"] = 20
        if "latency" in intent_l or "urllc" in intent_l:
            kpi_targets["latency_ms"] = 1 if "1 ms" in intent_l else 5
        if "throughput" in intent_l:
            kpi_targets["throughput_improvement_pct"] = 25

        return AgentAction(
            agent_id=self.agent_id,
            action_type="intent",
            parameters={
                "parsed_intent": parsed,
                "kpi_targets": kpi_targets,
                "policy_distribution": "A1_to_near_rt_ric",
                "primary_agent": parsed["primary_agent"],
            },
            confidence=0.91,
        )
