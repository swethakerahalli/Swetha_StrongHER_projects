"""Knowledge Agent - ontology reasoning with LLM augmentation."""

from __future__ import annotations

from typing import Any

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent
from src.common.utils import load_json, project_root
from src.knowledge_graph.kg_engine import KnowledgeGraphEngine
from src.llm.llm_provider import LLMProvider


class KnowledgeAgent(BaseAgent):
    def __init__(self, model_path=None):
        super().__init__("knowledge_agent", model_path)
        self.kg = KnowledgeGraphEngine()
        self.llm = LLMProvider()
        self.cfam = load_json(project_root() / "data" / "knowledge_base" / "nokia_cfam_references.json")

    def train(self, data: Any) -> dict[str, float]:
        self.is_trained = True
        return {"kg_nodes": self.kg.stats()["nodes"], "cfam_features": len(self.cfam.get("cfam_features", []))}

    def predict(self, observation: AgentObservation) -> AgentAction:
        cell_id = observation.context.get("cell_id", "CELL_000")
        threats = self.kg.get_threats_for_cell(cell_id)
        ues = self.kg.get_cell_ues(cell_id)

        query = (
            f"Cell {cell_id} has {len(ues)} UEs and {len(threats)} threats. "
            f"CQI={observation.features.get('cqi', 'N/A')}, "
            f"throughput={observation.features.get('throughput_mbps', 'N/A')} Mbps. "
            f"What root cause and agent coordination is recommended?"
        )
        llm_out = self.llm.generate(query)

        rca = []
        if observation.features.get("latency_ms", 0) > 20:
            rca.append({"cause": "high_latency", "feature_ref": "OSS_FC_017307"})
        if observation.features.get("spectrum_anomaly_score", 0) > 0.5:
            rca.append({"cause": "security_anomaly", "feature_ref": "SR001534"})

        return AgentAction(
            agent_id=self.agent_id,
            action_type="knowledge",
            parameters={
                "root_cause_analysis": rca,
                "kg_stats": self.kg.stats(),
                "connected_ues": len(ues),
                "active_threats": len(threats),
                "llm_insight": llm_out["text"][:500],
                "llm_provider": llm_out["provider"],
                "recommended_agents": self._recommend_agents(observation.features),
            },
            confidence=0.87,
        )

    def _recommend_agents(self, features: dict) -> list[str]:
        agents = []
        if features.get("cqi", 10) < 5:
            agents.append("resource")
        if features.get("latency_ms", 0) > 10:
            agents.append("scheduler")
        if features.get("spectrum_anomaly_score", 0) > 0.4:
            agents.append("security")
        if features.get("cell_utilization", 0.5) < 0.2:
            agents.append("energy")
        return agents or ["scheduler", "qoe"]
