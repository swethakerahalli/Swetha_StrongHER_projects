"""Agent registry."""

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent
from src.agents.control import CoordinatorAgent, SuperAgent
from src.agents.estimators import CHANNEL_FEATURES, CSIPredictionAgent, ChannelEstimationAgent
from src.agents.phy_extra import (
    AirInterfaceAgent,
    CsiFeedbackAgent,
    EqualizerAgent,
    KnowledgeAgent,
    PilotAgent,
    ResourceAgent,
    SelfHealingAgent,
    SpectrumAgent,
)
from src.agents.ran_agents import (
    BeamAgent,
    DigitalTwinAgent,
    ExplainabilityAgent,
    MobilityAgent,
    OptimizationAgent,
    OrchestratorAgent,
)
from src.agents.security import MitigationAgent, SecurityAgent

AGENT_CLASSES = {
    "channel": ChannelEstimationAgent,
    "csi_prediction": CSIPredictionAgent,
    "csi_feedback": CsiFeedbackAgent,
    "pilot": PilotAgent,
    "equalizer": EqualizerAgent,
    "air_interface": AirInterfaceAgent,
    "beam": BeamAgent,
    "spectrum": SpectrumAgent,
    "security": SecurityAgent,
    "mitigation": MitigationAgent,
    "self_healing": SelfHealingAgent,
    "mobility": MobilityAgent,
    "optimization": OptimizationAgent,
    "resource": ResourceAgent,
    "digital_twin": DigitalTwinAgent,
    "explainability": ExplainabilityAgent,
    "knowledge": KnowledgeAgent,
    "orchestrator": OrchestratorAgent,
    "coordinator": CoordinatorAgent,
    "super": SuperAgent,
}

DOMAIN_AGENTS = [
    "channel", "csi_prediction", "csi_feedback", "pilot", "equalizer", "air_interface",
    "beam", "spectrum", "security", "mitigation", "self_healing", "mobility",
    "optimization", "resource", "digital_twin", "explainability", "knowledge",
]

CONTROL_AGENTS = ["orchestrator", "coordinator", "super"]

__all__ = [
    "AGENT_CLASSES",
    "CONTROL_AGENTS",
    "DOMAIN_AGENTS",
    "AgentAction",
    "AgentObservation",
    "BaseAgent",
    "CHANNEL_FEATURES",
]
