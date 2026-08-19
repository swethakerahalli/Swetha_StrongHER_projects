"""Pydantic schemas for Autonomous RAN API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KPIStats(BaseModel):
    avg_throughput_mbps: float = 0.0
    avg_latency_ms: float = 0.0
    avg_cqi: float = 0.0
    avg_sinr_db: float = 0.0
    total_power_w: float = 0.0
    fairness_index: float = 1.0
    handover_success_rate: float = 1.0
    security_score: float = 1.0
    qoe_score: float = 4.0
    qos_sla_compliance: float = 1.0
    energy_efficiency: float = 0.0
    beamforming_gain_db: float = 0.0
    csi_accuracy: float = 0.0
    slice_efficiency: float = 0.0
    automation_level: float = 0.0
    global_utility: float = 0.0
    carbon_kg_co2_per_h: float = 0.0
    carbon_intensity_gco2_kwh: float = 380.0
    renewable_pct: float = 15.0
    prb_utilization_pct: float = 0.0
    traffic_congestion_pct: float = 0.0
    peak_traffic_mbps: float = 0.0
    timestamp: int = 0


class TargetKPIProgress(BaseModel):
    kpi_key: str
    label: str
    current_value: float
    target_value: float
    target_min: float | None = None
    target_max: float | None = None
    unit: str = "percent"
    achieved: bool = False
    progress_pct: float = 0.0


class AgentStatus(BaseModel):
    name: str
    agent_id: str
    ai_driven: bool = True
    ai_type: str = "sklearn"
    is_trained: bool = False
    model_loaded: bool = False


class KPIComparison(BaseModel):
    before: KPIStats
    after: KPIStats
    before_label: str = "Industry Baseline (Conventional RAN)"
    after_label: str = "Autonomous Intelligent RAN"
    delta: dict[str, float] = {}
    delta_pct: dict[str, float] = {}
    parameter_changes: list[dict] = []


class CellState(BaseModel):
    cell_id: str
    load: float = 0.5
    power_w: float = 400.0
    sleep: bool = False
    tx_power_dbm: float = 30.0
    prb_utilization: float = 0.5
    mimo_streams: int = 2
    beamforming_mode: str = "MRT"


class UEState(BaseModel):
    ue_id: str
    cell_id: str
    slice: str = "eMBB"
    cqi: float = 8.0
    sinr_db: float = 12.0
    throughput_mbps: float = 10.0
    latency_ms: float = 5.0
    mcs: int = 10
    prb_allocated: int = 5
    buffer_occupancy: float = 0.3


class ParameterUpdateRequest(BaseModel):
    cell_id: str | None = None
    ue_id: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    source: str = "manual"


class AgentRunRequest(BaseModel):
    intent: str = "Optimize throughput and reduce latency"
    cell_id: str = "CELL_000"
    ue_id: str | None = None
    agents: list[str] | None = None


class AgentRunResponse(BaseModel):
    action_id: str
    agents_invoked: list[str]
    actions: list[dict]
    kpi_before: KPIStats
    kpi_after: KPIStats
    parameter_updates: list[dict]
    super_agent_decision: dict = {}


class ClosedLoopRequest(BaseModel):
    iterations: int = 10


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    actions_taken: list[dict] = []
    kpi_snapshot: KPIStats | None = None
    llm_provider: str = ""
