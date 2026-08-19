"""FastAPI application: Digital Twin RAN API, KPI, agents, chatbot."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.agent_monitoring_service import AgentMonitoringService
from src.api.agent_performance_service import AgentPerformanceService
from src.api.coordination_service import CoordinationService
from src.api.resource_carbon_service import ResourceCarbonService
from src.api.e2e_service import E2EResultsService
from src.api.ran_parameter_service import RANParameterService
from src.api.schemas import (
    AgentRunRequest, AgentRunResponse, ChatRequest, ChatResponse,
    ClosedLoopRequest, KPIComparison, KPIStats, ParameterUpdateRequest,
)
from src.api.state_store import RANStateStore
from src.chatbot.chatbot_service import ChatbotService
from src.common.utils import load_json, project_root

app = FastAPI(
    title="Autonomous RAN Digital Twin API",
    description="Multi-Agent AI-Native RAN — Digital Twin, KPI, Parameter Control",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = RANStateStore.get()
param_svc = RANParameterService(store)
e2e_svc = E2EResultsService()
agent_perf_svc = AgentPerformanceService(store.baseline_svc)
agent_monitoring_svc = AgentMonitoringService(agent_perf_svc)
resource_carbon_svc = ResourceCarbonService(store)
coordination_svc = CoordinationService(store)
chatbot = ChatbotService(param_svc)

STATIC_DIR = project_root() / "static"
PLOTS_DIR = project_root() / "outputs" / "plots"
CCFK_DASHBOARD_DIR = STATIC_DIR / "ccfk-dashboard"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if PLOTS_DIR.exists():
    app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")
CCFK_DIR = project_root().parent / "ccfk" / "source" / "AdvancedTheme"
if CCFK_DIR.exists():
    app.mount("/ccfk-theme", StaticFiles(directory=str(CCFK_DIR)), name="ccfk-theme")


# ── Health ──────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "autonomous-ran-api", "agents": list(store.controller.agents.keys())}


# ── Digital Twin ────────────────────────────────────────────────────
@app.get("/api/twin/state")
def twin_state():
    state = store.twin.observe()
    state["fidelity"] = store.twin.fidelity_score()
    return state


@app.post("/api/twin/step")
def twin_step():
    before = store.snapshot_before()
    state = store.twin.step()
    after = store.compute_kpi()
    return {"state": state, "kpi_before": before.model_dump(), "kpi_after": after.model_dump()}


@app.get("/api/twin/history")
def twin_history(limit: int = 50):
    return {"history": store.twin.history[-limit:]}


@app.get("/api/twin/visualization")
def twin_visualization():
    """Live topology: cells, UEs, slice colors for dashboard canvas."""
    return store.twin.visualization_data()


# ── RAN Parameters ──────────────────────────────────────────────────
@app.get("/api/ran/cells")
def get_cells():
    return {"cells": [c.model_dump() for c in param_svc.get_all_cells()]}


@app.get("/api/ran/cells/{cell_id}")
def get_cell(cell_id: str):
    cells = {c.cell_id: c for c in param_svc.get_all_cells()}
    if cell_id not in cells:
        raise HTTPException(404, f"Cell {cell_id} not found")
    return cells[cell_id].model_dump()


@app.put("/api/ran/cells/{cell_id}")
def update_cell(cell_id: str, params: dict):
    try:
        result = param_svc.update_cell(cell_id, params)
        store.twin.step()
        kpi = store.compute_kpi()
        return {"update": result, "current_kpi": kpi.model_dump()}
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.get("/api/ran/ues")
def get_ues(limit: int = 20):
    return {"ues": [u.model_dump() for u in param_svc.get_all_ues(limit)]}


@app.put("/api/ran/ues/{ue_id}")
def update_ue(ue_id: str, params: dict):
    try:
        result = param_svc.update_ue(ue_id, params)
        store.twin.step()
        kpi = store.compute_kpi()
        return {"update": result, "current_kpi": kpi.model_dump()}
    except ValueError as e:
        raise HTTPException(404, str(e))


@app.post("/api/ran/parameters")
def update_parameters(req: ParameterUpdateRequest):
    return param_svc.apply_manual_update(req)


@app.get("/api/ran/parameters/changes")
def parameter_changes(limit: int = 50):
    return {"changes": store.parameter_changes[-limit:]}


@app.get("/api/e2e/summary")
def e2e_summary():
    return e2e_svc.get_summary()


@app.get("/docs/implementation")
def implementation_doc():
    md = project_root() / "docs" / "AUTONOMOUS_RAN_IMPLEMENTATION.md"
    if md.exists():
        return FileResponse(md, media_type="text/markdown")
    raise HTTPException(404, "Documentation not found")


# ── KPI ───────────────────────────────────────────────────────────
@app.get("/api/kpi/current", response_model=KPIStats)
def kpi_current():
    kpi = store.compute_kpi()
    store.current_kpi = kpi
    return kpi


@app.get("/api/kpi/baseline", response_model=KPIStats)
def kpi_baseline():
    return store.baseline_svc.industry_kpi()


@app.post("/api/kpi/baseline/reset")
def reset_baseline():
    kpi = store.reset_baseline()
    return {"baseline": kpi.model_dump(), "label": "Industry Baseline (Conventional RAN)"}


@app.get("/api/kpi/comparison", response_model=KPIComparison)
def kpi_comparison():
    cmp = store.get_industry_comparison()
    return KPIComparison(
        before=cmp["before"], after=cmp["after"],
        before_label=cmp["before_label"], after_label=cmp["after_label"],
        delta=cmp["delta"], delta_pct=cmp["delta_pct"],
        parameter_changes=store.parameter_changes[-10:],
    )


@app.get("/api/kpi/runtime-improvement")
def kpi_runtime_improvement():
    return store.baseline_svc.runtime_improvement_series(store.kpi_history)


@app.get("/api/kpi/benchmark")
def kpi_benchmark():
    return {"schedulers": store.baseline_svc.benchmark_comparison()}


@app.get("/api/plots/gallery")
def plots_gallery():
    plots = store.baseline_svc.list_plots()
    categorized = AgentPerformanceService.categorize_plots(plots)
    return {"plots": plots, "count": len(plots), **categorized}


@app.get("/api/agents/performance")
def agents_performance():
    """Per-agent KPI improvement vs industry baseline + validation metrics."""
    return agent_perf_svc.get_comparison()


@app.get("/api/resource/allocation")
def resource_allocation():
    return resource_carbon_svc.resource_allocation_stats()


@app.get("/api/carbon/stats")
def carbon_stats():
    return resource_carbon_svc.carbon_stats()


@app.get("/api/traffic/stats")
def traffic_stats():
    return resource_carbon_svc.traffic_stats()


@app.get("/api/coordination/stats")
def coordination_stats():
    return coordination_svc.coordination_stats()


@app.get("/api/green-agents/stats")
def green_agents_stats():
    return resource_carbon_svc.green_agents_stats()


@app.get("/api/super-agent/monitoring")
def super_agent_monitoring():
    """Per-agent health monitoring with degradation detection and optimization triggers."""
    kpi = store.compute_kpi().model_dump()
    return store.controller.run_monitoring_cycle(kpi)


@app.get("/api/constraints/status")
def constraints_status():
    """RAN operational constraints — QoS, power, energy, security, mobility, resource."""
    kpi = store.compute_kpi().model_dump()
    return agent_monitoring_svc.get_constraints_status(kpi)


@app.get("/api/super-agent/optimizations")
def super_agent_optimizations(limit: int = 20):
    return {
        "optimizations": store.controller.super_agent.optimization_log[-limit:],
        "monitoring_cycles": store.controller.super_agent.monitoring_log[-limit:],
    }


@app.get("/api/kpi/history")
def kpi_history():
    return {"history": store.kpi_history}


@app.get("/api/kpi/targets")
def kpi_targets():
    return {"targets": [t.model_dump() for t in store.compute_target_kpis()]}


@app.get("/api/kpi/slices")
def kpi_by_slice():
    slices: dict[str, dict] = {}
    for ue in store.twin.ues.values():
        sl = slices.setdefault(ue.slice, {"throughput": [], "latency": [], "count": 0})
        sl["throughput"].append(ue.throughput_mbps)
        sl["latency"].append(ue.latency_ms)
        sl["count"] += 1
    result = {}
    for name, data in slices.items():
        result[name] = {
            "avg_throughput_mbps": round(float(np.mean(data["throughput"])), 2),
            "avg_latency_ms": round(float(np.mean(data["latency"])), 2),
            "ue_count": data["count"],
        }
    return {"slices": result}


@app.get("/api/agents/status")
def agents_status():
    return {"agents": store.controller.get_agent_status(), "count": len(store.controller.agents)}


@app.get("/api/knowledge/summary")
def knowledge_summary():
    kb = project_root() / "data" / "knowledge_base"
    files = ["3gpp_references.json", "oran_references.json", "nokia_cfam_references.json",
             "nokia_insights_cache.json", "sharepoint_references.json", "telecom_ontology.json"]
    summary = {}
    for f in files:
        path = kb / f
        if path.exists():
            data = load_json(path)
            summary[f] = {"exists": True, "entries": len(data) if isinstance(data, list) else len(data.keys())}
        else:
            summary[f] = {"exists": False}
    return {"knowledge_base": summary, "mcp_sources": ["system-insights", "sharepoint", "confluence", "pronto-prod"]}


# ── Agents ────────────────────────────────────────────────────────
@app.post("/api/agents/run", response_model=AgentRunResponse)
def run_agents(req: AgentRunRequest):
    result = param_svc.run_agents_and_apply(intent=req.intent, cell_id=req.cell_id)
    return AgentRunResponse(
        action_id=result["action_id"],
        agents_invoked=result["agents_invoked"],
        actions=result["actions"],
        kpi_before=result["kpi_before"],
        kpi_after=result["kpi_after"],
        parameter_updates=result["parameter_updates"],
        super_agent_decision=result.get("super_agent_decision", {}),
    )


@app.get("/api/agents/actions")
def agent_actions(limit: int = 20):
    return {"actions": store.action_log[-limit:]}


@app.get("/api/super-agent/status")
def super_agent_status():
    return store.controller.super_agent.get_status()


@app.get("/api/super-agent/validations")
def super_agent_validations(limit: int = 20):
    return {"validations": store.controller.super_agent.validation_log[-limit:]}


# ── Closed Loop ───────────────────────────────────────────────────
@app.post("/api/closed-loop/run")
def closed_loop(req: ClosedLoopRequest):
    before = store.snapshot_before()
    result = store.controller.run_autonomous_loop(iterations=req.iterations)
    after = store.compute_kpi()
    return {
        "result": result,
        "kpi_before": before.model_dump(),
        "kpi_after": after.model_dump(),
        "comparison": store.compare_kpi(before, after),
    }


# ── Chatbot ───────────────────────────────────────────────────────
@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    return chatbot.handle_message(req.message, req.session_id)


# ── Dashboard ─────────────────────────────────────────────────────
@app.get("/")
def root_redirect():
    return dashboard_hub()


@app.get("/dashboard")
def dashboard_hub():
    hub = STATIC_DIR / "dashboard" / "hub.html"
    if hub.exists():
        return FileResponse(hub)
    return dashboard_classic()


@app.get("/dashboard/classic")
def dashboard_classic():
    index = STATIC_DIR / "dashboard" / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(404, "Classic dashboard not found")


@app.get("/dashboard/ccfk")
@app.get("/dashboard/ccfk/")
def dashboard_ccfk():
    index = CCFK_DASHBOARD_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    raise HTTPException(404, "CCFK dashboard not found — see ccfk-dashboard/README.md")


@app.get("/dashboard/legacy")
def dashboard_redirect():
    """Backward-compatible alias for classic dashboard."""
    return dashboard_classic()
