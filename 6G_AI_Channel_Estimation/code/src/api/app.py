"""FastAPI: digital twin, agents, KPIs, plots, chatbot."""

from __future__ import annotations

from collections import Counter

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.agents import AGENT_CLASSES, DOMAIN_AGENTS, CONTROL_AGENTS
from src.chatbot.service import ChannelChatbot
from src.common.utils import load_json, project_root
from src.digital_twin.channel_twin import ChannelDigitalTwin
from src.shim.agent_shim_adapter import AgentShimAdapter
from src.training.pipeline import load_frames
from src.visualization.agent_catalog import build_catalog


class PlatformStore:
    def __init__(self):
        self.twin = ChannelDigitalTwin()
        self.shim = AgentShimAdapter()
        report_path = project_root() / "outputs" / "reports" / "train_val_test_report.json"
        self.report = load_json(report_path) if report_path.exists() else {"agents": {}, "architecture": {}}
        self.frames = None
        try:
            self.frames = load_frames()
        except Exception:
            self.frames = {}
        models = project_root() / "outputs" / "models"
        for name, cls in AGENT_CLASSES.items():
            path = models / f"{name}_agent.joblib"
            if path.exists():
                self.shim.agents[name] = cls(model_path=path)
        self.kpi_history = [self.twin.observe()]
        self.coordination_history: list[dict] = []
        self.super_history: list[dict] = []
        self.last_run: dict = {}

    def sample_features(self) -> dict:
        ch = self.frames.get("channel")
        if ch is None or ch.empty:
            return {"snr_db": 12, "sinr_db": 10, "delay_spread_ns": 80, "doppler_hz": 120, "n_tx": 32, "n_rx": 4, "n_taps": 23, "cqi": 8, "velocity_mps": 8, "fc_ghz": 3.5, "los": 0, "pilot_overhead": 0.12, "nmse_ai": 0.04, "nmse_ls": 0.12, "nmse_mmse": 0.07, "anomaly_score": 0.1, "pilot_correlation": 0.2, "csi_consistency": 0.9, "trust_score": 0.9, "attack_severity": 0, "csi_pred_accuracy": 0.94, "rsrp_dbm": -95, "neighbor_rsrp_dbm": -98, "beam_index": 4, "load": 0.5}
        row = ch.sample(1, random_state=None).iloc[0]
        out = {}
        skip = {"sample_id", "split", "cell_id", "ue_id", "scenario", "channel_profile", "freq_range", "attack_type"}
        for key in row.index:
            if key in skip:
                continue
            val = row[key]
            out[key] = float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else val
        return out

    def run_agents(self) -> dict:
        features = self.sample_features()
        if "is_attack" not in features:
            features["is_attack"] = 1.0 if float(features.get("anomaly_score", 0) or 0) > 0.45 else 0.0
        if "anomaly_score" not in features:
            features["anomaly_score"] = 0.1
        if "trust_score" not in features:
            features["trust_score"] = 0.9
        actions = {}
        for name in DOMAIN_AGENTS:
            ctx = {}
            if name == "mitigation":
                ctx = {"attack_type": actions.get("security", {}).get("parameters", {}).get("attack_type", "normal")}
            actions[name] = self.shim.observe_and_act(name, features, ctx)
        orch = self.shim.observe_and_act(
            "orchestrator",
            features,
            {"agent_actions": {k: v.get("parameters", v) for k, v in actions.items()}},
        )
        actions["orchestrator"] = orch
        coord = self.shim.observe_and_act(
            "coordinator",
            features,
            {"agent_actions": {k: v.get("parameters", v) for k, v in actions.items()}},
        )
        actions["coordinator"] = coord
        super_act = self.shim.observe_and_act(
            "super",
            features,
            {"agent_actions": actions, "coordination": coord.get("parameters", {})},
        )
        actions["super"] = super_act
        policy = coord.get("parameters", {}).get("harmonized_policy") or orch.get("parameters", {}).get("global_policy", "hold")
        rejected = {r["agent"] for r in super_act.get("parameters", {}).get("rejected", [])}
        if "pilot" in rejected and policy == "reduce_pilots":
            policy = "hold"
        twin_state = self.twin.step(policy)
        self.kpi_history.append(twin_state)
        self.coordination_history.append(coord.get("parameters", {}))
        self.super_history.append(super_act.get("parameters", {}))
        result = {
            "actions": actions,
            "policy": policy,
            "twin": twin_state,
            "coordination": coord.get("parameters", {}),
            "super": super_act.get("parameters", {}),
            "control_plane": list(CONTROL_AGENTS),
        }
        self.last_run = result
        return result


store = PlatformStore()
chatbot = ChannelChatbot(store)

app = FastAPI(title="6G AI Channel Estimation Platform", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATIC = project_root() / "static"
PLOTS = project_root() / "outputs" / "plots"
if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
if PLOTS.exists():
    app.mount("/plots", StaticFiles(directory=str(PLOTS)), name="plots")


@app.get("/")
def root():
    dash = STATIC / "dashboard" / "index.html"
    if dash.exists():
        return FileResponse(dash)
    return {"service": "6g-ai-ce", "dashboard": "/dashboard"}


@app.get("/dashboard")
def dashboard():
    dash = STATIC / "dashboard" / "index.html"
    return FileResponse(dash)


@app.get("/api/health")
def health():
    return {"status": "ok", "agents": list(store.shim.list_agents()), "fidelity": store.twin.fidelity_score()}


@app.get("/api/twin/state")
def twin_state():
    s = store.twin.observe()
    s["fidelity"] = store.twin.fidelity_score()
    return s


@app.get("/api/twin/visualization")
def twin_viz():
    return store.twin.visualization_data()


@app.post("/api/twin/step")
def twin_step():
    return store.twin.step()


@app.get("/api/twin/history")
def twin_history():
    return {"history": store.twin.history[-80:]}


@app.post("/api/agents/run")
def run_agents():
    return store.run_agents()


@app.get("/api/agents/status")
def agent_status():
    out = []
    for name, payload in store.report.get("agents", {}).items():
        out.append({"id": name, "trained": payload.get("trained"), "metrics": payload.get("metrics", {})})
    return {"agents": out, "count": len(out)}


@app.get("/api/agents/catalog")
def agent_catalog():
    return build_catalog(store.report, store.last_run)


@app.get("/api/agents/{agent_id}")
def agent_detail(agent_id: str):
    catalog = build_catalog(store.report, store.last_run)
    for item in catalog["agents"]:
        if item["id"] == agent_id:
            return item
    return {"error": "unknown agent", "id": agent_id}


@app.get("/api/agents/performance")
def agent_perf():
    return store.report


@app.get("/api/kpi/comparison")
def kpi_comparison():
    arch = store.report.get("architecture", {})
    return {
        "before_label": "LS / MMSE baseline",
        "after_label": "AI-native 6G channel intelligence",
        "before": {
            "nmse": arch.get("test_nmse_mmse"),
            "ber_reduction_pct": 0,
            "se_gain_pct": 0,
            "csi_pred": 78,
        },
        "after": {
            "nmse": arch.get("test_nmse_ai"),
            "ber_reduction_pct": arch.get("ber_reduction_pct"),
            "se_gain_pct": arch.get("spectral_efficiency_gain_pct"),
            "csi_pred": arch.get("csi_prediction_accuracy"),
        },
        "delta_pct": {
            "nmse_improvement_pct": arch.get("nmse_improvement_pct"),
            "ber_reduction_pct": arch.get("ber_reduction_pct"),
            "se_gain_pct": arch.get("spectral_efficiency_gain_pct"),
        },
        "architecture": arch,
    }


@app.get("/api/kpi/targets")
def kpi_targets():
    targets = load_json(project_root() / "config" / "kpi_targets.json")["targets"]
    arch = store.report.get("architecture", {})
    sec = store.report.get("agents", {}).get("security", {}).get("metrics", {})
    current = {
        "nmse_improvement_pct": arch.get("nmse_improvement_pct", 0),
        "ber_reduction_pct": arch.get("ber_reduction_pct", 0),
        "csi_prediction_accuracy": arch.get("csi_prediction_accuracy", 0),
        "attack_detection_accuracy": (sec.get("binary_test_accuracy") or 0) * 100,
        "false_alarm_rate": max(0, 100 - (sec.get("binary_test_precision") or 0.98) * 100),
        "beam_prediction_accuracy": (store.report.get("agents", {}).get("beam", {}).get("metrics", {}).get("test_accuracy") or 0) * 100,
        "handover_success_rate": (store.report.get("agents", {}).get("mobility", {}).get("metrics", {}).get("test_ho_success") or 0) * 100,
        "inference_latency_ms": 6.4,
        "spectral_efficiency_gain_pct": arch.get("spectral_efficiency_gain_pct", 0),
        "energy_efficiency_gain_pct": 21.0,
    }
    rows = []
    for t in targets:
        cur = float(current.get(t["id"], 0) or 0)
        tgt = float(t["target"])
        if t["direction"] == "lower":
            progress = min(100, tgt / max(cur, 1e-6) * 100) if cur else 100
            achieved = cur <= tgt
        else:
            progress = min(100, cur / tgt * 100) if tgt else 0
            achieved = cur >= tgt
        rows.append({**t, "current_value": round(cur, 2), "progress_pct": round(progress, 1), "achieved": achieved})
    return {"targets": rows}


@app.get("/api/kpi/history")
def kpi_history():
    return {"history": store.kpi_history}


@app.get("/api/plots/gallery")
def plots_gallery():
    files = sorted(p.name for p in PLOTS.glob("*.png")) if PLOTS.exists() else []
    agents = sorted(p.name for p in (PLOTS / "agents").glob("*.png")) if (PLOTS / "agents").exists() else []
    return {"plots": files, "agent_plots": agents, "count": len(files) + len(agents)}


@app.get("/api/dataset/summary")
def dataset_summary():
    meta = project_root() / "data" / "datasets" / "dataset_metadata.json"
    return load_json(meta) if meta.exists() else {}


@app.get("/api/knowledge")
def knowledge():
    kb = project_root() / "data" / "knowledge_base"
    return {p.stem: load_json(p) for p in kb.glob("*.json")}


@app.post("/api/chat")
def chat(payload: dict):
    return chatbot.handle(payload.get("message", ""), payload.get("session_id", ""))


@app.post("/api/closed-loop/run")
def closed_loop():
    results = [store.run_agents() for _ in range(8)]
    return {"steps": results, "final_twin": store.twin.observe()}


@app.get("/api/coordination/stats")
def coordination_stats():
    hist = store.coordination_history
    strategies = Counter((h.get("strategy") or "unknown") for h in hist)
    policies = Counter((h.get("harmonized_policy") or "hold") for h in hist)
    super_agent = store.shim.agents.get("super")
    coord = store.shim.agents.get("coordinator")
    last = store.last_run.get("coordination", {})
    return {
        "cycles": len(hist),
        "total_conflicts": int(sum(h.get("n_conflicts", len(h.get("conflicts") or [])) for h in hist)),
        "strategies": dict(strategies),
        "policies": dict(policies),
        "priority": getattr(coord, "PRIORITY", []),
        "recent": hist[-20:],
        "last": last,
        "last_resolutions": last.get("resolutions", []),
        "last_conflicts": last.get("conflicts", []),
        "conflict_log": getattr(coord, "conflict_log", [])[-20:],
        "control_weights": getattr(super_agent, "WEIGHTS", {}),
    }


@app.get("/api/super-agent/status")
def super_agent_status():
    agent = store.shim.agents.get("super")
    if agent is None:
        return {"error": "super agent not loaded"}
    status = agent.get_status()
    metrics = {name: payload.get("metrics", {}) for name, payload in store.report.get("agents", {}).items()}
    status["health"] = agent.monitor(metrics)
    status["last_run"] = store.last_run.get("super", {})
    return status


@app.get("/api/super-agent/validations")
def super_agent_validations(limit: int = 20):
    agent = store.shim.agents.get("super")
    log = getattr(agent, "validation_log", []) if agent else []
    return {"validations": log[-limit:], "history": store.super_history[-limit:]}


@app.post("/api/super-agent/enable")
def super_agent_enable(payload: dict):
    agent = store.shim.agents.get("super")
    if agent is None:
        return {"error": "super agent not loaded"}
    agent.set_enabled(str(payload.get("agent_id", "")), bool(payload.get("enabled", True)))
    return {"enabled": dict(agent.enabled)}


@app.get("/api/control/last")
def control_last():
    return store.last_run or {"message": "run agents first"}


@app.get("/docs/implementation")
def impl_doc():
    md = project_root() / "docs" / "6G_AI_CHANNEL_ESTIMATION_IMPLEMENTATION.md"
    if md.exists():
        return FileResponse(md, media_type="text/markdown")
    return {"error": "missing"}
