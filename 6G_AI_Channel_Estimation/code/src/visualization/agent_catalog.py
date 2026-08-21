"""Per-agent dashboard catalog: roles, related plots, train/val/test pickers."""

from __future__ import annotations

from src.common.utils import load_json, project_root

ROLES = {
    "channel": "Estimate H from DMRS/CSI-RS. Compare LS, MMSE, and the AI ensemble (NMSE, BER, SE).",
    "csi_prediction": "Forecast CSI quality so the air interface can reduce pilot overhead when prediction is confident.",
    "csi_feedback": "Set CSI report period and compression (TR 38.843) from predicted accuracy.",
    "pilot": "Increase or reduce DMRS/CSI-RS density from NMSE and mobility.",
    "equalizer": "Choose MMSE vs regularized MMSE from estimated CSI quality and predicted BER.",
    "air_interface": "Configure CSI-RS period, SRS, and PTRS (TS 38.211 / 38.214).",
    "beam": "Select analog/digital beam index from AoA, SNR, and load.",
    "spectrum": "Recommend carrier hold or frequency hop under jamming and THz blockage.",
    "security": "Detect CSI attacks: contamination, jamming, spoofing, poisoning, adversarial, backdoor.",
    "mitigation": "Apply the response: reassign pilots, switch beam, hop carrier, isolate, retrain.",
    "self_healing": "Fall back to MMSE and roll back the AI estimator after a CSI attack.",
    "mobility": "Predict handover and keep CSI consistent across cells.",
    "optimization": "Maximize spectral efficiency without violating NMSE or twin fidelity.",
    "resource": "Allocate PRBs vs predicted SE; do not boost untrusted UEs.",
    "digital_twin": "Score radio-twin fidelity and veto unsafe actuation.",
    "explainability": "Rank the features that drive NMSE (permutation importance).",
    "knowledge": "Map radio context to 3GPP / Nokia procedures (DMRS, CDL/TDL, CSI-RS).",
    "orchestrator": "Fuse domain intents into a candidate global policy.",
    "coordinator": "Detect and resolve conflicts among agents (security-first, NMSE guard, twin veto).",
    "super": "Control plane: approve/reject, enable/disable, weighted utility, twin gate.",
}

RELATED_PLOTS = {
    "channel": [
        "benchmark_ls_mmse_ai.png",
        "cdf_nmse_estimators.png",
        "ber_vs_snr.png",
        "bar_nmse_by_scenario.png",
        "heatmap_scenario_profile_nmse.png",
        "agents/channel_hist_cdf.png",
        "agents/channel_tvt.png",
    ],
    "csi_prediction": ["cdf_scenario_kpis.png", "agents/csi_prediction_hist_cdf.png", "agents/csi_prediction_tvt.png"],
    "csi_feedback": ["agents/csi_feedback_hist_cdf.png", "agents/csi_feedback_tvt.png"],
    "pilot": ["benchmark_ls_mmse_ai.png", "agents/pilot_hist_cdf.png", "agents/pilot_tvt.png"],
    "equalizer": ["ber_vs_snr.png", "agents/equalizer_hist_cdf.png", "agents/equalizer_tvt.png"],
    "air_interface": ["hist_dataset_splits.png", "agents/air_interface_hist_cdf.png", "agents/air_interface_tvt.png"],
    "beam": ["agents/beam_hist_cdf.png", "agents/beam_tvt.png"],
    "spectrum": ["heatmap_attack_nmse.png", "agents/spectrum_hist_cdf.png", "agents/spectrum_tvt.png"],
    "security": [
        "classification_attack_distribution.png",
        "classification_confusion_matrix.png",
        "classification_roc.png",
        "classification_scatter.png",
        "heatmap_attack_nmse.png",
        "agents/security_hist_cdf.png",
        "agents/security_tvt.png",
    ],
    "mitigation": ["classification_attack_distribution.png", "agents/mitigation_hist_cdf.png", "agents/mitigation_tvt.png"],
    "self_healing": ["heatmap_attack_nmse.png", "agents/self_healing_hist_cdf.png", "agents/self_healing_tvt.png"],
    "mobility": ["mobility_spatial_map.png", "agents/mobility_hist_cdf.png", "agents/mobility_tvt.png"],
    "optimization": ["architecture_scorecard.png", "agents/optimization_hist_cdf.png", "agents/optimization_tvt.png"],
    "resource": ["agents/resource_hist_cdf.png", "agents/resource_tvt.png"],
    "digital_twin": ["digital_twin_map.png", "digital_twin_timeseries.png", "agents/digital_twin_hist_cdf.png", "agents/digital_twin_tvt.png"],
    "explainability": ["heatmap_feature_correlation.png", "agents/explainability_tvt.png"],
    "knowledge": ["agents/knowledge_tvt.png"],
    "orchestrator": ["architecture_radar.png", "architecture_scorecard.png", "agents/orchestrator_tvt.png"],
    "coordinator": ["coordination_conflicts.png", "agents/coordinator_hist_cdf.png", "agents/coordinator_tvt.png"],
    "super": ["super_agent_control.png", "agents/super_hist_cdf.png", "agents/super_tvt.png"],
}

PLOT_CATEGORIES = {
    "Dataset": ["hist_dataset_splits.png", "cdf_scenario_kpis.png", "heatmap_feature_correlation.png", "heatmap_scenario_profile_nmse.png", "bar_nmse_by_scenario.png"],
    "Estimators": ["benchmark_ls_mmse_ai.png", "cdf_nmse_estimators.png", "ber_vs_snr.png", "model_training_curves.png", "model_train_val_test.png"],
    "Architecture": ["architecture_scorecard.png", "architecture_radar.png"],
    "Security": ["classification_attack_distribution.png", "classification_confusion_matrix.png", "classification_roc.png", "classification_scatter.png", "heatmap_attack_nmse.png"],
    "Twin & mobility": ["digital_twin_map.png", "digital_twin_timeseries.png", "mobility_spatial_map.png"],
    "Control": ["coordination_conflicts.png", "super_agent_control.png"],
}


def pick_tvt(name: str, metrics: dict, architecture: dict | None = None) -> dict:
    architecture = architecture or {}
    if name == "channel":
        return {
            "metric": "physics NMSE (held-out test)",
            "train": None,
            "validation": None,
            "test": architecture.get("test_nmse_ai"),
            "extra": {
                "LS": architecture.get("test_nmse_ls"),
                "MMSE": architecture.get("test_nmse_mmse"),
                "AI": architecture.get("test_nmse_ai"),
            },
            "lower_better": True,
        }
    for kind in ("accuracy", "r2", "mean_acc", "mean_accuracy", "ho_success", "mean_fidelity", "success_rate"):
        tr = metrics.get(f"train_{kind}")
        va = metrics.get(f"validation_{kind}")
        te = metrics.get(f"test_{kind}")
        if any(v is not None for v in (tr, va, te)):
            return {"metric": kind, "train": tr, "validation": va, "test": te, "lower_better": False}
    if metrics.get("binary_test_accuracy") is not None:
        return {
            "metric": "binary accuracy",
            "train": metrics.get("binary_train_accuracy"),
            "validation": metrics.get("binary_validation_accuracy"),
            "test": metrics.get("binary_test_accuracy"),
            "lower_better": False,
        }
    if metrics.get("multiclass_test_accuracy") is not None:
        return {
            "metric": "multiclass accuracy",
            "train": metrics.get("multiclass_train_accuracy"),
            "validation": metrics.get("multiclass_validation_accuracy"),
            "test": metrics.get("multiclass_test_accuracy"),
            "lower_better": False,
        }
    if metrics.get("cnn_test_r2") is not None:
        return {
            "metric": "ensemble R²",
            "train": metrics.get("ensemble_train_r2"),
            "validation": metrics.get("ensemble_validation_r2"),
            "test": metrics.get("ensemble_test_r2"),
            "lower_better": False,
        }
    if metrics.get("test_mean_nmse_ai") is not None:
        return {
            "metric": "test NMSE AI",
            "train": None,
            "validation": None,
            "test": metrics.get("test_mean_nmse_ai"),
            "lower_better": True,
        }
    keys = [k for k in metrics if k.startswith("importance_")]
    if keys:
        return {"metric": "feature importance", "train": None, "validation": None, "test": None, "importances": {k.replace("importance_", ""): metrics[k] for k in keys}}
    return {"metric": "status", "train": None, "validation": None, "test": None}


def existing_plots(rel_paths: list[str]) -> list[str]:
    root = project_root() / "outputs" / "plots"
    out = []
    for rel in rel_paths:
        if (root / rel).exists():
            out.append(rel)
    return out


def build_catalog(report: dict, last_run: dict | None = None) -> dict:
    cfg = load_json(project_root() / "config" / "agents_config.json")
    arch = report.get("architecture", {})
    last_run = last_run or {}
    actions = last_run.get("actions") or {}
    agents = []
    for item in cfg.get("agents", []):
        aid = item["id"]
        metrics = report.get("agents", {}).get(aid, {}).get("metrics", {})
        plots = existing_plots(RELATED_PLOTS.get(aid, []))
        action = actions.get(aid) or {}
        agents.append({
            "id": aid,
            "name": item.get("name", aid),
            "layer": item.get("layer", ""),
            "models": item.get("models", []),
            "role": ROLES.get(aid, ""),
            "trained": bool(report.get("agents", {}).get(aid, {}).get("trained")),
            "metrics": metrics,
            "tvt": pick_tvt(aid, metrics, arch),
            "plots": plots,
            "last_action": {
                "type": action.get("action_type"),
                "parameters": action.get("parameters") or {},
                "confidence": action.get("confidence"),
            } if action else {},
        })
    plots_root = project_root() / "outputs" / "plots"
    agent_pngs = sorted(p.name for p in (plots_root / "agents").glob("*.png")) if (plots_root / "agents").exists() else []
    return {
        "agents": agents,
        "count": len(agents),
        "architecture": arch,
        "layers": sorted({a["layer"] for a in agents}),
        "plot_categories": PLOT_CATEGORIES,
        "agent_plots": [f"agents/{n}" for n in agent_pngs],
        "last_policy": last_run.get("policy"),
    }
