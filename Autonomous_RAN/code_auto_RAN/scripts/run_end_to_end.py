#!/usr/bin/env python3
"""End-to-end Autonomous RAN pipeline with LLM, external KB, and simulations."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.benchmarks.benchmark_runner import BenchmarkRunner
from src.common.utils import project_root, save_json
from src.data.dataset_generator import generate_datasets
from src.data.knowledge_base_builder import build_knowledge_base
from src.knowledge_graph.kg_engine import KnowledgeGraphEngine
from src.llm.llm_provider import LLMProvider
from src.orchestration.multi_agent_controller import MultiAgentController
from src.simulation.phy_channel_sim import PHYChannelSimulator


def main():
    print("=" * 60)
    print("Multi-Agentic AI-Native Autonomous RAN - End-to-End Demo")
    print("=" * 60)

    llm = LLMProvider()
    print(f"\n[LLM] Ollama available: {llm.is_ollama_available()}")
    print(f"[LLM] Fallback: Nokia cached knowledge + rule-based")

    print("\n[Phase 1] Generating synthetic datasets...")
    ds_paths = generate_datasets()
    print(f"  Generated {len(ds_paths)} dataset files")

    print("\n[Phase 2] Building knowledge base from 3GPP/O-RAN/Nokia CFAM/SharePoint...")
    kb_paths = build_knowledge_base()
    kg = KnowledgeGraphEngine()
    print(f"  Knowledge graph: {kg.stats()['nodes']} nodes, {kg.stats()['edges']} edges")

    print("\n[Phase 3] Training AI agents...")
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "scripts" / "train_agents.py")], check=True)

    print("\n[Phase 4] PHY channel simulation (AI-native air interface)...")
    phy = PHYChannelSimulator()
    csi_history = []
    for i in range(5):
        ch = phy.generate_csi(f"UE_{i:04d}", velocity=10.0)
        csi_history.append(ch.csi)
    pred = phy.predict_future_csi(csi_history)
    print(f"  CSI prediction shape: {pred.shape}, beamforming norm: {float(abs(phy.beamforming_weights(pred)).sum()):.3f}")

    print("\n[Phase 5] Intent management via LLM...")
    intent_result = llm.parse_intent("Reduce energy consumption by 20% while maintaining URLLC latency below 5ms")
    print(f"  Intent provider: {intent_result['llm_provider']}, primary agent: {intent_result['primary_agent']}")

    print("\n[Phase 6] Multi-agent closed-loop in digital twin...")
    controller = MultiAgentController()
    loop_result = controller.run_autonomous_loop(iterations=15)
    print(f"  Twin fidelity: {loop_result['twin_fidelity']}")

    print("\n[Phase 7] O-RAN xApps/rApps deployment (simulated)...")
    oran = controller.deploy_oran_policies()
    print(f"  Deployed {len(oran['xapps_deployed'])} xApps")

    print("\n[Phase 8] Benchmark vs RR/PF/Max-TP/SON...")
    benchmark = BenchmarkRunner()
    bench_result = benchmark.run_all(steps=80)
    ma = next(r for r in bench_result["results"] if r["scheduler"] == "multi_agent_autonomous")
    print(f"  Multi-agent: {ma['avg_throughput_mbps']} Mbps, {ma['avg_latency_ms']} ms latency")

    print("\n[Phase 9] Generating visualizations (histograms, CDFs, heatmaps, classification)...")
    from src.visualization.plot_generator import generate_all_plots
    viz_manifest = generate_all_plots()
    print(f"  Generated {viz_manifest['count']} plots in outputs/plots/")

    summary = {
        "project": "Multi-Agentic AI-Native Autonomous RAN",
        "phases_completed": 9,
        "llm": {"ollama": llm.is_ollama_available(), "intent": intent_result},
        "datasets": {k: str(v) for k, v in ds_paths.items()},
        "knowledge_base": {k: str(v) for k, v in kb_paths.items()},
        "phy_simulation": {"csi_history_len": len(csi_history)},
        "closed_loop": loop_result,
        "oran_deployment": oran,
        "benchmark": bench_result,
        "visualizations": viz_manifest,
        "external_sources": ["3gpp", "oran", "nokia_cfam", "sharepoint", "confluence"],
    }
    out = project_root() / "outputs" / "reports" / "end_to_end_summary.json"
    save_json(summary, out)
    print(f"\n[Complete] Summary: {out}")
    print("=" * 60)


if __name__ == "__main__":
    main()
