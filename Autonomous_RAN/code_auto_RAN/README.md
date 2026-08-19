# Multi-Agentic AI-Native Autonomous RAN and Air Interface for 6G Networks

End-to-end Python implementation based on `Autonomous_RAN_01.docx` and `Swetha_Autonomous_RAN_Proj_01.pptx`.

## Architecture

```
Intent (LLM) → Knowledge Graph → 8 AI Agents → Digital Twin → PHY Sim → O-RAN xApps/rApps → Closed-Loop
```

### AI Agents (8)
| Agent | Function | Model |
|-------|----------|-------|
| Scheduler | PRB allocation, QoS scheduling | Gradient Boosting |
| Resource | Spectrum, power, MIMO | Random Forest |
| Mobility | Handover prediction | Gradient Boosting Classifier |
| Security | Threat detection & mitigation | Isolation Forest + RF |
| Energy | Sleep mode, power optimization | Gradient Boosting |
| QoE | User experience prediction | Gradient Boosting |
| Knowledge | Root-cause analysis, KG reasoning | LLM + NetworkX |
| Intent | Operator intent → policy | LLM (Ollama / Nokia cache) |

### LLM Integration
Priority order (configurable in `config/llm_config.json`):
1. **Ollama** (local) — `http://localhost:11434` — install: `ollama pull llama3.2`
2. **Nokia System Insights cache** — CFAM/3GPP knowledge from `data/knowledge_base/`
3. **Rule-based** — ontology keyword matching

### External Knowledge Sources
| Source | File | Status |
|--------|------|--------|
| 3GPP specs | `data/knowledge_base/3gpp_references.json` | Curated TS 38.x, 28.x |
| O-RAN Alliance | `data/knowledge_base/oran_references.json` | RIC, E2SM-KPM/RC |
| Nokia CFAM (System Insights) | `data/knowledge_base/nokia_cfam_references.json` | OSS_FC_017307, SR003080 |
| SharePoint | `data/knowledge_base/sharepoint_references.json` | Project docs metadata |
| Confluence | `data/knowledge_base/confluence_references.json` | EE Confluence MCP |

## Dashboard, API & Chatbot

### Start the platform
```bash
# Terminal 1: API + Dashboard
python scripts/run_api_server.py
# Open http://localhost:8080/dashboard

# Terminal 2: Invoke APIs (before/after KPI demo)
python scripts/invoke_api_demo.py
```

### API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/twin/state` | GET | Digital Twin RAN state |
| `/api/agents/run` | POST | Super Agent runs all agents, applies RAN parameter changes |
| `/api/kpi/comparison` | GET | Before vs After KPI comparison |
| `/api/ran/cells/{id}` | PUT | Update cell parameters (power, load, sleep) |
| `/api/chat` | POST | RAN chatbot (Nokia LLM + API actions) |
| `/api/closed-loop/run` | POST | Observe→Act→Learn closed loop |
| `/api/super-agent/status` | GET | Super Agent validation status |

### Super Agent (Autonomous Intelligent RAN)
Controls and validates all 17 AI-driven agents:
`scheduler`, `resource`, `mobility`, `security`, `energy`, `qos`, `slice`, `qoe`,
`channel_estimation`, `beamforming`, `csi`, `air_interface`, `digital_twin`,
`spectrum`, `self_healing`, `knowledge`, `intent`

### Train / Validate / Test
```bash
python scripts/train_validate_test_agents.py   # All agents + per-agent plots
python scripts/generate_visualizations.py      # Global plots
```

### Optional: Enable Ollama LLM
```bash
# Install Ollama from https://ollama.com then:
ollama pull llama3.2
python scripts/run_end_to_end.py
```

### Individual Steps
```bash
python scripts/generate_datasets.py           # Synthetic RAN datasets
python scripts/fetch_external_knowledge.py    # Verify external KB files
python scripts/build_knowledge_base.py          # Knowledge graph + feature store
python scripts/train_agents.py                # Train ML agents
python scripts/generate_visualizations.py     # All plots, heatmaps, CDFs
python scripts/run_end_to_end.py              # Full pipeline
```

## Visualizations (`outputs/plots/`)

| Plot | Description |
|------|-------------|
| `hist_ran_kpi.png` | Histograms: CQI, SINR, throughput, latency, PRB, buffer |
| `hist_security.png` | Security feature histograms (normal vs attack) |
| `cdf_ran_by_slice.png` | CDFs of throughput, latency, SINR, RSRP per slice |
| `cdf_energy.png` | Power consumption CDF (active vs sleep) |
| `heatmap_ran_correlation.png` | RAN KPI feature correlation heatmap |
| `heatmap_mobility_correlation.png` | Mobility feature correlation heatmap |
| `heatmap_cell_slice_throughput.png` | Cell × slice throughput heatmap |
| `heatmap_slice_prb_hourly.png` | PRB utilization by slice × hour |
| `classification_security_confusion.png` | Security agent confusion matrix |
| `classification_threat_distribution.png` | Threat type bar chart |
| `classification_security_scatter.png` | Attack/normal feature scatter |
| `mobility_handover_analysis.png` | Handover success pie, type, delay hist |
| `mobility_trajectory_scatter.png` | UE trajectory scatter by velocity |
| `energy_analysis.png` | Power vs utilization scatter, per-cell bars |
| `model_training_metrics.png` | Per-agent training accuracy/R² |
| `model_federated_convergence.png` | Federated learning convergence |
| `simulation_timeseries.png` | Throughput/latency vs step per scheduler |
| `benchmark_comparison.png` | Bar chart benchmark KPIs |
| `benchmark_radar.png` | Radar chart multi-agent vs PF |
| `closed_loop_timeseries.png` | Digital twin closed-loop metrics |
| `phy_channel_simulation.png` | CSI heatmaps, SINR vs velocity |
| `federated_learning_rounds.png` | FL rounds overview |
| `dashboard_autonomous_ran.png` | Combined multi-agent dashboard |

## Directory Structure
```
code_auto_RAN/
├── config/              # system, KPI, agent, O-RAN, LLM configs (JSON)
├── data/
│   ├── datasets/        # Generated CSV datasets (saved locally)
│   └── knowledge_base/  # 3GPP, O-RAN, Nokia CFAM, SharePoint, Confluence, KG
├── src/
│   ├── agents/          # 8 AI agents
│   ├── llm/             # Ollama + Nokia cached LLM provider
│   ├── digital_twin/    # RAN digital twin
│   ├── knowledge_graph/ # Telecom ontology engine
│   ├── federated/       # FedAvg framework
│   ├── orchestration/   # Multi-agent controller + closed-loop
│   ├── simulation/      # RAN simulator, PHY channel, baselines
│   ├── oran/            # xApp/rApp JSON descriptors
│   └── benchmarks/      # KPI comparison framework
├── scripts/             # Runnable pipeline scripts
└── outputs/             # Models, reports, plots
```

## Simulations
- **RAN Simulator** — multi-agent vs RR/PF/Max-TP/SON baselines
- **PHY Channel Simulator** — CSI estimation, prediction, beamforming (TR 38.901 aligned)
- **Digital Twin** — policy validation before deployment (TS 28.104 aligned)

## All Data Saved Locally
All datasets, knowledge base JSON, trained models, and reports are persisted under `code_auto_RAN/data/` and `code_auto_RAN/outputs/`.
