# Multi-Agentic AI-Native Autonomous Intelligent RAN
## Complete Implementation Guide

**Author:** Swetha Kerahalli | **Organization:** Nokia MI RRD AS Algo Innov  
**Version:** 1.0.0 | **Date:** 2026-07-14  
**Dashboard:** http://localhost:8080/dashboard  
**API Docs:** http://localhost:8080/docs

---

## 1. Executive Summary

This project implements an end-to-end **Autonomous Intelligent RAN** platform with **17 AI-driven agents**, a **Digital Twin**, **REST API**, **live dashboard**, **Nokia MCP knowledge integration**, and **closed-loop optimization** aligned with 3GPP Rel-18/19, O-RAN Alliance, and Nokia CFAM features.

### Measured End-to-End Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Throughput (API demo) | 10.0 Mbps | 19.07 Mbps | **+90.7%** |
| Latency | 5.0 ms | 3.49 ms | **-30.2%** |
| Total Power | 2800 W | 1680 W | **-40.0%** |
| Multi-Agent vs Proportional Fair | 235.81 Mbps | 406.21 Mbps | **+72.3%** |
| Security Detection (multi-agent) | 8.7% | 100% | **+91.3 pp** |
| Super Agent Approvals | — | 17/17 agents | **100%** |

---

## 2. System Architecture

```
Operator Intent (LLM)
        │
        ▼
┌───────────────────┐     ┌─────────────────────┐
│  Knowledge Graph  │◄────│  Nokia MCP Sources   │
│  3GPP/O-RAN/CFAM  │     │  System Insights, SP │
└─────────┬─────────┘     └─────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│           17 AI Agents (sklearn + LLM)           │
│  scheduler, resource, mobility, security, energy │
│  qos, slice, qoe, channel_est, beamforming, csi  │
│  air_interface, digital_twin, spectrum, healing  │
│  knowledge, intent                               │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌───────────────────┐     ┌─────────────────────┐
│    Super Agent    │────►│  Digital Twin RAN    │
│  Validate/Control │     │  7 cells, 50 UEs     │
└─────────┬─────────┘     └──────────┬──────────┘
          │                          │
          ▼                          ▼
┌───────────────────┐     ┌─────────────────────┐
│  FastAPI REST API │     │  O-RAN xApps/rApps   │
│  KPI + Parameters │     │  Near-RT / Non-RT RIC│
└─────────┬─────────┘     └─────────────────────┘
          │
          ▼
┌───────────────────┐
│  Live Dashboard   │
│  + RAN Chatbot    │
└───────────────────┘
```

---

## 3. End-to-End Workflow (8 Phases)

| Phase | Script | Input | Output |
|-------|--------|-------|--------|
| 1. Dataset Generation | `scripts/generate_datasets.py` | `config/system_config.json` | 6 CSV files in `data/datasets/` |
| 2. Knowledge Base | `scripts/build_knowledge_base.py` | 3GPP/O-RAN/Nokia JSON + MCP | `data/knowledge_base/*.json` |
| 3. Agent Training | `scripts/train_validate_test_agents.py` | CSV datasets | `outputs/models/*.joblib` |
| 4. Digital Twin | `src/digital_twin/ran_twin.py` | Agent actions | Simulated KPI state |
| 5. API Invocation | `scripts/invoke_api_demo.py` | `POST /api/agents/run` | Before/after KPI JSON |
| 6. Benchmark | `src/benchmarks/benchmark_runner.py` | RAN simulator | `benchmark_results.csv` |
| 7. Visualizations | `scripts/generate_visualizations.py` | CSV + models | `outputs/plots/` (54+ plots) |
| 8. Dashboard | `scripts/run_api_server.py` | Live twin state | http://localhost:8080/dashboard |

**Full pipeline:** `python scripts/run_end_to_end.py`

---

## 4. Generated Datasets (CSV)

All datasets are synthetic, 3GPP-aligned, seed=42.

| CSV File | Rows | Key Columns | Used By Agents |
|----------|------|-------------|----------------|
| `ran_kpi_dataset.csv` | 80,000 | CQI/SINR/RSRP, 5QI/SST, DL/UL PRB, BLER, TA | scheduler, qos, qoe, beamforming, csi, air_interface |
| `mobility_traces.csv` | 80,000 | velocity, RSRP, A3/TTT, MRO late/early HO flags | mobility, channel_estimation |
| `security_events.csv` | 80,000 | packet_rate, spectrum anomaly, 5G-AKA, FM recovery | security, spectrum, self_healing |
| `energy_metrics.csv` | 80,000 | power, sleep, TX paths, EE bit/J, carbon | energy, resource, digital_twin |
| `slice_utilization.csv` | 80,000 | S-NSSAI, PRB, SLA, GFBR/MFBR, isolation | slice |
| `handover_events.csv` | 80,000 | source/target PCI, too-early/late, M8021C43 | mobility |

Full column dictionary: `data/datasets/DATASET_CATALOG.md`

**Regenerate:** `python scripts/generate_datasets.py`  
**Export inventory:** `python scripts/export_e2e_csv.py` → `outputs/reports/dataset_inventory.csv`

---

## 5. AI Agents — Data, Models, and Actions

All 17 agents are **AI-driven** (sklearn GradientBoosting/RandomForest/MLP or LLM).

### 5.1 RAN Control Agents

| Agent | Model | Training Data | Features | API Action Type |
|-------|-------|---------------|----------|-----------------|
| **scheduler** | GradientBoostingRegressor | ran_kpi | cqi, sinr_db, buffer, latency, mcs, prb | `schedule` → prb_assignment, mcs |
| **resource** | RandomForestRegressor | energy_metrics | utilization, traffic, power, renewable | `resource_allocation` → power, bandwidth, MIMO |
| **mobility** | GradientBoostingClassifier | mobility_traces | velocity, rsrp, handover_pending | `mobility` → handover recommendation |
| **energy** | GradientBoostingClassifier | energy_metrics | power, utilization, traffic, renewable | `energy` → sleep_mode, power_scale |
| **qos** | GradientBoostingClassifier | ran_kpi | latency, throughput, packet_loss, prb | `qos` → priority_boost per slice |
| **slice** | GradientBoostingClassifier | slice_utilization | prb_util, sla_compliance, latency_p99 | `slice` → prb_share, isolation |
| **qoe** | GradientBoostingRegressor | ran_kpi | throughput, latency, packet_loss | `qoe` → MOS estimate, optimization |

### 5.2 PHY / Air Interface Agents

| Agent | Model | Training Data | API Action Type |
|-------|-------|---------------|-----------------|
| **channel_estimation** | MLPRegressor | mobility + ran_kpi | `channel_estimation` → csi_accuracy |
| **beamforming** | RandomForestRegressor | ran_kpi | `beamforming` → mode, gain_db, num_beams |
| **csi** | GradientBoosting + PCA | ran_kpi | `csi` → compression_ratio, predicted throughput |
| **air_interface** | GradientBoostingRegressor | ran_kpi | `air_interface` → waveform, MCS, modulation |

### 5.3 Platform Agents

| Agent | Model | Training Data | API Action Type |
|-------|-------|---------------|-----------------|
| **digital_twin** | GradientBoostingRegressor | energy_metrics | `digital_twin` → fidelity, policy_validation |
| **spectrum** | GradientBoostingClassifier | security_events | `spectrum` → reallocate, bandwidth_mhz |
| **self_healing** | GradientBoostingClassifier | security_events | `self_healing` → recovery_action |
| **knowledge** | LLM + Knowledge Graph | KB JSON | RCA, policy references |
| **intent** | LLM (Ollama/Nokia cache) | operator text | parsed_intent, kpi_targets |

### 5.4 Super Agent

- **Role:** Validates all agent outputs, resolves conflicts, computes global utility
- **Weights:** 17 agents with weighted utility function from `config/kpis.json`
- **API:** `GET /api/super-agent/status`, `GET /api/super-agent/validations`

---

## 6. Model Training Pipeline

```bash
python scripts/train_validate_test_agents.py
```

**Process per agent:**
1. Load CSV dataset (see Section 4)
2. `agent.train(df)` — fit sklearn model
3. `agent.save(outputs/models/{name}_agent.joblib)`
4. Validate on 25% holdout — report accuracy/R² and avg confidence
5. Super Agent integration test — all 17 agents approved

**Training results (latest run):**
- scheduler R²=0.13 | resource R²=0.999 | mobility acc=1.0 | security acc=1.0
- energy acc=1.0 | qos acc=0.69 | slice acc=1.0 | qoe R²=0.97
- beamforming R²=0.999 | air_interface R²=0.13 | digital_twin R²=0.999
- spectrum acc=1.0 | self_healing acc=1.0
- Super Agent utility: **5.40**

**Export:** `outputs/reports/agent_training_results.csv`

---

## 7. API Reference

**Base URL:** `http://localhost:8080`

### 7.1 Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Service health + agent list |
| GET | `/api/twin/state` | Digital twin cells, UEs, fidelity |
| GET | `/api/kpi/current` | Live KPI snapshot |
| GET | `/api/kpi/comparison` | Baseline vs current KPI |
| GET | `/api/kpi/targets` | 9 target KPIs vs goals |
| GET | `/api/kpi/slices` | Per-slice throughput/latency |
| GET | `/api/e2e/summary` | Full E2E implementation results |
| GET | `/api/agents/status` | AI model loaded status per agent |
| POST | `/api/agents/run` | Run all agents + apply to twin |
| PUT | `/api/ran/cells/{id}` | Update cell power/load/sleep |
| PUT | `/api/ran/ues/{id}` | Update UE CQI/throughput/latency |
| POST | `/api/chat` | RAN chatbot (LLM + API actions) |
| POST | `/api/closed-loop/run` | Observe→Act→Learn loop |
| GET | `/api/knowledge/summary` | Nokia MCP KB status |

### 7.2 Agent Run Flow (`POST /api/agents/run`)

```json
{
  "intent": "Optimize throughput, reduce latency, improve energy",
  "cell_id": "CELL_000"
}
```

**Internal flow:**
1. `RANStateStore.snapshot_before()` — capture KPI baseline
2. `SuperAgentController.build_observation()` — feature vector
3. `IntentAgent` parses intent via LLM
4. All 17 agents `predict()` → `AgentAction` list
5. `SuperAgent.validate_and_control()` — approve/reject/modify
6. `RANParameterService._apply_to_twin()` — update UE/cell parameters
7. `RANDigitalTwin.step()` — simulate next state
8. `record_kpi()` — capture after KPI
9. Return `kpi_before`, `kpi_after`, `parameter_updates`, `super_agent_decision`

**Demo script:** `python scripts/invoke_api_demo.py`

---

## 8. Knowledge Base & Nokia MCP Integration

### 8.1 Knowledge Base Files

| File | Source | Content |
|------|--------|---------|
| `3gpp_references.json` | 3GPP TS 38.x, 28.x | Spec metadata + KPI definitions |
| `oran_references.json` | O-RAN Alliance | Architecture, E2SM-KPM, closed-loop |
| `nokia_cfam_references.json` | System Insights MCP | OSS_FC_017307, SR003080, SR001534 |
| `nokia_insights_cache.json` | System Insights `ask` | Autonomous RAN KPIs Rel-18/19 |
| `sharepoint_references.json` | SharePoint MCP | Project document metadata |
| `telecom_ontology.json` | Project ontology | Entities, threats, relationships |
| `knowledge_graph.json` | Builder | 175+ nodes (cells, UEs, agents, slices) |
| `feature_store_manifest.json` | Builder | Feature groups per agent |

### 8.2 Nokia MCP Servers

| MCP Server | Tools | Purpose |
|------------|-------|---------|
| **user-system-insights** | `ask`, `search_specs`, `read_spec_requirement` | 3GPP, CFAM, NIDD, O-RAN specs |
| **user-sharepoint** | `searchSharePoint`, `searchInSite` | Project documents |
| **user-EE CONFLUENCE MCP** | `confluence_search_content` | Team runbooks |
| **user-pronto-prod** | defect search (auth required) | Field RCA patterns |

**Refresh index:** `python scripts/fetch_external_knowledge.py`

---

## 9. Dashboard

**URL:** http://localhost:8080/dashboard

### Panels

1. **Target KPIs** — 9 goals from `config/kpis.json` with progress bars
2. **Operational KPI** — 15 live metrics before/after with delta %
3. **Slice KPIs** — eMBB, URLLC, mMTC breakdown
4. **AI Agent Status** — 17 agents, model loaded, sklearn/LLM type
5. **KPI Trend** — throughput/latency time series
6. **Digital Twin & Super Agent** — cells, UEs, fidelity, validations
7. **Nokia Knowledge Base** — MCP source status
8. **E2E Implementation Results** — pipeline phases, API demo metrics, benchmark table, CSV inventory, doc links
9. **Parameter Changes Log** — agent-driven RAN updates
10. **RAN Chatbot** — natural language control

**Start server:** `python scripts/run_api_server.py`

---

## 10. O-RAN Deployment

14 xApps configured in `config/oran_config.json`:
scheduler, resource, mobility, security, slice, energy, qos, beamforming, csi, air_interface, digital_twin, spectrum, self_healing

4 rApps: analytics, policy_management, federated_training, ai_model_management

**E2 interfaces:** E2SM-KPM (measurements), E2SM-RC (control)

---

## 11. Configuration Files

| File | Purpose |
|------|---------|
| `config/system_config.json` | 7 cells, 50 UEs, 3 slices (eMBB/URLLC/mMTC) |
| `config/agents_config.json` | All 17 agents, algorithms, I/O, ai_driven flag |
| `config/kpis.json` | 9 target KPIs + utility function weights |
| `config/oran_config.json` | xApps, rApps, E2SM metrics |
| `config/llm_config.json` | Ollama/Nokia LLM settings |

---

## 12. Output Artifacts

| Path | Description |
|------|-------------|
| `outputs/models/*.joblib` | 15 trained sklearn models |
| `outputs/plots/` | 22 global + 32 per-agent plots |
| `outputs/reports/agent_train_validate_test.json` | Full train/validate/test report |
| `outputs/reports/api_invocation_demo.json` | API before/after demo |
| `outputs/reports/benchmark_results.csv` | Scheduler comparison CSV |
| `outputs/reports/agent_training_results.csv` | Per-agent metrics CSV |
| `outputs/reports/api_kpi_before_after.csv` | KPI delta CSV |
| `outputs/reports/dataset_inventory.csv` | Dataset row/column counts |
| `outputs/reports/e2e_pipeline_phases.csv` | Pipeline phase status |
| `docs/AUTONOMOUS_RAN_IMPLEMENTATION.md` | This document |
| `docs/AUTONOMOUS_RAN_IMPLEMENTATION.docx` | Word document |
| `docs/AUTONOMOUS_RAN_IMPLEMENTATION.pdf` | PDF document |

---

## 13. Quick Start Commands

```bash
cd code_auto_RAN
pip install -r requirements.txt
pip install python-docx fpdf2

# Generate datasets (CSV)
python scripts/generate_datasets.py

# Train all 17 AI agents
python scripts/train_validate_test_agents.py

# Export E2E CSV reports
python scripts/export_e2e_csv.py

# Generate documentation (MD + DOCX + PDF)
python scripts/generate_documentation.py

# Start dashboard + API
python scripts/run_api_server.py
# Open: http://localhost:8080/dashboard

# API demo (before/after KPI)
python scripts/invoke_api_demo.py

# Full end-to-end pipeline
python scripts/run_end_to_end.py
```

---

## 14. Project Structure

```
code_auto_RAN/
├── config/           # system, agents, KPIs, O-RAN, LLM
├── data/
│   ├── datasets/     # 6 CSV files
│   └── knowledge_base/  # 3GPP, O-RAN, Nokia MCP JSON
├── docs/             # Implementation guide (MD, DOCX, PDF)
├── outputs/
│   ├── models/       # Trained .joblib models
│   ├── plots/        # Visualizations
│   └── reports/      # JSON + CSV results
├── scripts/          # Runnable pipelines
├── src/
│   ├── agents/       # 17 AI agents + Super Agent
│   ├── api/          # FastAPI app, schemas, E2E service
│   ├── chatbot/      # LLM chatbot
│   ├── digital_twin/ # RAN twin simulator
│   ├── orchestration/# Super Agent controller, closed loop
│   ├── simulation/   # RAN simulator, PHY, baselines
│   └── visualization/# Plot generators
└── static/dashboard/ # Live HTML dashboard
```

---

## 15. Standards Alignment

- **3GPP:** TS 38.214 (scheduling), TS 38.331 (mobility), TS 28.104 (NDT), TS 28.554 (KPIs), TS 28.100 (autonomy levels)
- **O-RAN:** Near-RT RIC xApps, Non-RT RIC rApps, E2SM-KPM/RC, A1 policies, SMO
- **Nokia CFAM:** OSS_FC_017307 (MRO), SR003080 (energy), SR001534 (self-healing)
- **ITU IMT-2030 / 6G:** AI-native RAN, multi-agent coordination, digital twin

---

*End of Implementation Guide*
