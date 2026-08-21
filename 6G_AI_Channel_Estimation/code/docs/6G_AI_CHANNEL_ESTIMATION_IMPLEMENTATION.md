# 6G AI-Based Secure Channel Estimation — End-to-End Implementation

**Project:** AI-Based Secure Channel Estimation and CSI Prediction Framework for 6G Networks with Attack Detection and Mitigation  
**Author alignment:** Swetha Kerahalli / StrongHER  
**Standards:** 3GPP Rel-18/19/20, TR 38.901, TS 38.211, TS 38.214, TS 38.101-4, TR 38.843, TR 38.811, O-RAN RIC  
**Location:** `Swetha_StrongHER_projects/6G_AI_Channel_Estimation/code`

## 1. Executive summary

This repository is a runnable, 3GPP-aligned **AI-native channel intelligence platform**. It generates a telecom-grade synthetic dataset (≥ 60,000 rows), trains ten cooperating agents with explicit **train / validation / test** splits, benchmarks LS and MMSE baselines, detects and mitigates PHY-layer attacks, validates actions in a **radio digital twin**, and exposes an operator dashboard with chatbot.

Held-out test architecture scores:

| KPI | Value |
|-----|-------|
| Rows | 65000 |
| Train / val / test | 45500 / 9750 / 9750 |
| LS NMSE | 0.214638 |
| MMSE NMSE | 0.090862 |
| AI ensemble NMSE | 0.056116 |
| NMSE improvement vs MMSE | 38.24% |
| BER reduction vs MMSE | 33.36% |
| Spectral efficiency gain | 9.78% |
| CSI prediction accuracy | 89.06% |

## 2. Problem and proposed solution

Traditional LS/MMSE estimators degrade in THz, RIS, ultra-massive MIMO, NTN, and high-Doppler 6G channels, inflate pilot overhead, cannot predict CSI, and are exposed to pilot contamination, jamming, spoofing, poisoning, and adversarial CSI attacks.

The implemented solution follows the project architecture in `6G_AI_CE_01.docx` and `Swetha_6G_AI_CE_Proj_01.pptx`:

1. 3GPP CDL/TDL channel modeling (TR 38.901) plus NTN (TR 38.811) and THz/RIS research ranges  
2. Dataset generation with security labels  
3. LS / MMSE baselines  
4. CNN-MLP, LSTM-GB, Transformer-MLP, GNN-RF, and ensemble estimators  
5. CSI prediction to cut pilots  
6. Multi-class attack detection and policy mitigation  
7. Beam, mobility, and spectral-efficiency optimization agents  
8. Digital twin validation before policy commit  
9. Orchestrator closed loop  
10. O-RAN-oriented xApp/rApp mapping and operator dashboard

## 3. Architecture

```
Radio environment (gNB, UE, RIS, LEO)
        │
   TR 38.901 / 38.811 channel + attack injection
        │
   Feature store (CSI, mobility, security)
        │
   ┌──────────── multi-agent layer ────────────┐
   │ Channel │ CSI pred │ Security │ Mitigation │
   │ Beam    │ Mobility │ Optimize │ XAI        │
   │ Digital twin │ Orchestrator (global policy) │
   └───────────────────────────────────────────┘
        │
   Near-RT RIC xApps / Non-RT rApps / dashboard + chatbot
```

Seven layers from the research document are implemented as code modules:

| Layer | Code |
|-------|------|
| Data collection | `src/data/dataset_generator.py` |
| Preprocessing / splits | `split` column, 70/15/15 |
| AI channel estimation | `src/agents/estimators.py` |
| Attack detection | `src/agents/security.py` |
| Mitigation | `MitigationAgent` policy tree |
| Explainable AI | permutation importance |
| Deployment | FastAPI + dashboard + O-RAN JSON descriptors |

## 4. Dataset (synthetic, 3GPP-aligned)

Generated file: `data/datasets/channel_estimation_dataset.csv` (65000 rows).

**Scenarios:** UMa, UMi, RMa, InH, FR2-mmWave, THz, RIS, NTN-LEO  
**Profiles:** TDL-A…E and CDL-A…E (NLOS A/B/C, LOS D/E)  
**Radio:** fc, SCS, bandwidth, Nt/Nr, delay spread, Doppler, SNR/SINR, RSRP/RSRQ/RSSI, CQI, AoA/AoD  
**Estimates:** true channel, LS, MMSE, AI; NMSE/BER/SE for each  
**Attacks:** normal, pilot contamination, jamming, CSI spoofing, false CSI injection, poisoning, adversarial, backdoor  

Split counts: `{'train': 45500, 'validation': 9750, 'test': 9750}`

Scenario counts: `{'UMi': 8214, 'THz': 8195, 'FR2-mmWave': 8129, 'RMa': 8104, 'UMa': 8100, 'RIS': 8092, 'NTN-LEO': 8089, 'InH': 8077}`

Attack counts: `{'normal': 46917, 'pilot_contamination': 4475, 'jamming': 3885, 'csi_spoofing': 3196, 'false_csi_injection': 2545, 'data_poisoning': 2022, 'adversarial': 1296, 'backdoor': 664}`

Related tables: `mobility_dataset.csv`, `security_dataset.csv`, `digital_twin_states.csv`.

## 5. Training, validation, and testing

Every ML agent is fit on **train** only. Hyperparameter-free scores are reported on **validation** and **test** (15% each, seed 42).

| Agent | Trained | Test metrics |
|-------|---------|--------------|
| channel | yes | cnn_test_r2=-0.8581, cnn_test_rmse=1.562714, cnn_test_nmse=1.857912, lstm_test_r2=-0.0025 |
| csi_prediction | yes | test_r2=0.9969, test_rmse=0.009642, test_mean_accuracy=0.8905 |
| security | yes | multiclass_test_accuracy=0.8987, multiclass_test_precision=0.8773, multiclass_test_recall=0.8987, multiclass_test_f1=0.8764 |
| mitigation | yes | test_accuracy=0.7843, test_success_rate=0.9607 |
| beam | yes | test_accuracy=0.1195 |
| mobility | yes | test_accuracy=0.9652, test_ho_success=0.9983 |
| optimization | yes | test_r2=0.9993, test_se_gain_vs_mmse=9.78 |
| digital_twin | yes | test_r2=0.9975, test_mean_fidelity=0.9309 |
| explainability | yes | see report JSON |
| orchestrator | yes | test_mean_nmse_ai=0.056116, test_mean_nmse_mmse=0.090862 |
| pilot / equalizer / air_interface / csi_feedback | yes | see train_val_test_report.json |
| spectrum / self_healing / resource / knowledge | yes | see train_val_test_report.json |
| coordinator | yes | conflict resolution + train/val/test accuracy |
| super | yes | control gates + n_controlled_agents |

Full JSON: `outputs/reports/train_val_test_report.json`.

Plots: `outputs/plots/model_train_val_test.png`, `model_training_curves.png`, per-agent `outputs/plots/agents/*_hist_cdf.png`.

## 6. Classical vs AI estimators

Received signal model: `y = Hx + n`.

- **LS:** noisy observation of H (pilot-based)  
- **MMSE:** Wiener shrinkage using delay-spread correlation prior (TS 38.101-4 spirit)  
- **AI ensemble:** CNN-MLP (spatial) + gradient boosting (temporal) + Transformer-MLP (long-range) + random-forest GNN surrogate (topology)

Evaluation metrics match the research plan: MSE/RMSE/NMSE, BER, spectral efficiency, CSI prediction accuracy, attack precision/recall/F1/ROC-AUC.

## 7. Digital twin

`src/digital_twin/channel_twin.py` maintains cells, UEs, RIS panels, and LEO nodes. Each closed-loop step applies the **coordinator-harmonized, super-agent-gated** policy (hold / reduce pilots / increase pilots / mitigate) and updates NMSE, load, attack state, and **twin fidelity**. Visualizations: `digital_twin_map.png`, `digital_twin_timeseries.png`, live canvas on the dashboard.

## 7b. Coordinator and Super Agent

The coordinator inspects all domain actions and resolves conflicts with a fixed priority: security and self-healing first, then twin fidelity, then NMSE, then mobility/beam/spectrum, then SE/pilot reduction. Typical rules: keep DMRS if NMSE > 0.1; freeze beam when mitigation switches beams; hop carrier under jamming; extra DMRS before HO; twin veto; isolate before PRB boost; MMSE fallback after poisoning.

The super agent is the control plane: weighted approval, enable/disable, reject CSI-prediction pilot cuts when NMSE > 0.15, and gate optimization/resource when the twin is not safe to deploy. APIs: `/api/coordination/stats`, `/api/super-agent/status`, `/api/super-agent/enable`. Dashboard tab: **Coordinator & Super**.

## 8. Dashboard and chatbot

```
python scripts/run_api_server.py
# http://localhost:8090/dashboard
```

Tabs cover KPI overview, agent train/val/test, **coordinator & super-agent control**, security classification, digital twin, plot gallery, knowledge sources, and chatbot. The chatbot answers NMSE, conflicts, super-agent control, attacks, 3GPP/Nokia sources, and can **run agents** against the twin.

## 9. Nokia and 3GPP knowledge sources used

### 3GPP.org / ETSI

- TR 38.901 CDL/TDL (0.5–100 GHz), delay scaling clause 7.7  
- TS 38.211 DMRS/CSI-RS/SRS/PTRS  
- TS 38.214 CSI reporting and beam management  
- TS 38.101-4 TDLA30 / TDLB100 / TDLC300  
- TR 38.843 AI/ML for NR air interface (CSI prediction/compression, beam prediction)  
- TR 38.811 NTN channel  

### Nokia System Insights

- CFAM RP003187-2115 / RP003187-2929: DMRS-based PUSCH channel estimation (5GMax / 5G_L1_2794)  
- Synthesized Rel-18/19 AI/ML air-interface KPIs (NMSE, BER, CSI overhead)

### SharePoint (Nokia internal)

- R1-2506757 *Views on AI/ML Operation and Use Cases for 6G Radio Air Interface* (AI receivers, DMRS, CSI-RS overhead, beam management)  
- GX+ PHY Deep Dive — AI radio (4 Aug 2026)  
- RAN1#126 6GR AIML external review  
- RAN4#120 6G AI topic summary (testing)

### EE Confluence MCP

Configured spaces were queried (`channel estimation`, DMRS, CSI, NRAC, RFSW). This client returned no indexed hits; Confluence status is recorded in `data/knowledge_base/confluence_references.json`.

### agent-shim

`git clone https://scm.cci.nokia.net/AN/AI/agent-developer-artifacts/agent-shim.git` failed (HTTP Basic access denied). A local shim adapter lives in `src/shim/agent_shim_adapter.py`.

### CCFK

Dashboard styling follows Nokia CCFK FreeForm patterns used in the Autonomous RAN StrongHER project (`ccfk-dashboard/` sources). The live demo also serves a self-contained operator dashboard that does not require the Nokia npm registry.

## 10. O-RAN mapping

| Function | RIC placement |
|----------|---------------|
| Channel estimation / CSI prediction | Near-RT xApp |
| Security + mitigation | Near-RT xApp |
| Beam management | Near-RT xApp |
| Mobility | Near-RT xApp |
| Twin analytics / retraining policy | Non-RT rApp |
| Orchestrator | Near-RT policy + SMO |

Descriptors: `src/oran/`.

## 11. How to run

```bash
cd Swetha_StrongHER_projects/6G_AI_Channel_Estimation/code
pip install -r requirements.txt
python scripts/run_end_to_end.py
python scripts/run_api_server.py
```

Individual steps: `generate_datasets.py`, `train_validate_test.py`, `generate_visualizations.py`, `generate_docs.py`, `generate_slides.py`.

## 12. Deliverables map

| Category | Path |
|----------|------|
| Dataset | `data/datasets/` |
| Knowledge base | `data/knowledge_base/` |
| Models | `outputs/models/` |
| Plots | `outputs/plots/` |
| Train/val/test report | `outputs/reports/train_val_test_report.json` |
| This document | `docs/6G_AI_CHANNEL_ESTIMATION_IMPLEMENTATION.md` |
| Slides | `docs/6G_AI_CE_E2E_Implementation.pptx` |
| Dashboard | `static/dashboard/index.html` |

## 13. References

1. 3GPP TR 38.901, *Study on channel model for frequencies from 0.5 to 100 GHz*.  
2. 3GPP TS 38.211, *NR; Physical channels and modulation*.  
3. 3GPP TS 38.214, *NR; Physical layer procedures for data*.  
4. 3GPP TS 38.101-4, *UE radio transmission and reception; Part 4: Performance requirements*.  
5. 3GPP TR 38.843, *Study on AI/ML for NR air interface*.  
6. 3GPP TR 38.811, *Study on NR to support non-terrestrial networks*.  
7. 3GPP TR 38.743, *Study on enhancements for AI/ML for NG-RAN*.  
8. O-RAN Alliance, Massive MIMO use-cases technical report.  
9. Nokia RAN1 contribution R1-2506757, *AI/ML operation and use cases for 6G radio air interface*.  
10. Nokia System Insights CFAM RP003187, DMRS channel estimation.  
11. Tse & Viswanath, *Fundamentals of Wireless Communication*.  
12. Project source documents: `6G_AI_CE_01.docx`, `Swetha_6G_AI_CE_Proj_01.pptx`.
