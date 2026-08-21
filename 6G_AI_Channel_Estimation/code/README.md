# 6G AI-Based Secure Channel Estimation

End-to-end StrongHER implementation from `6G_AI_CE_01.docx` and `Swetha_6G_AI_CE_Proj_01.pptx`.

## Quick start

```bash
cd Swetha_StrongHER_projects/6G_AI_Channel_Estimation/code
pip install -r requirements.txt
python scripts/run_end_to_end.py
python scripts/run_api_server.py
```

Open **http://localhost:8090/dashboard**

## What is included

| Item | Location |
|------|----------|
| ≥ 80,000-row 3GPP-aligned dataset | `data/datasets/channel_estimation_dataset.csv` |
| Train / validation / test (70/15/15) | `split` column + `outputs/reports/train_val_test_report.json` |
| 20 AI agents (PHY + security + coordinator + super) | `src/agents/` |
| Digital twin | `src/digital_twin/` |
| Plots (hist, CDF, heatmap, ROC, BER, twin) | `outputs/plots/` |
| Dashboard + chatbot | `static/dashboard/` + `src/api/` |
| CCFK dashboard | `ccfk-dashboard/` |
| Implementation document | `docs/6G_AI_CHANNEL_ESTIMATION_IMPLEMENTATION.md` |
| Slides | `docs/6G_AI_CE_E2E_Implementation.pptx` |
| 3GPP / Nokia / SharePoint / Confluence KB | `data/knowledge_base/` |

## Architecture

```
Intent / chatbot → Super Agent (control) → Coordinator (conflicts) → Orchestrator
  → Channel, CSI prediction/feedback, Pilot, Equalizer, Air-interface, Beam, Spectrum
  → Security, Mitigation, Self-healing, Mobility, Optimization, Resource, XAI, Knowledge
  → Digital Twin → O-RAN xApps/rApps
```

Baselines: **LS** and **MMSE**. AI: CNN-MLP, LSTM-GB, Transformer-MLP, GNN-RF ensemble.

## Knowledge sources

- 3GPP TR 38.901, TS 38.211, TS 38.214, TS 38.101-4, TR 38.843, TR 38.811
- Nokia System Insights CFAM RP003187 (DMRS channel estimation)
- SharePoint RAN1 R1-2506757 6G AI/ML air-interface use cases
- EE Confluence MCP (queried; no indexed hits on this client)
- CCFK component library
- agent-shim adapter (clone blocked by scm.cci.nokia.net auth)
