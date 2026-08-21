# Agent card — 6G AI channel estimation

Closed loop: **domain PHY/security/RAN agents → orchestrator → coordinator (conflict resolution) → super agent (control) → digital twin**.

## Control plane

| Agent | Task | Model | Split usage |
|-------|------|-------|-------------|
| coordinator | Detect and resolve conflicts (pilot density vs NMSE, beam vs mitigation, hop vs SE, HO vs CSI, twin veto, isolate vs PRB, MMSE fallback) | Priority rules + GB classifier | Fit train; accuracy + conflict rate on val/test |
| super | Enable/disable, approve/reject, weighted utility, twin fidelity gate | Control policy + GB classifier | Fit train; accuracy + n_controlled_agents |
| orchestrator | Propose a global policy from fused agent intents | Fusion rules | Test NMSE improvement |

**Coordinator priority:** security / mitigation / self-healing > twin fidelity > channel NMSE > mobility > beam > spectrum > SE / pilot reduction.

**Super-agent gates:** reject CSI-prediction pilot cuts if NMSE > 0.15; block optimization and resource if the twin is not safe to deploy.

## Channel-estimation domain agents

| Agent | Task | Model | Split usage |
|-------|------|-------|-------------|
| channel | Estimate H | CNN-MLP + GB LSTM + Transformer-MLP + RF GNN ensemble | Fit train; score val/test R²/RMSE/NMSE |
| csi_prediction | Forecast CSI accuracy | Gradient boosting | Fit train; R² on val/test |
| csi_feedback | CSI report period / compression (TR 38.843) | Gradient boosting | Fit train; R² on val/test |
| pilot | DMRS / CSI-RS density | Gradient boosting | Fit train; accuracy on val/test |
| equalizer | MMSE vs regularized MMSE | Gradient boosting | Fit train; R² of BER on val/test |
| air_interface | CSI-RS / SRS / PTRS configuration (TS 38.211/38.214) | Gradient boosting | Fit train; accuracy on val/test |
| beam | Beam index | Random forest | Accuracy val/test |
| spectrum | Frequency hop under jamming / blockage | Gradient boosting | Accuracy val/test |
| security | Attack class | Random forest + IsolationForest | Accuracy/F1/ROC on val/test |
| mitigation | Response policy | GB classifier + rule table | Accuracy + success rate |
| self_healing | MMSE fallback / model rollback after CSI attacks | Gradient boosting | Accuracy val/test |
| mobility | Handover pending | GB classifier | Accuracy + HO success |
| optimization | Spectral efficiency | GB regressor | R² + SE gain vs MMSE |
| resource | PRB boost vs SE | GB regressor | R² on val/test |
| digital_twin | Fidelity / safe-to-deploy | GB regressor | R² + mean fidelity |
| explainability | Feature ranking | Permutation importance | Importances on train sample |
| knowledge | Map radio context to 3GPP / Nokia procedures | Standards lookup | Profile/scenario coverage on test |

All trained artifacts: `outputs/models/*_agent.joblib`.
