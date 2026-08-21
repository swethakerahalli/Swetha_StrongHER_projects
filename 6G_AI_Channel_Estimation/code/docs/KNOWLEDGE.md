# Knowledge sources used for 6G AI channel estimation

## 3GPP.org / ETSI

| Spec | Role in this project |
|------|----------------------|
| TR 38.901 | CDL-A/E, TDL-A/E, delay-spread scaling |
| TS 38.211 | DMRS, CSI-RS, SRS, PTRS for estimation |
| TS 38.214 | CSI reporting, CQI/PMI/RI, beam management |
| TS 38.101-4 | TDLA30 / TDLB100 / TDLC300 performance profiles |
| TR 38.843 | Rel-18 AI/ML CSI prediction, beam prediction, LCM |
| TR 38.811 | NTN-TDL delay profiles (LEO) |
| TR 38.743 | AI/ML enhancements for NG-RAN |

## Nokia System Insights

- CFAM RP003187-2115 / RP003187-2929 — DMRS-based PUSCH channel estimation (5GMax / 5G_L1_2794)
- Synthesized Rel-18/19 AI/ML air-interface evaluation (NMSE, BER, CSI overhead)

## SharePoint (Nokia internal)

- R1-2506757 *Views on AI/ML Operation and Use Cases for 6G Radio Air Interface*
- GX+ PHY Deep Dive — AI radio (4 Aug 2026)
- RAN1#126 6GR AIML external review
- RAN4#120 6G AI topic summary

## EE Confluence MCP

Configured spaces were queried. This client returned no indexed hits; status is in `data/knowledge_base/confluence_references.json`.

## CCFK and agent-shim

- CCFK dashboard sources: `ccfk-dashboard/` (Nokia CSF FreeForm)
- agent-shim clone from `scm.cci.nokia.net` failed authentication; local adapter: `src/shim/agent_shim_adapter.py`
