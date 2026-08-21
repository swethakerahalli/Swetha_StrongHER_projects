# Autonomous RAN dataset catalog

Each CSV in `data/datasets/` has **80,000 samples**. Values are synthetic but column names, units, and ranges follow **3GPP**, Nokia **System Insights / CFAM (NIDD PM)**, **SharePoint** 5G KPI packs, and **EE Confluence** fallbacks (CFAM cache when MCP timed out).

Existing agent feature columns are unchanged so training scripts keep working. New columns are additive.

Regenerate: `python scripts/generate_datasets.py`

Topology used for IDs: **21 cells** (7 gNB × 3 sectors), **200 UEs**, PLMN `24407`, NR ARFCN `633334` (n78 / 3.5 GHz), 100 MHz, μ=1 (30 kHz SCS), 273 PRBs.

---

## Sources

| Source | What was used |
|--------|----------------|
| **3GPP** | TS 38.215 / 38.214 / 38.213 / 38.321 (L1/L2 measurements, MCS, HARQ, TA, PHR); TS 38.331 / 38.133 / 38.304 (RRC, HO events A1–A5, TTT, mobility state); TS 28.552 / 28.554 (PRB used, throughput, EE KPIs); TS 23.501 / 23.503 (S-NSSAI SST/SD, 5QI, PDB, GFBR/MFBR, AMBR); TS 33.501 (5G-AKA, NIA/NEA); TS 28.310 (energy saving) |
| **System Insights / CFAM** | `OSS_FC_017307` 5G MRO (late/early/wrong/ping-pong HO); `RP001677` HO PM **M8021C43**; `SR003080` TX-path switching for EE; `SR001534` BTS autonomous recovery (DSP/Radio/BTS reset) |
| **SharePoint** | *5G counters and KPIs, v8.0.pptx* (RRM-CA); *NPN 5G Radio KPI Configuration*; *Analytic for 5G Slices*; *Swetha_Autonomous_RAN_Proj_01.pptx*; *Multi-agent green telecom carbon-aware RAN* |
| **EE Confluence MCP** | Search timed out; same CFAM + project docs used as documented in `confluence_references.json` |

Slice mapping (TS 23.501):

| Slice | SST | SD | 5QI | PDB (ms) | Typical use |
|-------|-----|----|-----|----------|-------------|
| eMBB | 1 | 000001 | 9 | 100 | Mobile broadband |
| URLLC | 2 | 000002 | 82 | 5 | Time-critical / industrial |
| mMTC | 3 | 000003 | 9 | 300 | Massive IoT |

---

## 1. `ran_kpi_dataset.csv`

**What it is:** Per-UE, per-TTI-like radio and QoS samples for scheduling, CSI, beamforming, QoE, and air-interface agents. One row = one UE observation in one cell.

**Grain:** UE × time × serving cell  
**Rows:** 80,000

| Column | Unit / type | What it specifies | Spec / source |
|--------|-------------|-------------------|---------------|
| `timestamp` | step index | Simulation time bucket | internal |
| `ue_id` | ID | UE identity (`UE_0000`…`UE_0199`) | TS 38.331 |
| `cell_id` | ID | Serving NR cell | TS 38.300 |
| `gnb_id` | ID | Serving gNB (3 cells per gNB) | TS 38.300 |
| `pci` | 0–1007 | Physical Cell Identity | TS 38.211 |
| `plmn_id` | MCC+MNC | Serving PLMN | TS 23.003 |
| `nr_arfcn` | ARFCN | NR-ARFCN of serving carrier (n78) | TS 38.101 |
| `slice` | eMBB/URLLC/mMTC | Network slice name | TS 23.501 |
| `sst` | 1/2/3 | Slice/Service Type | TS 23.501 |
| `sd` | hex | Slice Differentiator | TS 23.501 |
| `fiveqi` | 5QI | 5G QoS Identifier | TS 23.501 Table 5.7.4-1 |
| `qfi` | 1–3 | QoS Flow Identifier | TS 23.501 |
| `pdb_ms` | ms | Packet Delay Budget for the 5QI | TS 23.501 |
| `drb_id` | int | Data Radio Bearer ID | TS 38.331 |
| `rnti` | 1–65519 | C-RNTI | TS 38.321 |
| `rrc_state` | connected/inactive/idle | RRC state | TS 38.331 |
| `cqi` | 0–15 | Channel Quality Indicator (legacy agent feature) | TS 38.214 |
| `wideband_cqi` | 0–15 | Wideband CQI | TS 38.214 / SharePoint 5G KPI pack |
| `ri` | 1–4 | Rank Indicator (MIMO layers reported) | TS 38.214 |
| `pmi` | 0–15 | Precoding Matrix Indicator | TS 38.214 |
| `mcs` | 0–28 | Modulation and Coding Scheme index | TS 38.214 |
| `mcs_table` | 64qam/256qam | MCS table in use | TS 38.214 |
| `sinr_db` | dB | DL SINR (legacy) | TS 38.215 |
| `ul_sinr_db` | dB | UL SINR | TS 38.215 / CFAM SR003080 |
| `rsrp_dbm` | dBm | CSI/CRS-like RSRP (legacy) | TS 38.215 |
| `rsrq_db` | dB | RSRQ | TS 38.215 |
| `ss_rsrp_dbm` | dBm | SS-RSRP | TS 38.215 |
| `ss_sinr_db` | dB | SS-SINR | TS 38.215 |
| `rssi_dbm` | dBm | RSSI | TS 38.215 |
| `pathloss_db` | dB | Coupling loss / path loss | TS 38.213 |
| `prb_allocated` | PRBs | Scheduled PRBs this sample (legacy) | TS 38.214 |
| `dl_prb_util` | 0–1 | DL PRB used / available (cell) | TS 28.552 DRB.PrbUsedDl / SharePoint KPI pack |
| `ul_prb_util` | 0–1 | UL PRB used / available | TS 28.552 DRB.PrbUsedUl |
| `buffer_occupancy` | 0–1 | UE BSR / RLC buffer fill (legacy) | TS 38.321 |
| `throughput_mbps` | Mbps | Combined UL+DL user throughput (legacy target) | TS 28.552 / 28.554 |
| `dl_throughput_mbps` | Mbps | DL DRB throughput | TS 28.552 |
| `ul_throughput_mbps` | Mbps | UL DRB throughput | TS 28.552 |
| `latency_ms` | ms | User-plane latency (legacy) | TS 28.554 |
| `jitter_ms` | ms | Delay variation | TS 28.554 / slice analytics |
| `packet_loss` | 0–1 | Loss ratio (legacy) | TS 28.554 |
| `bler` | 0–1 | PDSCH residual BLER | TS 38.321 HARQ |
| `harq_retx` | 0–7 | HARQ retransmission count | TS 38.321 |
| `ta_us` | µs | Timing advance (distance proxy) | TS 38.213 |
| `phr_db` | dB | Power Headroom Report | TS 38.321 / 38.213 |
| `se_bps_hz` | bit/s/Hz | Spectral efficiency | TS 28.554 |
| `rrc_connected_ues` | count | RRC connected UEs in cell | SharePoint 5G counters / NIDD PM |
| `bwp_id` | 0–3 | Bandwidth part ID | TS 38.213 |
| `beam_id` | 0–7 | SSB / CSI-RS beam index | TS 38.214 Rel-18 beam mgmt |
| `mu_numerology` | 1 | NR numerology μ | TS 38.211 |
| `bandwidth_mhz` | MHz | Carrier bandwidth | TS 38.104 |

**Used by:** scheduler, qos, qoe, traffic, csi, beamforming, air_interface, channel_estimation, edge_inference, coordination, agent_optimizer.

---

## 2. `mobility_traces.csv`

**What it is:** UE trajectory + neighbour measurement context for mobility prediction and MRO. One row = one UE measurement occasion.

**Grain:** UE × time  
**Rows:** 80,000

| Column | Unit / type | What it specifies | Spec / source |
|--------|-------------|-------------------|---------------|
| `timestamp` | step | Measurement time | internal |
| `ue_id` | ID | UE | TS 38.331 |
| `cell_id` | ID | Serving cell (legacy) | TS 38.300 |
| `serving_pci` | 0–1007 | Serving PCI | TS 38.211 |
| `neighbor_cell_id` | ID | Best neighbour cell | TS 38.331 |
| `neighbor_pci` | 0–1007 | Best neighbour PCI | TS 38.331 |
| `x_m`, `y_m` | m | Planar position in the twin | TR 38.901 |
| `distance_to_site_m` | m | Distance to serving site | TR 38.901 |
| `velocity_mps` | m/s | Speed (legacy) | TS 38.304 mobility state |
| `direction_deg` | deg | Heading (legacy) | TR 38.901 |
| `mobility_state` | normal/medium/high | RRC mobility state | TS 38.304 |
| `indoor_flag` | 0/1 | Indoor vs outdoor | TR 38.901 |
| `neighbor_cells` | JSON list | Three neighbour cell IDs (legacy) | TS 38.331 measConfig |
| `rsrp_dbm` | dBm | Serving RSRP (legacy) | TS 38.215 |
| `rsrq_db` | dB | Serving RSRQ | TS 38.215 |
| `sinr_db` | dB | Serving SINR | TS 38.215 |
| `ss_rsrp_dbm` | dBm | Serving SS-RSRP | TS 38.215 |
| `neighbor_rsrp_dbm` | dBm | Best neighbour RSRP | TS 38.331 event A3 |
| `a3_offset_db` | dB | A3 offset | TS 38.331 |
| `hysteresis_db` | dB | Meas hysteresis | TS 38.331 |
| `ttt_ms` | ms | Time-To-Trigger | TS 38.331 |
| `s_measure_dbm` | dBm | s-Measure (start neighbour meas) | TS 38.331 |
| `meas_event` | A1/A2/A3/A4/A5/none | Triggered report event | TS 38.331 |
| `meas_id` | 1–31 | Measurement identity | TS 38.331 |
| `handover_pending` | 0/1 | HO likely this sample (legacy label) | agent target |
| `rlf_detected` | 0/1 | Radio link failure | TS 38.331 / 38.133 |
| `ping_pong_flag` | 0/1 | Ping-pong HO risk | CFAM OSS_FC_017307 |
| `cfam_late_ho_ind` | 0/1 | Late HO indicator | CFAM RP001677 M8021C43 |
| `cfam_early_ho_ind` | 0/1 | Early HO indicator | CFAM OSS_FC_017307 |
| `cell_reselection_priority` | 0–7 | Idle reselection priority | TS 38.304 |
| `beam_id` | 0–7 | Serving SSB beam | TS 38.214 |

**Used by:** mobility, channel_estimation.

---

## 3. `security_events.csv`

**What it is:** Security, jamming, and fault-management events for threat detection and self-healing. One row = one event.

**Grain:** event  
**Rows:** 80,000 (~82% normal / 18% attack)

| Column | Unit / type | What it specifies | Spec / source |
|--------|-------------|-------------------|---------------|
| `event_id` | ID | Unique event (`SEC_000000`…) | internal |
| `timestamp` | step | Event time | internal |
| `source_ip` | IPv4 | Observed source (N2/N3 / UE path proxy) | TS 33.501 |
| `target_cell` | ID | Affected cell (legacy) | TS 38.300 |
| `target_pci` | PCI | Affected PCI | TS 38.211 |
| `threat_type` | enum | `normal`, `jamming`, `spoofing`, `ddos`, `rogue_gnb`, `pilot_contamination` | TS 33.501 / air-interface threat model |
| `is_attack` | 0/1 | Attack label (legacy target) | agent target |
| `severity` | info/minor/major/critical | FM severity | CFAM SR001534 / FTM |
| `packet_rate_pps` | pkt/s | Observed packet rate (legacy) | flow analytics |
| `auth_failures` | count | Failed authentications (legacy) | TS 33.501 5G-AKA |
| `nas_integrity_fail` | 0/1 | NAS integrity check fail | TS 33.501 NIA |
| `nas_auth_fail` | 0/1 | NAS / 5G-AKA failure | TS 33.501 |
| `fiveg_aka_result` | success/fail | Primary authentication result | TS 33.501 |
| `integrity_alg` | NIA1/2/3 | NAS/AS integrity algorithm | TS 33.501 |
| `cipher_alg` | NEA0/2/3 | NAS/AS ciphering algorithm | TS 33.501 |
| `spectrum_anomaly_score` | 0–1 | RF anomaly score (legacy) | PHY jamming / SharePoint KPI ops |
| `flow_entropy` | 0–1 | Flow randomness (legacy); low in attacks | security agent |
| `jammer_power_dbm` | dBm | Estimated jammer / interference power | PHY |
| `affected_prb_count` | PRBs | PRBs hit by jammer | TS 38.214 |
| `rogue_pci` | PCI or -1 | Fake gNB PCI if `rogue_gnb` | TS 33.501 |
| `bytes_transferred` | bytes | Bytes in the flow/window | internal |
| `ngap_cause` | string | NGAP release / failure cause | TS 38.413 |
| `fm_alarm_id` | enum | Nokia-style FM alarm class | CFAM SR001534 |
| `cfam_recovery_action` | enum | Autonomous recovery action | CFAM SR001534 / FTM-4194 |
| `detection_confidence` | 0–1 | Detector confidence | internal |
| `ts_33501_control` | SEAF/AUSF or N/A | 5G security function involved | TS 33.501 |

**Used by:** security, spectrum, self_healing.

---

## 4. `energy_metrics.csv`

**What it is:** Per-cell energy, sleep, and carbon samples for energy, resource, carbon, and digital-twin agents.

**Grain:** cell × time  
**Rows:** 80,000

| Column | Unit / type | What it specifies | Spec / source |
|--------|-------------|-------------------|---------------|
| `timestamp` | step | Sample time | internal |
| `cell_id` | ID | Cell (legacy) | TS 38.300 |
| `gnb_id` | ID | gNB | TS 38.300 |
| `hour_of_day` | 0–23 | Hour (diurnal load / solar) | carbon-aware RAN whitepaper |
| `power_consumption_w` | W | Cell RF+baseband power (legacy) | TS 28.554 EE |
| `pa_power_dbm` | dBm | PA output power | TS 38.104 / SR003080 |
| `cell_utilization` | 0–1 | Load (legacy) | TS 28.552 PRB used |
| `sleep_state` | 0/1 | Cell in energy-saving sleep (legacy) | TS 28.310 |
| `deep_sleep_flag` | 0/1 | Deep vs micro sleep | TS 28.310 |
| `num_active_tx_paths` | 0–16 | Active TX branches | CFAM SR003080 |
| `carrier_shutdown` | 0/1 | SCell / carrier off | TS 28.310 |
| `renewable_pct` | % | Renewable share of supply (legacy) | carbon-aware RAN SP doc |
| `carbon_intensity_gco2_kwh` | gCO2/kWh | Grid carbon intensity (legacy) | carbon-aware RAN |
| `traffic_demand_mbps` | Mbps | Offered cell traffic (legacy) | TS 28.552 |
| `energy_per_bit_uj` | µJ/bit | Energy per bit | TS 28.554 |
| `ee_bit_per_joule` | bit/J | Energy efficiency | TS 28.554 |
| `grid_power_w` | W | Power drawn from grid | site EE |
| `battery_soc_pct` | % | Backup battery state of charge | site EE |
| `temperature_c` | °C | RRH / cabinet temperature | FM / EE |
| `rectifier_load_pct` | % | Rectifier load | site EE |
| `cfam_tx_path_switch` | 0/1 | TX-path reduced vs full MIMO | SR003080 |
| `ts_28554_ee_class` | enum | `full_power` / `symbol_es` / `cell_es` | TS 28.554 / 28.310 |

**Used by:** energy, resource, carbon, ran_sleep, renewable_energy, digital_twin.

---

## 5. `slice_utilization.csv`

**What it is:** Per-slice (S-NSSAI) resource and SLA samples for the slice / green-slice agents.

**Grain:** S-NSSAI × time  
**Rows:** 80,000

| Column | Unit / type | What it specifies | Spec / source |
|--------|-------------|-------------------|---------------|
| `timestamp` | step | Sample time | internal |
| `slice` | name | eMBB / URLLC / mMTC (legacy) | TS 23.501 |
| `sst` | 1/2/3 | Slice/Service Type | TS 23.501 |
| `sd` | hex | Slice Differentiator | TS 23.501 |
| `s_nssai` | SST-SD | Single NSSAI | TS 23.501 |
| `nsi_id` | ID | Network Slice Instance | TS 23.501 / 28.541 |
| `dnn` | name | Data Network Name | TS 23.501 |
| `fiveqi` | 5QI | QoS class of the slice | TS 23.501 |
| `pdb_ms` | ms | Packet Delay Budget | TS 23.501 |
| `prb_utilization` | 0–1 | Slice PRB share used (legacy) | TS 28.552 |
| `prb_dl_util` | 0–1 | DL PRB util of slice | TS 28.552 / SharePoint slice analytics |
| `prb_ul_util` | 0–1 | UL PRB util of slice | TS 28.552 |
| `active_ues` | count | Active PDU sessions / UEs (legacy) | TS 28.552 |
| `max_ues` | count | Slice admission cap | TS 23.503 |
| `sla_compliance` | 0–1 | Fraction of samples meeting SLA (legacy) | TS 28.554 |
| `availability` | 0–1 | Service availability | TS 28.554 |
| `drop_rate` | 0–1 | Session / packet drop | TS 28.554 |
| `throughput_mbps` | Mbps | Slice throughput (legacy) | TS 28.552 |
| `latency_p99_ms` | ms | 99th percentile latency (legacy) | TS 28.554 |
| `gfbr_mbps` | Mbps | Guaranteed Flow Bit Rate | TS 23.501 |
| `mfbr_mbps` | Mbps | Maximum Flow Bit Rate | TS 23.501 |
| `session_ambr_mbps` | Mbps | Session-AMBR | TS 23.501 |
| `isolation_level` | hard/soft/shared | Resource isolation | TS 23.501 / O-RAN slice |
| `resource_share_pct` | % | Configured PRB share | SharePoint *Analytic for 5G Slices* |

**Used by:** slice, green_slice.

---

## 6. `handover_events.csv`

**What it is:** Completed (or failed) handover attempts for MRO and mobility KPIs. One row = one HO attempt.

**Grain:** HO event  
**Rows:** 80,000 (~97.8% success)

| Column | Unit / type | What it specifies | Spec / source |
|--------|-------------|-------------------|---------------|
| `event_id` | ID | `HO_000000`… | internal |
| `timestamp` | step | HO time | internal |
| `ue_id` | ID | UE | TS 38.331 |
| `source_cell` | ID | Source cell (legacy) | TS 38.331 |
| `target_cell` | ID | Target cell (legacy) | TS 38.331 |
| `source_pci` | PCI | Source PCI | TS 38.211 |
| `target_pci` | PCI | Target PCI | TS 38.211 |
| `ho_type` | enum | intra_freq / inter_freq / inter_gnb / intra_du / ng_based | TS 38.300 / 38.331 |
| `rsrp_source_dbm` | dBm | Source RSRP at HO (legacy) | TS 38.215 |
| `rsrp_target_dbm` | dBm | Target RSRP at HO (legacy) | TS 38.215 |
| `velocity_mps` | m/s | UE speed at HO (legacy) | TS 38.304 |
| `success` | 0/1 | HO success (legacy target) | TS 38.133 |
| `failure_cause` | enum | too_late / too_early / wrong_cell / rlf / ho_prep_failure / timeout / empty if OK | TS 38.331 / CFAM MRO |
| `delay_ms` | ms | HO interruption (prep+exec+complete) (legacy) | TS 38.133 |
| `ho_prep_ms` | ms | HO preparation time | TS 38.331 |
| `ho_exec_ms` | ms | HO execution time | TS 38.331 |
| `a3_offset_db` | dB | A3 offset used | TS 38.331 |
| `ttt_ms` | ms | Time-To-Trigger used | TS 38.331 |
| `too_early` | 0/1 | Too-early HO | CFAM OSS_FC_017307 |
| `too_late` | 0/1 | Too-late HO | CFAM M8021C43 |
| `wrong_cell` | 0/1 | Wrong-cell HO | CFAM OSS_FC_017307 |
| `ping_pong` | 0/1 | Ping-pong HO | CFAM OSS_FC_017307 |
| `rlf_after_ho` | 0/1 | RLF shortly after HO | TS 38.133 |
| `meas_gap_ms` | ms | Meas gap (inter-freq) | TS 38.133 |
| `rrc_cause` | string | RRC / HO cause text | TS 38.331 |
| `cfam_m8021c43_late_ho` | 0/1 | PM counter late-HO flag | RP001677 M8021C43 |
| `mro_action` | enum | Suggested CIO/TTT change | CFAM OSS_FC_017307 |
| `beam_switch` | 0/1 | Beam-level HO / switch | Rel-18 beam management |

**Used by:** mobility (events), plots / MRO analytics.

---

## Files written with the CSVs

| File | Role |
|------|------|
| `dataset_metadata.json` | Generator settings, spec list, backward-compatible column names |
| `DATASET_CATALOG.md` | This catalog |

Agents still train on the original feature names listed in `src/agents/*_agent.py` (`cqi`, `sinr_db`, `handover_pending`, `is_attack`, `sleep_state`, `prb_utilization`, …).
