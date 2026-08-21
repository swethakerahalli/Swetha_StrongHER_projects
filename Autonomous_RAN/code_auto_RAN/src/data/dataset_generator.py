"""Synthetic RAN datasets: 80k samples each, 3GPP / CFAM / SharePoint aligned."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common.utils import load_config, project_root, save_json

# 3GPP SST (TS 23.501): 1=eMBB, 2=URLLC, 3=mMTC
SST_BY_SLICE = {"eMBB": 1, "URLLC": 2, "mMTC": 3}
FIVEQI_BY_SLICE = {"eMBB": 9, "URLLC": 82, "mMTC": 9}
PDB_MS_BY_SLICE = {"eMBB": 100.0, "URLLC": 5.0, "mMTC": 300.0}
SD_BY_SLICE = {"eMBB": "000001", "URLLC": "000002", "mMTC": "000003"}
MCS_TABLE_BY_SLICE = {"eMBB": "256qam", "URLLC": "64qam", "mMTC": "64qam"}


class RANDatasetGenerator:
    """Generate synthetic telecom datasets aligned with 3GPP KPI definitions."""

    def __init__(self, seed: int = 42, num_samples: int | None = None):
        cfg = load_config("system_config.json")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        sim = cfg["simulation"]
        self.num_cells = int(sim.get("num_cells", 21))
        self.num_ues = int(sim.get("num_ues", 200))
        self.num_steps = int(sim.get("simulation_steps", 500))
        self.num_samples = int(num_samples or sim.get("dataset_samples", 80_000))
        self.slices = list(cfg["network_slices"].keys())
        self.n_prb = int(sim.get("num_prbs", 273))
        self.bw_mhz = float(sim.get("bandwidth_mhz", 100))
        self.fc_ghz = float(sim.get("carrier_frequency_ghz", 3.5))
        self.output_dir = project_root() / cfg["paths"]["datasets"]
        self.plmn_id = "24407"

    def generate_all(self) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "ran_kpi": self._generate_ran_kpi(),
            "mobility": self._generate_mobility(),
            "security": self._generate_security(),
            "energy": self._generate_energy(),
            "slice_utilization": self._generate_slice_utilization(),
            "handover_events": self._generate_handover_events(),
            "metadata": self._write_metadata(),
        }
        return paths

    def _n(self) -> int:
        return self.num_samples

    def _ue_ids(self, idx: np.ndarray) -> np.ndarray:
        return np.array([f"UE_{int(i):04d}" for i in idx])

    def _cell_ids(self, idx: np.ndarray) -> np.ndarray:
        return np.array([f"CELL_{int(i):03d}" for i in idx])

    def _gnb_ids(self, cell_idx: np.ndarray) -> np.ndarray:
        return np.array([f"gNB_{int(i) // 3:03d}" for i in cell_idx])

    def _slice_of(self, idx: np.ndarray) -> np.ndarray:
        sl = np.array(self.slices)
        return sl[idx % len(sl)]

    def _map_slice(self, slices: np.ndarray, mapping: dict) -> np.ndarray:
        out = np.empty(len(slices), dtype=object)
        for k, v in mapping.items():
            out[slices == k] = v
        return out

    def _generate_ran_kpi(self) -> Path:
        """UE/cell air-interface + PM KPIs (TS 38.214/215, TS 28.552, TS 23.501)."""
        n, rng = self._n(), self.rng
        ue_idx = rng.integers(0, self.num_ues, n)
        cell_idx = rng.integers(0, self.num_cells, n)
        slices = self._slice_of(ue_idx)
        sst = self._map_slice(slices, SST_BY_SLICE).astype(int)
        fiveqi = self._map_slice(slices, FIVEQI_BY_SLICE).astype(int)
        pdb = self._map_slice(slices, PDB_MS_BY_SLICE).astype(float)
        sd = self._map_slice(slices, SD_BY_SLICE)
        mcs_table = self._map_slice(slices, MCS_TABLE_BY_SLICE)

        sinr = np.clip(rng.normal(14.0, 7.5, n), -8.0, 35.0)
        # URLLC slightly better SINR; mMTC worse
        sinr = np.where(slices == "URLLC", sinr + 3.0, sinr)
        sinr = np.where(slices == "mMTC", sinr - 4.0, sinr)
        cqi = np.clip(np.floor(sinr / 2.0 + 3.0).astype(int), 0, 15)
        wideband_cqi = np.clip(cqi + rng.integers(-1, 2, n), 0, 15)
        rsrp = np.clip(rng.normal(-95.0, 11.0, n), -140.0, -44.0)
        rsrq = np.clip(rng.normal(-11.5, 3.5, n), -19.5, -3.0)
        ss_rsrp = np.clip(rsrp + rng.normal(-1.5, 1.5, n), -140.0, -44.0)
        ss_sinr = np.clip(sinr + rng.normal(-1.0, 2.0, n), -10.0, 40.0)
        rssi = np.clip(rsrp + rng.uniform(8.0, 22.0, n), -100.0, -20.0)
        pathloss = np.clip(-(rsrp + 23.0) + rng.normal(0, 3, n), 50.0, 160.0)
        ul_sinr = np.clip(sinr + rng.normal(-2.0, 3.0, n), -10.0, 35.0)
        mcs = np.clip(cqi + rng.integers(-1, 3, n), 0, 28)
        ri = np.clip((sinr > 18).astype(int) + (sinr > 24).astype(int) + 1, 1, 4)
        pmi = rng.integers(0, 16, n)
        prb = np.clip((cqi / 15.0 * 24 + rng.integers(1, 8, n)).astype(int), 1, 50)
        buf = np.clip(rng.beta(2.0, 2.2, n), 0.0, 1.0)
        dl_tp = np.clip((cqi / 15.0) * rng.exponential(55.0, n), 0.05, 1500.0)
        ul_tp = np.clip(dl_tp * rng.uniform(0.25, 0.7, n), 0.02, 400.0)
        tp = dl_tp + ul_tp
        latency = np.clip(rng.exponential(pdb / 8.0, n), 0.4, 250.0)
        latency = np.where(slices == "URLLC", np.clip(rng.exponential(1.2, n), 0.3, 8.0), latency)
        pkt_loss = np.clip(rng.beta(1.2, 40.0, n) * (0.02 if True else 1), 0.0, 0.08)
        bler = np.clip(np.exp(-sinr / 8.0) * rng.uniform(0.3, 1.2, n) * 0.15, 0.0, 0.4)
        harq = np.clip(rng.poisson(bler * 8.0), 0, 7).astype(int)
        ta_us = np.clip(pathloss / 0.3 + rng.normal(0, 5, n), 0.0, 667.0)  # ~NR TA range
        phr = np.clip(23.0 - (pathloss - 90.0) / 4.0 + rng.normal(0, 2, n), -23.0, 40.0)
        se = np.clip((cqi / 15.0) * 5.5 * (1.0 - bler), 0.05, 7.5)
        dl_prb_util = np.clip(prb / float(self.n_prb) * rng.uniform(3.0, 8.0, n), 0.02, 1.0)
        ul_prb_util = np.clip(dl_prb_util * rng.uniform(0.35, 0.85, n), 0.01, 1.0)
        jitter = np.clip(latency * rng.uniform(0.05, 0.35, n), 0.05, 40.0)
        rrc_states = np.array(["connected", "inactive", "idle"])
        rrc = rrc_states[rng.choice(3, n, p=[0.72, 0.18, 0.10])]
        rrc_conn = np.clip((dl_prb_util * 80 + rng.integers(5, 25, n)).astype(int), 1, 400)
        pci = (cell_idx * 17 + 3) % 1008
        beam_id = rng.integers(0, 8, n)
        bwp_id = rng.integers(0, 4, n)
        rnti = rng.integers(1, 65519, n)
        drb_id = 1 + (sst - 1)
        qfi = np.where(slices == "URLLC", 2, np.where(slices == "eMBB", 1, 3))
        nr_arfcn = np.full(n, 633334)  # n78 ~3.5 GHz
        mu = np.full(n, 1)  # 30 kHz SCS for n78 100 MHz
        ts = rng.integers(0, max(self.num_steps, n // max(self.num_ues, 1) + 1), n)

        df = pd.DataFrame({
            "timestamp": ts,
            "ue_id": self._ue_ids(ue_idx),
            "cell_id": self._cell_ids(cell_idx),
            "gnb_id": self._gnb_ids(cell_idx),
            "pci": pci,
            "plmn_id": self.plmn_id,
            "nr_arfcn": nr_arfcn,
            "slice": slices,
            "sst": sst,
            "sd": sd,
            "fiveqi": fiveqi,
            "qfi": qfi,
            "pdb_ms": pdb,
            "drb_id": drb_id,
            "rnti": rnti,
            "rrc_state": rrc,
            "cqi": cqi,
            "wideband_cqi": wideband_cqi,
            "ri": ri,
            "pmi": pmi,
            "mcs": mcs,
            "mcs_table": mcs_table,
            "sinr_db": np.round(sinr, 2),
            "ul_sinr_db": np.round(ul_sinr, 2),
            "rsrp_dbm": np.round(rsrp, 2),
            "rsrq_db": np.round(rsrq, 2),
            "ss_rsrp_dbm": np.round(ss_rsrp, 2),
            "ss_sinr_db": np.round(ss_sinr, 2),
            "rssi_dbm": np.round(rssi, 2),
            "pathloss_db": np.round(pathloss, 2),
            "prb_allocated": prb,
            "dl_prb_util": np.round(dl_prb_util, 4),
            "ul_prb_util": np.round(ul_prb_util, 4),
            "buffer_occupancy": np.round(buf, 3),
            "throughput_mbps": np.round(tp, 2),
            "dl_throughput_mbps": np.round(dl_tp, 2),
            "ul_throughput_mbps": np.round(ul_tp, 2),
            "latency_ms": np.round(latency, 2),
            "jitter_ms": np.round(jitter, 2),
            "packet_loss": np.round(pkt_loss, 4),
            "bler": np.round(bler, 4),
            "harq_retx": harq,
            "ta_us": np.round(ta_us, 2),
            "phr_db": np.round(phr, 2),
            "se_bps_hz": np.round(se, 3),
            "rrc_connected_ues": rrc_conn,
            "bwp_id": bwp_id,
            "beam_id": beam_id,
            "mu_numerology": mu,
            "bandwidth_mhz": self.bw_mhz,
        })
        path = self.output_dir / "ran_kpi_dataset.csv"
        df.to_csv(path, index=False)
        return path

    def _generate_mobility(self) -> Path:
        """UE mobility traces + meas-report context (TS 38.331/38.133/38.304, CFAM MRO)."""
        n, rng = self._n(), self.rng
        ue_idx = rng.integers(0, self.num_ues, n)
        cell_idx = rng.integers(0, self.num_cells, n)
        neigh = (cell_idx + rng.integers(1, max(self.num_cells, 2), n)) % self.num_cells
        x = rng.uniform(0, 2000, n)
        y = rng.uniform(0, 2000, n)
        speed = np.abs(rng.normal(8.0, 12.0, n))
        # Mix pedestrian / vehicular / rail
        mix = rng.random(n)
        speed = np.where(mix < 0.55, rng.uniform(0.2, 2.0, n), speed)
        speed = np.where((mix >= 0.55) & (mix < 0.9), rng.uniform(5.0, 35.0, n), speed)
        speed = np.where(mix >= 0.9, rng.uniform(40.0, 80.0, n), speed)
        heading = rng.uniform(-180, 180, n)
        rsrp = np.clip(rng.normal(-96.0, 10.0, n), -140.0, -44.0)
        neigh_rsrp = np.clip(rsrp + rng.normal(4.0, 6.0, n), -140.0, -44.0)
        rsrq = np.clip(rng.normal(-12.0, 3.5, n), -19.5, -3.0)
        sinr = np.clip(rng.normal(12.0, 8.0, n), -8.0, 35.0)
        a3_off = np.round(rng.choice([1.0, 2.0, 3.0, 4.0, 6.0], n), 1)
        ttt = rng.choice([40, 80, 160, 256, 320, 480, 640], n)
        hyst = rng.choice([0.0, 1.0, 2.0, 3.0], n)
        s_measure = rng.choice([-140, -115, -110, -105, -100], n)
        ho_pending = ((neigh_rsrp - rsrp) > (a3_off + hyst)) & (rng.random(n) < 0.35)
        ho_pending = ho_pending.astype(int)
        event = np.where(ho_pending == 1, rng.choice(["A3", "A5", "A4"], n), rng.choice(["A1", "A2", "none"], n))
        mob_state = np.where(speed < 3, "normal", np.where(speed < 30, "medium", "high"))
        indoor = (rng.random(n) < 0.22).astype(int)
        rlf = ((rsrp < -118) & (rng.random(n) < 0.08)).astype(int)
        ping_pong = ((ho_pending == 1) & (rng.random(n) < 0.04)).astype(int)
        late_ho = ((ho_pending == 1) & (rsrp < -112) & (rng.random(n) < 0.15)).astype(int)
        early_ho = ((ho_pending == 1) & (rsrp > -85) & (rng.random(n) < 0.08)).astype(int)
        meas_id = rng.integers(1, 32, n)
        dist = np.hypot(x % 500 - 250, y % 500 - 250)
        ts = rng.integers(0, max(self.num_steps, 2000), n)
        neigh_json = [f'["CELL_{int((c+1)%self.num_cells):03d}","CELL_{int((c+2)%self.num_cells):03d}","CELL_{int((c+3)%self.num_cells):03d}"]' for c in cell_idx]

        df = pd.DataFrame({
            "timestamp": ts,
            "ue_id": self._ue_ids(ue_idx),
            "cell_id": self._cell_ids(cell_idx),
            "serving_pci": (cell_idx * 17 + 3) % 1008,
            "neighbor_cell_id": self._cell_ids(neigh),
            "neighbor_pci": (neigh * 17 + 3) % 1008,
            "x_m": np.round(x, 2),
            "y_m": np.round(y, 2),
            "distance_to_site_m": np.round(dist, 2),
            "velocity_mps": np.round(speed, 2),
            "direction_deg": np.round(heading, 2),
            "mobility_state": mob_state,
            "indoor_flag": indoor,
            "neighbor_cells": neigh_json,
            "rsrp_dbm": np.round(rsrp, 2),
            "rsrq_db": np.round(rsrq, 2),
            "sinr_db": np.round(sinr, 2),
            "ss_rsrp_dbm": np.round(np.clip(rsrp - 1.0, -140, -44), 2),
            "neighbor_rsrp_dbm": np.round(neigh_rsrp, 2),
            "a3_offset_db": a3_off,
            "hysteresis_db": hyst,
            "ttt_ms": ttt,
            "s_measure_dbm": s_measure,
            "meas_event": event,
            "meas_id": meas_id,
            "handover_pending": ho_pending,
            "rlf_detected": rlf,
            "ping_pong_flag": ping_pong,
            "cfam_late_ho_ind": late_ho,
            "cfam_early_ho_ind": early_ho,
            "cell_reselection_priority": rng.integers(0, 8, n),
            "beam_id": rng.integers(0, 8, n),
        })
        path = self.output_dir / "mobility_traces.csv"
        df.to_csv(path, index=False)
        return path

    def _generate_security(self) -> Path:
        """Security / FM / spectrum events (TS 33.501, CFAM SR001534)."""
        n, rng = self._n(), self.rng
        threat_types = np.array(["normal", "jamming", "spoofing", "ddos", "rogue_gnb", "pilot_contamination"])
        threat = rng.choice(threat_types, n, p=[0.82, 0.05, 0.04, 0.04, 0.03, 0.02])
        is_attack = (threat != "normal").astype(int)
        rate = np.where(is_attack == 1, rng.exponential(2500.0, n), rng.exponential(120.0, n))
        auth_fail = np.where(is_attack == 1, rng.poisson(4.0, n), rng.poisson(0.12, n))
        spec = np.where(is_attack == 1, rng.uniform(0.65, 1.0, n), rng.uniform(0.0, 0.32, n))
        entropy = np.where(is_attack == 1, rng.uniform(0.05, 0.45, n), rng.uniform(0.5, 0.98, n))
        nas_int = ((threat == "spoofing") | ((is_attack == 1) & (rng.random(n) < 0.25))).astype(int)
        nas_auth = ((threat == "ddos") | (auth_fail > 2)).astype(int)
        jam_pwr = np.where(threat == "jamming", rng.uniform(-40.0, 10.0, n), rng.uniform(-110.0, -80.0, n))
        affected_prb = np.where(threat == "jamming", rng.integers(8, 80, n), rng.integers(0, 4, n))
        severity = np.where(is_attack == 0, "info", np.where(spec > 0.85, "critical", np.where(spec > 0.7, "major", "minor")))
        algs_i = rng.choice(["NIA2", "NIA3", "NIA1"], n, p=[0.7, 0.25, 0.05])
        algs_c = rng.choice(["NEA2", "NEA3", "NEA0"], n, p=[0.72, 0.25, 0.03])
        aka = np.where(nas_auth == 1, "fail", "success")
        fm_alarm = np.where(is_attack == 1, rng.choice(["BTS_RESET", "RADIO_FAULT", "DSP_FAULT", "JAMMING_DET", "NGAP_FAIL"], n), "none")
        recovery = np.where(is_attack == 1, rng.choice(["rate_limit", "quarantine", "slice_protect", "bts_autonomous_recovery", "none"], n), "none")
        ngap = np.where(is_attack == 1, rng.choice(["radio-connection-with-ue-lost", "unspecified", "om-intervention", "none"], n), "none")
        conf = np.where(is_attack == 1, rng.uniform(0.7, 0.99, n), rng.uniform(0.01, 0.35, n))
        cell_idx = rng.integers(0, self.num_cells, n)
        ts = rng.integers(0, max(self.num_steps, 5000), n)

        df = pd.DataFrame({
            "event_id": [f"SEC_{i:06d}" for i in range(n)],
            "timestamp": ts,
            "source_ip": [f"10.{int(a)}.{int(b)}.{int(c)}" for a, b, c in zip(rng.integers(0, 255, n), rng.integers(0, 255, n), rng.integers(1, 254, n))],
            "target_cell": self._cell_ids(cell_idx),
            "target_pci": (cell_idx * 17 + 3) % 1008,
            "threat_type": threat,
            "is_attack": is_attack,
            "severity": severity,
            "packet_rate_pps": np.round(rate, 2),
            "auth_failures": auth_fail.astype(int),
            "nas_integrity_fail": nas_int,
            "nas_auth_fail": nas_auth,
            "fiveg_aka_result": aka,
            "integrity_alg": algs_i,
            "cipher_alg": algs_c,
            "spectrum_anomaly_score": np.round(spec, 3),
            "flow_entropy": np.round(entropy, 3),
            "jammer_power_dbm": np.round(jam_pwr, 2),
            "affected_prb_count": affected_prb,
            "rogue_pci": np.where(threat == "rogue_gnb", rng.integers(0, 1008, n), -1),
            "bytes_transferred": rng.integers(100, 5_000_000, n),
            "ngap_cause": ngap,
            "fm_alarm_id": fm_alarm,
            "cfam_recovery_action": recovery,
            "detection_confidence": np.round(conf, 3),
            "ts_33501_control": np.where(is_attack == 1, "SEAF/AUSF", "N/A"),
        })
        path = self.output_dir / "security_events.csv"
        df.to_csv(path, index=False)
        return path

    def _generate_energy(self) -> Path:
        """Cell energy / EE KPIs (TS 28.554 EE, CFAM SR003080 TX-path switching)."""
        n, rng = self._n(), self.rng
        cell_idx = rng.integers(0, self.num_cells, n)
        hour = rng.integers(0, 24, n)
        load = np.clip(0.25 + 0.35 * np.sin((hour - 8) / 24.0 * 2 * np.pi) + rng.normal(0, 0.15, n), 0.05, 0.98)
        sleep = ((load < 0.18) & (rng.random(n) < 0.45)).astype(int)
        deep_sleep = ((sleep == 1) & (rng.random(n) < 0.35)).astype(int)
        tx_paths = np.where(sleep == 1, 0, np.where(load < 0.4, rng.choice([2, 4], n), rng.choice([4, 8, 16], n)))
        pa = np.where(sleep == 1, 0.0, np.clip(15.0 + 31.0 * load + rng.normal(0, 1.5, n), 0.0, 50.0))
        power = np.where(sleep == 1, np.where(deep_sleep == 1, 25.0, 55.0), 180.0 + 720.0 * load + 8.0 * tx_paths)
        traffic = load * rng.uniform(40.0, 280.0, n)
        renewable = np.clip(20.0 + 25.0 * np.sin((hour - 6) / 24.0 * 2 * np.pi) + rng.normal(0, 8, n), 0.0, 85.0)
        carbon = np.clip(420.0 - 2.2 * renewable + rng.normal(0, 25, n), 80.0, 650.0)
        ebit = np.where(traffic > 1, (power / (traffic * 1e6)) * 1e6, 50.0)  # uJ/bit approx
        ee = np.where(power > 1, (traffic * 1e6) / (power + 1e-6), 0.0)
        carrier_off = ((load < 0.22) & (rng.random(n) < 0.2)).astype(int)
        ts = rng.integers(0, max(self.num_steps, 4000), n)

        df = pd.DataFrame({
            "timestamp": ts,
            "cell_id": self._cell_ids(cell_idx),
            "gnb_id": self._gnb_ids(cell_idx),
            "hour_of_day": hour,
            "power_consumption_w": np.round(power, 2),
            "pa_power_dbm": np.round(pa, 2),
            "cell_utilization": np.round(load, 3),
            "sleep_state": sleep,
            "deep_sleep_flag": deep_sleep,
            "num_active_tx_paths": tx_paths,
            "carrier_shutdown": carrier_off,
            "renewable_pct": np.round(renewable, 2),
            "carbon_intensity_gco2_kwh": np.round(carbon, 2),
            "traffic_demand_mbps": np.round(traffic, 2),
            "energy_per_bit_uj": np.round(np.clip(ebit, 0.01, 200.0), 4),
            "ee_bit_per_joule": np.round(np.clip(ee, 0.0, 1e9), 2),
            "grid_power_w": np.round(power * (1.0 - renewable / 100.0), 2),
            "battery_soc_pct": np.round(np.clip(rng.normal(72, 15, n), 10, 100), 1),
            "temperature_c": np.round(np.clip(28 + 18 * load + rng.normal(0, 3, n), 15, 85), 1),
            "rectifier_load_pct": np.round(np.clip(load * 90 + rng.normal(0, 5, n), 5, 100), 1),
            "cfam_tx_path_switch": (tx_paths < 8).astype(int),
            "ts_28554_ee_class": np.where(sleep == 1, "cell_es", np.where(load < 0.4, "symbol_es", "full_power")),
        })
        path = self.output_dir / "energy_metrics.csv"
        df.to_csv(path, index=False)
        return path

    def _generate_slice_utilization(self) -> Path:
        """NSSI / S-NSSAI resource and SLA metrics (TS 23.501/23.503, TS 28.554)."""
        n, rng = self._n(), self.rng
        sl_idx = rng.integers(0, len(self.slices), n)
        sl = np.array(self.slices)[sl_idx]
        sst = self._map_slice(sl, SST_BY_SLICE).astype(int)
        fiveqi = self._map_slice(sl, FIVEQI_BY_SLICE).astype(int)
        pdb = self._map_slice(sl, PDB_MS_BY_SLICE).astype(float)
        sd = self._map_slice(sl, SD_BY_SLICE)
        util = np.clip(rng.beta(2.4, 1.8, n), 0.05, 0.98)
        util = np.where(sl == "URLLC", np.clip(util * 0.55, 0.05, 0.7), util)
        sla = np.clip(0.995 - 0.08 * np.maximum(util - 0.85, 0) + rng.normal(0, 0.008, n), 0.85, 1.0)
        tp = util * np.where(sl == "eMBB", rng.uniform(200, 800, n), np.where(sl == "URLLC", rng.uniform(20, 80, n), rng.uniform(5, 40, n)))
        lat_p99 = np.where(sl == "URLLC", rng.exponential(2.5, n), np.where(sl == "eMBB", rng.exponential(12.0, n), rng.exponential(40.0, n)))
        ues = np.where(sl == "mMTC", rng.integers(80, 400, n), rng.integers(8, 120, n))
        gfbr = np.where(sl == "URLLC", rng.uniform(1.0, 10.0, n), np.where(sl == "eMBB", rng.uniform(20.0, 100.0, n), 0.1))
        mfbr = gfbr * rng.uniform(1.2, 2.5, n)
        ambr = np.where(sl == "eMBB", rng.uniform(200, 1000, n), np.where(sl == "URLLC", rng.uniform(20, 100, n), rng.uniform(1, 10, n)))
        drop = np.clip((1.0 - sla) * rng.uniform(0.3, 1.2, n), 0.0, 0.12)
        iso = np.where(sl == "URLLC", "hard", np.where(sl == "eMBB", "soft", "shared"))
        ts = rng.integers(0, max(self.num_steps, 4000), n)
        nsi = np.array([f"NSI-{s}-{int(i % 4):02d}" for s, i in zip(sl, rng.integers(0, 4, n))])

        df = pd.DataFrame({
            "timestamp": ts,
            "slice": sl,
            "sst": sst,
            "sd": sd,
            "s_nssai": [f"{int(a)}-{b}" for a, b in zip(sst, sd)],
            "nsi_id": nsi,
            "dnn": np.where(sl == "eMBB", "internet", np.where(sl == "URLLC", "urllc.corp", "iot.nb")),
            "fiveqi": fiveqi,
            "pdb_ms": pdb,
            "prb_utilization": np.round(util, 4),
            "prb_dl_util": np.round(np.clip(util * rng.uniform(0.9, 1.15, n), 0.02, 1.0), 4),
            "prb_ul_util": np.round(np.clip(util * rng.uniform(0.35, 0.8, n), 0.01, 1.0), 4),
            "active_ues": ues,
            "max_ues": np.where(sl == "mMTC", 2000, np.where(sl == "eMBB", 400, 80)),
            "sla_compliance": np.round(sla, 4),
            "availability": np.round(np.clip(0.999 - drop * 0.2, 0.99, 1.0), 6),
            "drop_rate": np.round(drop, 5),
            "throughput_mbps": np.round(tp, 2),
            "latency_p99_ms": np.round(np.clip(lat_p99, 0.4, 400.0), 2),
            "gfbr_mbps": np.round(gfbr, 2),
            "mfbr_mbps": np.round(mfbr, 2),
            "session_ambr_mbps": np.round(ambr, 2),
            "isolation_level": iso,
            "resource_share_pct": np.round(np.where(sl == "eMBB", 55 + util * 10, np.where(sl == "URLLC", 20 + util * 8, 15 + util * 10)), 2),
        })
        path = self.output_dir / "slice_utilization.csv"
        df.to_csv(path, index=False)
        return path

    def _generate_handover_events(self) -> Path:
        """HO execution records (TS 38.331/38.133, CFAM OSS_FC_017307 / RP001677 M8021C43)."""
        n, rng = self._n(), self.rng
        src = rng.integers(0, self.num_cells, n)
        tgt = (src + rng.integers(1, max(self.num_cells, 2), n)) % self.num_cells
        ue_idx = rng.integers(0, self.num_ues, n)
        success = (rng.random(n) < 0.978).astype(int)
        rsrp_s = np.clip(rng.normal(-98.0, 8.0, n), -140.0, -44.0)
        rsrp_t = np.clip(rsrp_s + rng.normal(7.0, 5.0, n), -140.0, -44.0)
        vel = np.abs(rng.normal(12.0, 14.0, n))
        ho_type = rng.choice(["intra_freq", "inter_freq", "inter_gnb", "intra_du", "ng_based"], n, p=[0.45, 0.25, 0.15, 0.10, 0.05])
        prep = rng.uniform(8.0, 35.0, n)
        exec_ms = rng.uniform(12.0, 55.0, n)
        complete = prep + exec_ms + rng.uniform(5.0, 25.0, n)
        fail_cause = np.full(n, "", dtype=object)
        fail_idx = np.flatnonzero(success == 0)
        if fail_idx.size:
            fail_cause[fail_idx] = rng.choice(
                ["too_late", "too_early", "wrong_cell", "rlf", "ho_prep_failure", "timeout"],
                size=fail_idx.size,
                p=[0.35, 0.20, 0.10, 0.20, 0.10, 0.05],
            )
        too_late = (fail_cause == "too_late").astype(int)
        too_early = (fail_cause == "too_early").astype(int)
        wrong = (fail_cause == "wrong_cell").astype(int)
        ping = ((success == 1) & (rng.random(n) < 0.03)).astype(int)
        rlf_after = ((success == 0) | (rng.random(n) < 0.015)).astype(int)
        mro = np.where(too_late == 1, "increase_cio", np.where(too_early == 1, "decrease_cio", np.where(ping == 1, "increase_ttt", "none")))
        cause_rrc = np.where(success == 1, "handover-successful", rng.choice(["radio-link-failure", "t304-expiry", "ho-failure"], n))
        ts = rng.integers(0, max(self.num_steps, 5000), n)
        a3 = rng.choice([1.0, 2.0, 3.0, 4.0], n)
        ttt = rng.choice([40, 80, 160, 320, 640], n)

        df = pd.DataFrame({
            "event_id": [f"HO_{i:06d}" for i in range(n)],
            "timestamp": ts,
            "ue_id": self._ue_ids(ue_idx),
            "source_cell": self._cell_ids(src),
            "target_cell": self._cell_ids(tgt),
            "source_pci": (src * 17 + 3) % 1008,
            "target_pci": (tgt * 17 + 3) % 1008,
            "ho_type": ho_type,
            "rsrp_source_dbm": np.round(rsrp_s, 2),
            "rsrp_target_dbm": np.round(rsrp_t, 2),
            "velocity_mps": np.round(vel, 2),
            "success": success,
            "failure_cause": fail_cause,
            "delay_ms": np.round(complete, 2),
            "ho_prep_ms": np.round(prep, 2),
            "ho_exec_ms": np.round(exec_ms, 2),
            "a3_offset_db": a3,
            "ttt_ms": ttt,
            "too_early": too_early,
            "too_late": too_late,
            "wrong_cell": wrong,
            "ping_pong": ping,
            "rlf_after_ho": rlf_after,
            "meas_gap_ms": np.where(ho_type == "inter_freq", 6, 0),
            "rrc_cause": cause_rrc,
            "cfam_m8021c43_late_ho": too_late,
            "mro_action": mro,
            "beam_switch": rng.integers(0, 2, n),
        })
        path = self.output_dir / "handover_events.csv"
        df.to_csv(path, index=False)
        return path

    def _write_metadata(self) -> Path:
        catalog = project_root() / "data" / "datasets" / "DATASET_CATALOG.md"
        meta = {
            "generator": "RANDatasetGenerator",
            "seed": self.seed,
            "num_cells": self.num_cells,
            "num_ues": self.num_ues,
            "num_steps": self.num_steps,
            "num_samples_per_dataset": self.num_samples,
            "slices": self.slices,
            "sst_mapping": SST_BY_SLICE,
            "fiveqi_mapping": FIVEQI_BY_SLICE,
            "catalog": str(catalog.name),
            "3gpp_alignment": [
                "TS 38.215", "TS 38.214", "TS 38.331", "TS 38.133", "TS 38.321", "TS 38.304",
                "TS 38.213", "TS 28.552", "TS 28.554", "TS 23.501", "TS 23.503", "TS 33.501", "TS 28.310",
            ],
            "nokia_cfam": [
                "OSS_FC_017307 (5G MRO)",
                "RP001677 / M8021C43 (late HO PM)",
                "SR003080 (TX path switching EE)",
                "SR001534 (BTS autonomous recovery)",
            ],
            "sharepoint_refs": [
                "5G counters and KPIs, v8.0.pptx (RRM-CA Core Team)",
                "NPN- 5G Radio KPI Configuration_Full.pptx",
                "Analytic for 5G Slices v5.pptx",
                "Swetha_Autonomous_RAN_Proj_01.pptx",
                "Multi-agent_green_telecom_carbon_aware_RAN.pdf",
            ],
            "confluence": "EE Confluence MCP timed out during generation; CFAM + local project docs used as fallback",
            "system_insights": "ask/search_specs timed out during generation; cached nokia_cfam_references.json used",
            "backward_compatible_columns": {
                "ran_kpi_dataset.csv": ["timestamp", "ue_id", "cell_id", "slice", "cqi", "sinr_db", "rsrp_dbm", "rsrq_db", "mcs", "prb_allocated", "buffer_occupancy", "throughput_mbps", "latency_ms", "packet_loss"],
                "mobility_traces.csv": ["timestamp", "ue_id", "cell_id", "x_m", "y_m", "velocity_mps", "direction_deg", "neighbor_cells", "rsrp_dbm", "handover_pending"],
                "security_events.csv": ["event_id", "timestamp", "source_ip", "target_cell", "threat_type", "is_attack", "packet_rate_pps", "auth_failures", "spectrum_anomaly_score", "flow_entropy", "bytes_transferred"],
                "energy_metrics.csv": ["timestamp", "cell_id", "power_consumption_w", "cell_utilization", "sleep_state", "renewable_pct", "carbon_intensity_gco2_kwh", "traffic_demand_mbps"],
                "slice_utilization.csv": ["timestamp", "slice", "prb_utilization", "active_ues", "sla_compliance", "throughput_mbps", "latency_p99_ms"],
                "handover_events.csv": ["event_id", "timestamp", "ue_id", "source_cell", "target_cell", "ho_type", "rsrp_source_dbm", "rsrp_target_dbm", "velocity_mps", "success", "failure_cause", "delay_ms"],
            },
        }
        path = self.output_dir / "dataset_metadata.json"
        save_json(meta, path)
        return path


def generate_datasets(seed: int = 42, num_samples: int = 80_000) -> dict[str, Path]:
    return RANDatasetGenerator(seed=seed, num_samples=num_samples).generate_all()
