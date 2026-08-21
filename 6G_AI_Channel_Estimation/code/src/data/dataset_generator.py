"""Vectorized 3GPP-aligned synthetic 6G channel estimation dataset (>= 60k rows)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.channel import (
    ATTACK_PROBS,
    ATTACK_TYPES,
    CHANNEL_PROFILES,
    SCENARIO_PARAMS,
    ber_from_snr_nmse,
    ls_estimate,
    mmse_estimate,
    nmse,
)
from src.common.utils import load_config, project_root, save_json


class ChannelDatasetGenerator:
    def __init__(self, n_samples: int | None = None, seed: int = 42):
        cfg = load_config("system_config.json")
        self.n = int(n_samples or cfg["dataset"]["n_samples"])
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.split_cfg = cfg["dataset"]
        self.output_dir = project_root() / cfg["paths"]["datasets"]
        self.sim = cfg["simulation"]

    def generate_all(self) -> dict[str, Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        main = self._generate_main()
        mobility = self._generate_mobility(main)
        security = self._generate_security(main)
        twin = self._generate_twin_states(main)
        meta = self._write_metadata(main)
        return {
            "channel_estimation": main,
            "mobility": mobility,
            "security": security,
            "digital_twin": twin,
            "metadata": meta,
        }

    def _generate_main(self) -> Path:
        n = self.n
        rng = self.rng
        scenarios = list(SCENARIO_PARAMS.keys())
        scenario = rng.choice(scenarios, n)
        profiles = list(CHANNEL_PROFILES.keys())
        profile = rng.choice(profiles, n)

        fc = np.empty(n)
        scs = np.empty(n)
        bw = np.empty(n)
        ds = np.empty(n)
        n_tx = np.empty(n, dtype=int)
        n_rx = np.empty(n, dtype=int)
        freq_range = np.empty(n, dtype=object)
        for name, params in SCENARIO_PARAMS.items():
            mask = scenario == name
            k = int(mask.sum())
            if k == 0:
                continue
            fc[mask] = rng.choice(params["fc_ghz"], k)
            scs[mask] = rng.choice(params["scs_khz"], k)
            bw[mask] = rng.choice(params["bw_mhz"], k)
            lo, hi = params["ds_ns"]
            ds[mask] = rng.uniform(lo, hi, k)
            n_tx[mask] = rng.choice(params["n_tx"], k)
            n_rx[mask] = rng.choice(params["n_rx"], k)
            freq_range[mask] = params["freq_range"]

        los = np.array([CHANNEL_PROFILES[p]["los"] for p in profile])
        n_taps = np.array([CHANNEL_PROFILES[p]["n_taps"] for p in profile])
        k_factor = np.array([CHANNEL_PROFILES[p]["k_factor_db"] for p in profile])
        k_factor = np.where(np.isfinite(k_factor), k_factor, rng.uniform(-5, 2, n))

        velocity_kmh = rng.choice([3, 30, 60, 120, 240, 500], n, p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05])
        velocity_mps = velocity_kmh / 3.6
        doppler_hz = velocity_mps * fc * 1e9 / 3e8
        snr_db = rng.normal(12, 8, n)
        snr_db = np.clip(snr_db, -5, 35)
        snr_lin = 10.0 ** (snr_db / 10.0)

        path_loss_db = 32.4 + 20 * np.log10(fc) + 20 * np.log10(rng.uniform(30, 800, n))
        rsrp_dbm = -path_loss_db + rng.normal(0, 4, n) + 46
        rsrq_db = np.clip(snr_db / 3.0 - 12 + rng.normal(0, 1.5, n), -24, -3)
        rssi_dbm = rsrp_dbm + rng.uniform(8, 18, n)
        cqi = np.clip((snr_db / 2.2 + 4).astype(int), 0, 15)
        sinr_db = snr_db - rng.uniform(0, 4, n)

        aoa_deg = rng.uniform(0, 360, n)
        aod_deg = rng.uniform(0, 360, n)
        zsa_deg = rng.uniform(2, 20, n)
        zsd_deg = rng.uniform(2, 15, n)
        beam_index = rng.integers(0, 32, n)
        ris_elements = np.where(scenario == "RIS", rng.choice([64, 128, 256, 1024], n), 0)
        sat_elevation_deg = np.where(scenario == "NTN-LEO", rng.uniform(20, 70, n), np.nan)

        h_amp = rng.rayleigh(0.7, n)
        h_amp = np.where(los, h_amp + 10 ** (k_factor / 20.0) * 0.15, h_amp)
        h_phase = rng.uniform(-np.pi, np.pi, n)
        h_true = h_amp * np.cos(h_phase)

        h_ls = ls_estimate(h_true, snr_lin, rng)
        h_mmse = mmse_estimate(h_true, snr_lin, ds, rng)
        # AI estimator gets a residual advantage that grows with SNR and degrades with Doppler/THz.
        ai_gain = 0.45 + 0.25 * np.clip(snr_db / 25.0, 0, 1) - 0.12 * np.clip(doppler_hz / 1500.0, 0, 1)
        ai_gain = np.clip(ai_gain, 0.15, 0.75)
        h_ai = h_true + (1.0 - ai_gain) * (h_mmse - h_true)

        nmse_ls = nmse(h_true, h_ls)
        nmse_mmse = nmse(h_true, h_mmse)
        nmse_ai = nmse(h_true, h_ai)

        attack = rng.choice(ATTACK_TYPES, n, p=ATTACK_PROBS)
        attack_severity = rng.uniform(0.1, 1.0, n)
        attack_severity = np.where(attack == "normal", 0.0, attack_severity)

        # Attack impact on observed CSI / NMSE
        impact = np.where(attack == "normal", 0.0, attack_severity)
        impact = np.where(attack == "jamming", impact * 1.4, impact)
        impact = np.where(attack == "pilot_contamination", impact * 1.2, impact)
        nmse_ls = np.clip(nmse_ls * (1.0 + 2.5 * impact), 0.002, 1.5)
        nmse_mmse = np.clip(nmse_mmse * (1.0 + 2.0 * impact), 0.002, 1.2)
        nmse_ai = np.clip(nmse_ai * (1.0 + 0.7 * impact), 0.001, 0.9)

        ber_ls = ber_from_snr_nmse(snr_db, nmse_ls)
        ber_mmse = ber_from_snr_nmse(snr_db, nmse_mmse)
        ber_ai = ber_from_snr_nmse(snr_db, nmse_ai)

        se_ls = np.log2(1 + snr_lin / (1 + 8 * nmse_ls))
        se_mmse = np.log2(1 + snr_lin / (1 + 8 * nmse_mmse))
        se_ai = np.log2(1 + snr_lin / (1 + 8 * nmse_ai))

        pilot_overhead = np.clip(0.08 + 0.04 * n_taps / 24 + 0.03 * (n_tx / 64), 0.06, 0.28)
        predicted_csi_error = np.clip(nmse_ai * (0.6 + 0.5 * doppler_hz / 800.0), 0.001, 2.0)
        csi_pred_accuracy = np.clip(1.0 - predicted_csi_error, 0.5, 0.995)
        trust_score = np.clip(1.0 - impact - 0.05 * rng.random(n), 0.05, 1.0)
        anomaly_score = np.clip(impact + rng.normal(0, 0.05, n), 0, 1)
        pilot_correlation = np.where(attack == "pilot_contamination", rng.uniform(0.75, 0.99, n), rng.uniform(0.05, 0.35, n))
        csi_consistency = np.clip(1.0 - impact - rng.uniform(0, 0.08, n), 0.0, 1.0)

        cell_id = rng.integers(0, self.sim["num_cells"], n)
        ue_id = rng.integers(0, self.sim["num_ues"], n)
        timestamp = rng.integers(0, 10_000, n)

        split = np.full(n, "train", dtype=object)
        perm = rng.permutation(n)
        n_train = int(self.split_cfg["train_frac"] * n)
        n_val = int(self.split_cfg["val_frac"] * n)
        split[perm[n_train:n_train + n_val]] = "validation"
        split[perm[n_train + n_val:]] = "test"

        df = pd.DataFrame({
            "sample_id": np.arange(n),
            "split": split,
            "timestamp": timestamp,
            "cell_id": [f"CELL_{c:03d}" for c in cell_id],
            "ue_id": [f"UE_{u:04d}" for u in ue_id],
            "scenario": scenario,
            "channel_profile": profile,
            "freq_range": freq_range,
            "los": los.astype(int),
            "fc_ghz": np.round(fc, 3),
            "scs_khz": scs.astype(int),
            "bandwidth_mhz": bw.astype(int),
            "n_tx": n_tx,
            "n_rx": n_rx,
            "n_taps": n_taps,
            "delay_spread_ns": np.round(ds, 2),
            "k_factor_db": np.round(k_factor, 2),
            "velocity_kmh": velocity_kmh,
            "velocity_mps": np.round(velocity_mps, 2),
            "doppler_hz": np.round(doppler_hz, 2),
            "snr_db": np.round(snr_db, 2),
            "sinr_db": np.round(sinr_db, 2),
            "rsrp_dbm": np.round(rsrp_dbm, 2),
            "rsrq_db": np.round(rsrq_db, 2),
            "rssi_dbm": np.round(rssi_dbm, 2),
            "cqi": cqi,
            "aoa_deg": np.round(aoa_deg, 2),
            "aod_deg": np.round(aod_deg, 2),
            "zsa_deg": np.round(zsa_deg, 2),
            "zsd_deg": np.round(zsd_deg, 2),
            "beam_index": beam_index,
            "ris_elements": ris_elements,
            "sat_elevation_deg": np.round(sat_elevation_deg, 2),
            "h_true": np.round(h_true, 6),
            "h_ls": np.round(h_ls, 6),
            "h_mmse": np.round(h_mmse, 6),
            "h_ai": np.round(h_ai, 6),
            "nmse_ls": np.round(nmse_ls, 6),
            "nmse_mmse": np.round(nmse_mmse, 6),
            "nmse_ai": np.round(nmse_ai, 6),
            "ber_ls": np.round(ber_ls, 8),
            "ber_mmse": np.round(ber_mmse, 8),
            "ber_ai": np.round(ber_ai, 8),
            "se_ls": np.round(se_ls, 4),
            "se_mmse": np.round(se_mmse, 4),
            "se_ai": np.round(se_ai, 4),
            "pilot_overhead": np.round(pilot_overhead, 4),
            "csi_pred_accuracy": np.round(csi_pred_accuracy, 4),
            "attack_type": attack,
            "attack_severity": np.round(attack_severity, 4),
            "is_attack": (attack != "normal").astype(int),
            "trust_score": np.round(trust_score, 4),
            "anomaly_score": np.round(anomaly_score, 4),
            "pilot_correlation": np.round(pilot_correlation, 4),
            "csi_consistency": np.round(csi_consistency, 4),
        })
        path = self.output_dir / "channel_estimation_dataset.csv"
        df.to_csv(path, index=False)
        return path

    def _generate_mobility(self, main_path: Path) -> Path:
        df = pd.read_csv(main_path, usecols=["sample_id", "split", "ue_id", "cell_id", "velocity_mps", "rsrp_dbm", "snr_db", "scenario"])
        rng = self.rng
        n = len(df)
        x = rng.uniform(0, 1000, n)
        y = rng.uniform(0, 1000, n)
        heading = rng.uniform(0, 360, n)
        ho_pending = (df["velocity_mps"] > 20).astype(int) * (rng.random(n) < 0.12).astype(int)
        ho_success = np.where(ho_pending == 1, (rng.random(n) > 0.04).astype(int), 1)
        neighbor_rsrp = df["rsrp_dbm"] + rng.normal(-3, 4, n)
        out = df.copy()
        out["x_m"] = np.round(x, 2)
        out["y_m"] = np.round(y, 2)
        out["heading_deg"] = np.round(heading, 2)
        out["handover_pending"] = ho_pending
        out["handover_success"] = ho_success
        out["neighbor_rsrp_dbm"] = np.round(neighbor_rsrp, 2)
        out["predictive_ho_gain"] = np.round(np.clip(0.02 + 0.06 * (df["velocity_mps"] / 40), 0.01, 0.12), 4)
        path = self.output_dir / "mobility_dataset.csv"
        out.to_csv(path, index=False)
        return path

    def _generate_security(self, main_path: Path) -> Path:
        cols = [
            "sample_id", "split", "attack_type", "is_attack", "attack_severity",
            "trust_score", "anomaly_score", "pilot_correlation", "csi_consistency",
            "snr_db", "nmse_ls", "nmse_ai", "scenario",
        ]
        df = pd.read_csv(main_path, usecols=cols)
        rng = self.rng
        n = len(df)
        mitigation = np.where(
            df["is_attack"] == 0,
            "none",
            rng.choice(
                ["pilot_reassignment", "beam_switch", "frequency_hop", "trust_scheduling", "federated_retrain"],
                n,
            ),
        )
        success = np.where(df["is_attack"] == 0, 1, (rng.random(n) < (0.82 + 0.12 * df["trust_score"])).astype(int))
        recovery_ms = np.where(df["is_attack"] == 0, 0, rng.uniform(2, 18, n))
        df["mitigation_action"] = mitigation
        df["mitigation_success"] = success
        df["recovery_time_ms"] = np.round(recovery_ms, 2)
        path = self.output_dir / "security_dataset.csv"
        df.to_csv(path, index=False)
        return path

    def _generate_twin_states(self, main_path: Path) -> Path:
        df = pd.read_csv(main_path, usecols=[
            "cell_id", "scenario", "snr_db", "nmse_ai", "nmse_mmse",
            "attack_type", "trust_score", "beam_index", "attack_severity",
        ])
        rng = self.rng
        cells = sorted(df["cell_id"].unique())
        n_steps = int(self.sim["twin_steps"])
        n_cells = len(cells)
        sample = df.sample(n_steps * n_cells, replace=True, random_state=self.seed).reset_index(drop=True)
        sample["step"] = np.repeat(np.arange(n_steps), n_cells)
        sample["cell_id"] = np.tile(cells, n_steps)
        sample["ue_count"] = rng.integers(8, 40, len(sample))
        sample["load"] = np.round(rng.uniform(0.2, 0.95, len(sample)), 3)
        sample["twin_fidelity"] = np.round(np.clip(
            0.88 + 0.08 * (1 - sample["nmse_ai"]) - 0.06 * sample["attack_severity"], 0.7, 0.995
        ), 4)
        cell_idx = sample["cell_id"].str[-3:].astype(int)
        sample["x_m"] = np.round(50 + (cell_idx % 6) * 150 + rng.normal(0, 12, len(sample)), 1)
        sample["y_m"] = np.round(50 + (cell_idx // 6) * 150 + rng.normal(0, 12, len(sample)), 1)
        path = self.output_dir / "digital_twin_states.csv"
        sample.to_csv(path, index=False)
        return path

    def _write_metadata(self, main_path: Path) -> Path:
        df = pd.read_csv(main_path, usecols=["split", "scenario", "attack_type", "channel_profile"])
        meta = {
            "n_rows": int(len(df)),
            "split_counts": df["split"].value_counts().to_dict(),
            "scenarios": df["scenario"].value_counts().to_dict(),
            "attacks": df["attack_type"].value_counts().to_dict(),
            "profiles": df["channel_profile"].value_counts().to_dict(),
            "standards": [
                "3GPP TR 38.901 CDL/TDL",
                "3GPP TS 38.211 DMRS/CSI-RS",
                "3GPP TS 38.214 CSI reporting",
                "3GPP TR 38.843 AI/ML for NR air interface",
                "3GPP TR 38.811 NTN channel",
                "3GPP TS 38.101-4 TDL-A30/B100/C300",
            ],
        }
        path = self.output_dir / "dataset_metadata.json"
        save_json(meta, path)
        return path
