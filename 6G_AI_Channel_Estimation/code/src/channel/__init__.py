"""3GPP TR 38.901 CDL/TDL channel profiles and classical estimators."""

from __future__ import annotations

import numpy as np

# TR 38.901 Table 7.7.2 — representative TDL tap counts and delay-spread scaling.
# CDL-A/B/C = NLOS, CDL-D/E = LOS. Delay spreads are nominal RMS values before
# scenario scaling (Clause 7.7.3).
CHANNEL_PROFILES = {
    "TDL-A": {"los": False, "n_taps": 23, "rms_ds_ns": 30, "k_factor_db": -np.inf},
    "TDL-B": {"los": False, "n_taps": 23, "rms_ds_ns": 100, "k_factor_db": -np.inf},
    "TDL-C": {"los": False, "n_taps": 24, "rms_ds_ns": 300, "k_factor_db": -np.inf},
    "TDL-D": {"los": True, "n_taps": 14, "rms_ds_ns": 30, "k_factor_db": 13.3},
    "TDL-E": {"los": True, "n_taps": 14, "rms_ds_ns": 100, "k_factor_db": 22.0},
    "CDL-A": {"los": False, "n_taps": 23, "rms_ds_ns": 30, "k_factor_db": -np.inf},
    "CDL-B": {"los": False, "n_taps": 23, "rms_ds_ns": 100, "k_factor_db": -np.inf},
    "CDL-C": {"los": False, "n_taps": 24, "rms_ds_ns": 300, "k_factor_db": -np.inf},
    "CDL-D": {"los": True, "n_taps": 14, "rms_ds_ns": 30, "k_factor_db": 13.3},
    "CDL-E": {"los": True, "n_taps": 14, "rms_ds_ns": 100, "k_factor_db": 22.0},
}

# Scenario-dependent RMS delay spread (ns) inspired by TR 38.901 Table 7.5-6
# and TS 38.101-4 TDL-A30/B100/C300, plus TR 38.811 NTN and THz research ranges.
SCENARIO_PARAMS = {
    "UMa": {"fc_ghz": [3.5, 6.0], "scs_khz": [15, 30], "bw_mhz": [20, 40, 100], "ds_ns": (30, 300), "n_tx": [8, 16, 32], "n_rx": [2, 4], "freq_range": "FR1"},
    "UMi": {"fc_ghz": [3.5, 28.0], "scs_khz": [30, 60, 120], "bw_mhz": [40, 100, 200], "ds_ns": (20, 150), "n_tx": [16, 32, 64], "n_rx": [2, 4], "freq_range": "FR1/FR2"},
    "RMa": {"fc_ghz": [0.7, 2.1], "scs_khz": [15], "bw_mhz": [10, 20], "ds_ns": (30, 370), "n_tx": [4, 8], "n_rx": [2], "freq_range": "FR1"},
    "InH": {"fc_ghz": [3.5, 28.0], "scs_khz": [30, 120], "bw_mhz": [40, 100], "ds_ns": (10, 80), "n_tx": [8, 16], "n_rx": [2, 4], "freq_range": "FR1/FR2"},
    "FR2-mmWave": {"fc_ghz": [28.0, 39.0], "scs_khz": [60, 120], "bw_mhz": [100, 200, 400], "ds_ns": (10, 50), "n_tx": [32, 64, 128], "n_rx": [4, 8], "freq_range": "FR2-1"},
    "THz": {"fc_ghz": [100.0, 140.0, 220.0], "scs_khz": [120, 480, 960], "bw_mhz": [400, 800, 2000], "ds_ns": (2, 20), "n_tx": [64, 128, 256], "n_rx": [4, 8], "freq_range": "sub-THz"},
    "RIS": {"fc_ghz": [28.0, 39.0], "scs_khz": [120], "bw_mhz": [100, 200], "ds_ns": (15, 80), "n_tx": [32, 64], "n_rx": [4], "freq_range": "FR2-RIS"},
    "NTN-LEO": {"fc_ghz": [2.0, 20.0], "scs_khz": [15, 30, 60], "bw_mhz": [10, 20, 50], "ds_ns": (5, 100), "n_tx": [4, 8], "n_rx": [2, 4], "freq_range": "NTN"},
}

ATTACK_TYPES = [
    "normal",
    "pilot_contamination",
    "jamming",
    "csi_spoofing",
    "false_csi_injection",
    "data_poisoning",
    "adversarial",
    "backdoor",
]

ATTACK_PROBS = np.array([0.72, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01])


def nmse(h_true: np.ndarray, h_hat: np.ndarray) -> np.ndarray:
    num = np.abs(h_true - h_hat) ** 2
    den = np.abs(h_true) ** 2 + 0.08
    return np.clip(num / den, 0.0, 3.0)


def ls_estimate(h_true: np.ndarray, snr_lin: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    noise_std = np.sqrt(1.0 / (2.0 * snr_lin))
    noise = rng.normal(0, 1, size=h_true.shape) * noise_std
    return h_true + noise


def mmse_estimate(h_true: np.ndarray, snr_lin: np.ndarray, delay_spread_ns: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """MMSE-like shrinkage using delay-spread as a correlation prior."""
    ls = ls_estimate(h_true, snr_lin, rng)
    corr = np.exp(-delay_spread_ns / 250.0)
    wiener = (corr * snr_lin) / (corr * snr_lin + 1.0)
    return wiener * ls


def ber_from_snr_nmse(snr_db: np.ndarray, nmse_val: np.ndarray) -> np.ndarray:
    """QPSK BER approximation with residual estimation error as extra noise."""
    snr_eff_db = snr_db - 10.0 * np.log10(1.0 + nmse_val * 10.0)
    snr_eff = 10.0 ** (snr_eff_db / 10.0)
    arg = np.clip(np.sqrt(2.0 * snr_eff), 0, 12)
    # erfc(x)/2 ≈ Q(x√2) for QPSK
    from scipy.special import erfc

    return np.clip(erfc(arg / np.sqrt(2.0)) / 2.0, 1e-7, 0.5)
