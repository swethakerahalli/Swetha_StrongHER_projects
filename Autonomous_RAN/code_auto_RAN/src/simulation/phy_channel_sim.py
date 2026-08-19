"""PHY-layer channel simulation stub (AI-native air interface support)."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ChannelState:
    csi: np.ndarray
    sinr_db: float
    predicted_csi: np.ndarray | None = None


class PHYChannelSimulator:
    """Simplified NR channel model aligned with TR 38.901 concepts."""

    def __init__(self, num_antennas: int = 4, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.num_antennas = num_antennas

    def generate_csi(self, ue_id: str, velocity: float = 0.0) -> ChannelState:
        h = self.rng.normal(0, 1, (self.num_antennas, self.num_antennas)) + \
            1j * self.rng.normal(0, 1, (self.num_antennas, self.num_antennas))
        doppler = min(1.0, velocity / 30.0)
        h *= (1 - 0.1 * doppler)
        sinr = float(10 * np.log10(np.abs(h).mean() ** 2 / 0.01 + 1e-9))
        return ChannelState(csi=h, sinr_db=sinr)

    def neural_channel_estimate(self, pilots: np.ndarray, interference: float = 0.1) -> np.ndarray:
        """AI-based channel estimation (CNN surrogate using regularized LS)."""
        noise_var = interference * 0.01
        h_ls = pilots / (np.abs(pilots) ** 2 + noise_var)
        return h_ls * 0.7 + self.rng.normal(0, 0.05, pilots.shape) * 0.3

    def predict_future_csi(self, history: list[np.ndarray], k: int = 3) -> np.ndarray:
        """Generative channel prediction: H_hat_{t+1} = f(H_t, ..., H_{t-k})."""
        if not history:
            return self.rng.normal(0, 1, (self.num_antennas, self.num_antennas))
        recent = history[-k:]
        weights = np.linspace(0.2, 1.0, len(recent))
        weights /= weights.sum()
        pred = sum(w * h for w, h in zip(weights, recent))
        pred += self.rng.normal(0, 0.05, pred.shape)
        return pred

    def beamforming_weights(self, csi: np.ndarray) -> np.ndarray:
        """MRT beamforming from estimated CSI."""
        v = csi[:, 0] if csi.ndim > 1 else csi
        return v / (np.linalg.norm(v) + 1e-9)
