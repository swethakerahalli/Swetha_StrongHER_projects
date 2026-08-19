"""Air Interface Agent — AI-native waveform, MCS, and PHY adaptation."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from src.agents.ai_predictor import ml_predict, scale_to_range
from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent
from src.simulation.phy_channel_sim import PHYChannelSimulator


class AirInterfaceAgent(BaseAgent):
    FEATURE_KEYS = ["sinr_db", "cqi", "rsrp_dbm", "mcs", "prb_allocated"]

    def __init__(self, model_path=None):
        super().__init__("air_interface_agent", model_path)
        self.phy = PHYChannelSimulator()
        self.waveform_model = None

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        X = data[self.FEATURE_KEYS].fillna(0).values
        y = data["throughput_mbps"].values if "throughput_mbps" in data.columns else data["sinr_db"].values
        self.model = GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)
        self.model.fit(X, y)
        self.waveform_model = GradientBoostingRegressor(n_estimators=30, random_state=42)
        self.waveform_model.fit(X, data["mcs"].values if "mcs" in data.columns else y)
        self.is_trained = True
        return {"r2_score": float(self.model.score(X, y)), "samples": len(data)}

    def save(self, path) -> None:
        import joblib
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "waveform_model": self.waveform_model,
                     "agent_id": self.agent_id, "train_cols": getattr(self, "_train_cols", self.FEATURE_KEYS)}, path)

    def load(self, path) -> None:
        import joblib
        payload = joblib.load(path)
        self.model = payload["model"]
        self.waveform_model = payload.get("waveform_model")
        self._train_cols = payload.get("train_cols", self.FEATURE_KEYS)
        self.is_trained = True

    def predict(self, observation: AgentObservation) -> AgentAction:
        f = observation.features
        spectral_eff = ml_predict(self.model, f, self.FEATURE_KEYS)
        optimal_mcs = int(scale_to_range(
            ml_predict(self.waveform_model or self.model, f, self.FEATURE_KEYS), 0, 28
        ))
        ch = self.phy.generate_csi("UE_AI", velocity=f.get("velocity_mps", 0))
        waveform = "OFDM-AI" if f.get("sinr_db", 10) > 8 else "OFDM-robust"
        modulation = "256QAM" if optimal_mcs > 20 else ("64QAM" if optimal_mcs > 12 else "16QAM")

        return AgentAction(
            agent_id=self.agent_id,
            action_type="air_interface",
            parameters={
                "waveform": waveform,
                "modulation": modulation,
                "optimal_mcs": optimal_mcs,
                "spectral_efficiency_bps_hz": round(spectral_eff / 100, 3),
                "channel_coherence": round(float(abs(ch.csi).mean()), 4),
                "phy_adaptation": "ai_native",
            },
            confidence=0.92,
        )
