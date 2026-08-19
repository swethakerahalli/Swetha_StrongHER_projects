"""Synthetic RAN, mobility, security, and energy dataset generation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.utils import load_config, project_root, save_json


class RANDatasetGenerator:
    """Generate synthetic telecom datasets aligned with 3GPP KPI definitions."""

    def __init__(self, seed: int = 42):
        cfg = load_config("system_config.json")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        sim = cfg["simulation"]
        self.num_cells = sim["num_cells"]
        self.num_ues = sim["num_ues"]
        self.num_steps = sim["simulation_steps"]
        self.slices = list(cfg["network_slices"].keys())
        self.output_dir = project_root() / cfg["paths"]["datasets"]

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

    def _generate_ran_kpi(self) -> Path:
        rows = []
        for t in range(self.num_steps):
            for ue in range(self.num_ues):
                cell = int(self.rng.integers(0, self.num_cells))
                sinr = float(self.rng.normal(15, 8))
                cqi = int(np.clip(np.floor(sinr / 2) + 3, 0, 15))
                rsrp = float(self.rng.normal(-95, 12))
                rsrq = float(self.rng.normal(-12, 4))
                mcs = int(np.clip(cqi + self.rng.integers(-1, 2), 0, 28))
                prb = int(self.rng.integers(1, 20))
                buf = float(self.rng.uniform(0, 1))
                tp = max(0.1, (cqi / 15) * self.rng.exponential(50))
                latency = float(max(0.5, self.rng.exponential(5)))
                pkt_loss = float(self.rng.uniform(0, 0.05))
                slice_id = self.slices[ue % len(self.slices)]
                rows.append({
                    "timestamp": t,
                    "ue_id": f"UE_{ue:04d}",
                    "cell_id": f"CELL_{cell:03d}",
                    "slice": slice_id,
                    "cqi": cqi,
                    "sinr_db": round(sinr, 2),
                    "rsrp_dbm": round(rsrp, 2),
                    "rsrq_db": round(rsrq, 2),
                    "mcs": mcs,
                    "prb_allocated": prb,
                    "buffer_occupancy": round(buf, 3),
                    "throughput_mbps": round(tp, 2),
                    "latency_ms": round(latency, 2),
                    "packet_loss": round(pkt_loss, 4),
                })
        path = self.output_dir / "ran_kpi_dataset.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _generate_mobility(self) -> Path:
        rows = []
        for ue in range(self.num_ues):
            x, y = self.rng.uniform(0, 1000, 2)
            vx, vy = self.rng.uniform(-30, 30, 2)
            cell = int(self.rng.integers(0, self.num_cells))
            for t in range(0, self.num_steps, 5):
                x += vx * 0.1
                y += vy * 0.1
                if self.rng.random() < 0.02:
                    vx, vy = self.rng.uniform(-30, 30, 2)
                neighbors = [f"CELL_{(cell + d) % self.num_cells:03d}" for d in range(1, 4)]
                rows.append({
                    "timestamp": t,
                    "ue_id": f"UE_{ue:04d}",
                    "cell_id": f"CELL_{cell:03d}",
                    "x_m": round(x, 2),
                    "y_m": round(y, 2),
                    "velocity_mps": round(float(np.hypot(vx, vy)), 2),
                    "direction_deg": round(float(np.degrees(np.arctan2(vy, vx))), 2),
                    "neighbor_cells": json.dumps(neighbors),
                    "rsrp_dbm": round(float(self.rng.normal(-95, 10)), 2),
                    "handover_pending": int(self.rng.random() < 0.05),
                })
                if self.rng.random() < 0.03:
                    cell = (cell + int(self.rng.integers(1, self.num_cells))) % self.num_cells
        path = self.output_dir / "mobility_traces.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _generate_security(self) -> Path:
        threat_types = ["normal", "jamming", "spoofing", "ddos", "rogue_gnb", "pilot_contamination"]
        rows = []
        for i in range(5000):
            threat = self.rng.choice(threat_types, p=[0.85, 0.04, 0.03, 0.03, 0.03, 0.02])
            is_attack = threat != "normal"
            rows.append({
                "event_id": f"SEC_{i:06d}",
                "timestamp": int(self.rng.integers(0, self.num_steps)),
                "source_ip": f"10.{self.rng.integers(0,255)}.{self.rng.integers(0,255)}.{self.rng.integers(1,254)}",
                "target_cell": f"CELL_{int(self.rng.integers(0, self.num_cells)):03d}",
                "threat_type": threat,
                "is_attack": int(is_attack),
                "packet_rate_pps": round(float(self.rng.exponential(1000 if is_attack else 100)), 2),
                "auth_failures": int(self.rng.poisson(3 if is_attack else 0.1)),
                "spectrum_anomaly_score": round(float(self.rng.uniform(0.7, 1.0) if is_attack else self.rng.uniform(0, 0.3)), 3),
                "flow_entropy": round(float(self.rng.uniform(0, 0.5) if is_attack else self.rng.uniform(0.5, 1.0)), 3),
                "bytes_transferred": int(self.rng.integers(100, 1000000)),
            })
        path = self.output_dir / "security_events.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _generate_energy(self) -> Path:
        rows = []
        for t in range(self.num_steps):
            for cell in range(self.num_cells):
                load = float(self.rng.uniform(0.1, 0.95))
                sleep = int(load < 0.15 and self.rng.random() < 0.3)
                power = 800 * load if not sleep else 50
                rows.append({
                    "timestamp": t,
                    "cell_id": f"CELL_{cell:03d}",
                    "power_consumption_w": round(power, 2),
                    "cell_utilization": round(load, 3),
                    "sleep_state": sleep,
                    "renewable_pct": round(float(self.rng.uniform(0, 40)), 2),
                    "carbon_intensity_gco2_kwh": round(float(self.rng.uniform(200, 500)), 2),
                    "traffic_demand_mbps": round(load * self.rng.uniform(50, 200), 2),
                })
        path = self.output_dir / "energy_metrics.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _generate_slice_utilization(self) -> Path:
        rows = []
        for t in range(self.num_steps):
            for sl in self.slices:
                util = float(self.rng.uniform(0.2, 0.9))
                rows.append({
                    "timestamp": t,
                    "slice": sl,
                    "prb_utilization": round(util, 3),
                    "active_ues": int(self.rng.integers(5, self.num_ues // 2)),
                    "sla_compliance": round(float(self.rng.uniform(0.9, 1.0)), 4),
                    "throughput_mbps": round(util * self.rng.uniform(100, 500), 2),
                    "latency_p99_ms": round(float(self.rng.exponential(8)), 2),
                })
        path = self.output_dir / "slice_utilization.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _generate_handover_events(self) -> Path:
        rows = []
        for i in range(800):
            success = int(self.rng.random() < 0.985)
            rows.append({
                "event_id": f"HO_{i:05d}",
                "timestamp": int(self.rng.integers(0, self.num_steps)),
                "ue_id": f"UE_{int(self.rng.integers(0, self.num_ues)):04d}",
                "source_cell": f"CELL_{int(self.rng.integers(0, self.num_cells)):03d}",
                "target_cell": f"CELL_{int(self.rng.integers(0, self.num_cells)):03d}",
                "ho_type": self.rng.choice(["intra_freq", "inter_freq", "inter_rat"]),
                "rsrp_source_dbm": round(float(self.rng.normal(-95, 8)), 2),
                "rsrp_target_dbm": round(float(self.rng.normal(-88, 8)), 2),
                "velocity_mps": round(float(self.rng.uniform(0, 35)), 2),
                "success": success,
                "failure_cause": "" if success else self.rng.choice(["rlf", "too_early", "too_late", "ping_pong"]),
                "delay_ms": round(float(self.rng.uniform(20, 80)), 2),
            })
        path = self.output_dir / "handover_events.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_metadata(self) -> Path:
        meta = {
            "generator": "RANDatasetGenerator",
            "seed": self.seed,
            "num_cells": self.num_cells,
            "num_ues": self.num_ues,
            "num_steps": self.num_steps,
            "slices": self.slices,
            "3gpp_alignment": ["TS 38.215", "TS 38.214", "TS 28.552", "TS 33.501"],
        }
        path = self.output_dir / "dataset_metadata.json"
        save_json(meta, path)
        return path


def generate_datasets(seed: int = 42) -> dict[str, Path]:
    return RANDatasetGenerator(seed=seed).generate_all()
