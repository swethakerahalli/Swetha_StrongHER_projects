"""RAN Digital Twin - virtual network for policy validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.common.utils import load_config


@dataclass
class UETwin:
    ue_id: str
    cell_id: str
    slice: str
    cqi: float = 8.0
    sinr_db: float = 12.0
    buffer: float = 0.3
    throughput_mbps: float = 10.0
    latency_ms: float = 5.0
    x: float = 0.0
    y: float = 0.0
    velocity: float = 0.0


@dataclass
class CellTwin:
    cell_id: str
    load: float = 0.5
    power_w: float = 400.0
    sleep: bool = False
    connected_ues: list[str] = field(default_factory=list)
    bandwidth_mhz: int = 100
    mimo_streams: int = 4
    tx_power_dbm: float = 40.0
    spectral_efficiency: float = 0.5
    carrier_aggregation: bool = True


class RANDigitalTwin:
    """Virtual RAN environment aligned with 3GPP NDT (TS 28.104)."""

    def __init__(self, seed: int = 42):
        cfg = load_config("system_config.json")
        sim = cfg["simulation"]
        self.rng = np.random.default_rng(seed)
        self.num_cells = sim["num_cells"]
        self.num_ues = sim["num_ues"]
        self.slices = list(cfg["network_slices"].keys())
        self.cells: dict[str, CellTwin] = {}
        self.ues: dict[str, UETwin] = {}
        self.history: list[dict] = []
        self.beamforming_gain_db: float = 3.0
        self.csi_accuracy: float = 0.92
        self.carbon_intensity_gco2_kwh: float = 380.0
        self.renewable_pct: float = 15.0
        self._init_network()

    def _init_network(self) -> None:
        for c in range(self.num_cells):
            cid = f"CELL_{c:03d}"
            self.cells[cid] = CellTwin(cell_id=cid)
        for u in range(self.num_ues):
            uid = f"UE_{u:04d}"
            cell = f"CELL_{u % self.num_cells:03d}"
            sl = self.slices[u % len(self.slices)]
            self.ues[uid] = UETwin(
                ue_id=uid, cell_id=cell, slice=sl,
                x=float(self.rng.uniform(0, 1000)),
                y=float(self.rng.uniform(0, 1000)),
                velocity=float(self.rng.uniform(0, 20)),
            )
            self.cells[cell].connected_ues.append(uid)
        self.history.append(self.observe())

    def observe(self) -> dict[str, Any]:
        return {
            "timestamp": len(self.history),
            "cells": {k: {"load": v.load, "power_w": v.power_w, "sleep": v.sleep} for k, v in self.cells.items()},
            "ues": {
                k: {
                    "cell": v.cell_id, "cqi": v.cqi, "sinr": v.sinr_db,
                    "throughput": v.throughput_mbps, "latency": v.latency_ms,
                }
                for k, v in self.ues.items()
            },
            "avg_throughput": float(np.mean([u.throughput_mbps for u in self.ues.values()])),
            "avg_latency": float(np.mean([u.latency_ms for u in self.ues.values()])),
            "total_power_w": sum(c.power_w for c in self.cells.values()),
        }

    def apply_actions(self, actions: list[dict]) -> None:
        for act in actions:
            atype = act.get("action_type", "")
            params = act.get("parameters", {})
            if atype == "schedule":
                boost = params.get("predicted_throughput_mbps", 10) / 50
                for ue in self.ues.values():
                    ue.throughput_mbps *= (1 + boost * 0.1)
                    ue.latency_ms = max(0.5, ue.latency_ms * (1 - boost * 0.05))
            elif atype == "resource_allocation":
                for cell in self.cells.values():
                    cell.power_w = params.get("power_dbm", 30) * 10
                    cell.tx_power_dbm = params.get("power_dbm", 40)
                    cell.bandwidth_mhz = params.get("bandwidth_mhz", 100)
                    cell.mimo_streams = params.get("mimo_streams", 4)
                    cell.spectral_efficiency = params.get("spectral_efficiency", 0.5)
                    cell.carrier_aggregation = params.get("carrier_aggregation", True)
            elif atype == "energy":
                for cell in self.cells.values():
                    if params.get("sleep_mode"):
                        cell.sleep = True
                        cell.power_w = 50
                    else:
                        cell.sleep = False
                        cell.power_w = 400 * params.get("power_scale_factor", 1)
                    if params.get("carbon_aware_mode"):
                        self.carbon_intensity_gco2_kwh = max(180, self.carbon_intensity_gco2_kwh * 0.95)
            elif atype == "carbon_reduction":
                self.carbon_intensity_gco2_kwh = params.get("carbon_intensity_gco2_kwh", self.carbon_intensity_gco2_kwh)
                self.renewable_pct = params.get("renewable_routing_pct", self.renewable_pct)
                scale = 1 - params.get("carbon_reduction_pct", 0) / 200
                for cell in self.cells.values():
                    cell.power_w = max(50, cell.power_w * scale)
            elif atype == "ran_sleep":
                n_sleep = params.get("cells_to_sleep", 0)
                slept = 0
                for cell in self.cells.values():
                    if slept < n_sleep and params.get("sleep_mode"):
                        cell.sleep = True
                        cell.power_w = 45
                        slept += 1
                    elif not params.get("sleep_mode"):
                        cell.sleep = False
                        cell.power_w = max(cell.power_w, 200)
            elif atype == "renewable_energy":
                self.renewable_pct = params.get("renewable_routing_pct", self.renewable_pct)
                if params.get("green_power_priority"):
                    self.carbon_intensity_gco2_kwh = max(150, self.carbon_intensity_gco2_kwh * 0.92)
            elif atype == "edge_inference":
                reduction = params.get("latency_reduction_pct", 0) / 100
                for ue in self.ues.values():
                    ue.latency_ms = max(0.3, ue.latency_ms * (1 - reduction * 0.5))
            elif atype == "green_slice":
                cap = params.get("power_cap_w")
                saving = params.get("energy_saving_pct", 0) / 100
                target = params.get("slice", "eMBB")
                for cell in self.cells.values():
                    if cap:
                        cell.power_w = min(cell.power_w, cap / max(len(self.cells), 1))
                    else:
                        cell.power_w = max(50, cell.power_w * (1 - saving * 0.3))
                for ue in self.ues.values():
                    if ue.slice == target:
                        ue.throughput_mbps *= 1 + params.get("slice_efficiency_gain_pct", 0) / 200
            elif atype == "traffic_optimization":
                reduction = params.get("congestion_reduction_pct", 0) / 100
                boost = params.get("peak_throughput_boost_pct", 0) / 100
                for cell in self.cells.values():
                    cell.load = max(0.1, cell.load * (1 - reduction * 0.4))
                for ue in self.ues.values():
                    ue.throughput_mbps *= 1 + boost * 0.15
                    ue.buffer = max(0.05, ue.buffer * (1 - reduction * 0.35))
                    ue.latency_ms = max(0.4, ue.latency_ms * (1 - reduction * 0.2))
            elif atype == "mobility" and params.get("handover_recommended"):
                target = params.get("target_cell")
                for ue in list(self.ues.values())[:5]:
                    old = ue.cell_id
                    if old in self.cells and ue.ue_id in self.cells[old].connected_ues:
                        self.cells[old].connected_ues.remove(ue.ue_id)
                    ue.cell_id = target
                    if target in self.cells:
                        self.cells[target].connected_ues.append(ue.ue_id)
            elif atype == "beamforming":
                gain = params.get("beamforming_gain_db", 3.0)
                self.beamforming_gain_db = gain
                for ue in self.ues.values():
                    ue.sinr_db = min(30, ue.sinr_db + gain * 0.3)
            elif atype == "csi" or atype == "channel_estimation":
                acc = params.get("csi_accuracy", 0.9)
                self.csi_accuracy = acc
                for ue in self.ues.values():
                    ue.cqi = min(15, ue.cqi + acc * 0.5)
            elif atype == "qos":
                for ue in self.ues.values():
                    if params.get("qos_action") == "boost_priority":
                        ue.latency_ms *= 0.85
                        ue.throughput_mbps *= 1.05
            elif atype == "slice":
                target_slice = params.get("slice", "eMBB")
                action = params.get("slice_action", "maintain")
                prb_share = params.get("prb_share_pct", 0.2)
                for ue in self.ues.values():
                    if ue.slice != target_slice:
                        continue
                    if action in ("boost_priority", "rebalance"):
                        ue.throughput_mbps *= 1 + prb_share
                        ue.latency_ms = max(0.5, ue.latency_ms * 0.9)
            elif atype == "air_interface":
                mcs = params.get("optimal_mcs", 15)
                for ue in self.ues.values():
                    ue.throughput_mbps *= 1 + mcs / 60
            elif atype == "digital_twin":
                self.csi_accuracy = max(self.csi_accuracy, params.get("fidelity_score", 0.9))
            elif atype == "spectrum" and params.get("spectrum_action") == "reallocate":
                for cell in self.cells.values():
                    cell.load = max(0.1, cell.load * 0.92)
            elif atype == "self_healing" and params.get("healing_required"):
                for ue in self.ues.values():
                    ue.latency_ms = max(0.5, ue.latency_ms * 0.9)
            elif atype == "security" and params.get("threat_detected"):
                for ue in list(self.ues.values())[:2]:
                    ue.throughput_mbps *= 0.9

    def step(self, actions: list[dict] | None = None) -> dict[str, Any]:
        if actions:
            self.apply_actions(actions)
        for ue in self.ues.values():
            ue.cqi = float(np.clip(ue.cqi + self.rng.normal(0, 0.5), 0, 15))
            ue.sinr_db = float(np.clip(ue.sinr_db + self.rng.normal(0, 1), -5, 30))
            ue.buffer = float(np.clip(ue.buffer + self.rng.uniform(-0.1, 0.15), 0, 1))
            ue.throughput_mbps = max(0.5, (ue.cqi / 15) * self.rng.exponential(30))
            ue.latency_ms = max(0.5, self.rng.exponential(4))
            ue.x += ue.velocity * 0.01
        for cell in self.cells.values():
            n = len(cell.connected_ues) or 1
            cell.load = min(1.0, n / (self.num_ues / self.num_cells))
        state = self.observe()
        self.history.append(state)
        return state

    def fidelity_score(self) -> float:
        if len(self.history) < 2:
            return 0.95
        tp_vars = [h["avg_throughput"] for h in self.history[-10:]]
        stability = 1.0 - min(1.0, float(np.std(tp_vars)) / (float(np.mean(tp_vars)) + 1))
        return round(0.85 + 0.15 * stability, 3)

    def _cell_positions(self) -> dict[str, tuple[float, float]]:
        """Hexagonal layout for gNodeB sites in the twin coordinate space."""
        cx, cy, r = 500.0, 500.0, 220.0
        positions: dict[str, tuple[float, float]] = {}
        for i, cid in enumerate(sorted(self.cells.keys())):
            if i == 0:
                positions[cid] = (cx, cy)
            else:
                angle = (i - 1) * (2 * np.pi / max(self.num_cells - 1, 1)) - np.pi / 2
                positions[cid] = (cx + r * np.cos(angle), cy + r * np.sin(angle))
        return positions

    def visualization_data(self) -> dict[str, Any]:
        """Topology + metrics for dashboard digital twin canvas."""
        positions = self._cell_positions()
        cells = []
        for cid, cell in self.cells.items():
            px, py = positions[cid]
            cells.append({
                "cell_id": cid,
                "x": round(px, 1),
                "y": round(py, 1),
                "load": round(cell.load, 3),
                "power_w": round(cell.power_w, 1),
                "sleep": cell.sleep,
                "ue_count": len(cell.connected_ues),
                "radius": round(70 + cell.load * 50, 1),
            })
        ues = []
        for uid, ue in self.ues.items():
            ues.append({
                "ue_id": uid,
                "x": round(ue.x % 1000, 1),
                "y": round(ue.y % 1000, 1),
                "cell_id": ue.cell_id,
                "slice": ue.slice,
                "throughput": round(ue.throughput_mbps, 2),
                "latency": round(ue.latency_ms, 2),
                "cqi": round(ue.cqi, 1),
                "sinr": round(ue.sinr_db, 1),
            })
        obs = self.observe()
        return {
            "cells": cells,
            "ues": ues,
            "bounds": {"width": 1000, "height": 1000},
            "fidelity": self.fidelity_score(),
            "csi_accuracy": round(self.csi_accuracy, 3),
            "beamforming_gain_db": round(self.beamforming_gain_db, 2),
            "slice_colors": {"eMBB": "#1da1f2", "URLLC": "#ff5252", "mMTC": "#00c853"},
            "timestamp": obs["timestamp"],
            "avg_throughput": obs["avg_throughput"],
            "avg_latency": obs["avg_latency"],
            "total_power_w": obs["total_power_w"],
            "num_cells": self.num_cells,
            "num_ues": self.num_ues,
        }

    def what_if(self, policy: dict) -> dict[str, float]:
        snapshot = {k: UETwin(**{f.name: getattr(v, f.name) for f in UETwin.__dataclass_fields__.values()})
                    for k, v in self.ues.items()}
        self.apply_actions([{"action_type": policy.get("type", ""), "parameters": policy}])
        result = self.observe()
        for k, v in snapshot.items():
            self.ues[k] = v
        return {
            "predicted_throughput": result["avg_throughput"],
            "predicted_latency": result["avg_latency"],
            "predicted_power": result["total_power_w"],
        }
