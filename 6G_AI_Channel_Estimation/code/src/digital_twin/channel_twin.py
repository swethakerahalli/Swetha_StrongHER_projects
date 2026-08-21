"""6G radio digital twin: cells, UEs, RIS, LEO, attacks, fidelity."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.common.utils import project_root


@dataclass
class TwinCell:
    cell_id: str
    x: float
    y: float
    scenario: str
    snr_db: float
    nmse: float
    load: float
    beam_index: int
    attack_type: str
    trust: float
    ue_count: int
    fidelity: float


class ChannelDigitalTwin:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.step_idx = 0
        self.history: list[dict] = []
        self.cells: list[TwinCell] = []
        self._load_seed()

    def _load_seed(self) -> None:
        path = project_root() / "data" / "datasets" / "digital_twin_states.csv"
        if not path.exists():
            self.cells = [
                TwinCell(f"CELL_{i:03d}", 100 + (i % 6) * 150, 100 + (i // 6) * 150, "UMa", 12, 0.05, 0.5, 0, "normal", 0.9, 16, 0.92)
                for i in range(12)
            ]
            return
        df = pd.read_csv(path)
        latest = df[df["step"] == df["step"].min()]
        if latest.empty:
            latest = df.head(24)
        self.cells = [
            TwinCell(
                row.cell_id, float(row.x_m), float(row.y_m), str(row.scenario),
                float(row.snr_db), float(row.nmse_ai), float(row.load), int(row.beam_index),
                str(row.attack_type), float(row.trust_score), int(row.ue_count), float(row.twin_fidelity),
            )
            for row in latest.itertuples()
        ]
        if not self.cells:
            self.cells = [
                TwinCell(f"CELL_{i:03d}", 90 + (i % 6) * 155, 120 + (i // 6) * 180, "UMa", 12, 0.05, 0.5, 0, "normal", 0.9, 16, 0.92)
                for i in range(24)
            ]
        self._full = df

    def observe(self) -> dict:
        return {
            "step": self.step_idx,
            "n_cells": len(self.cells),
            "mean_snr_db": round(float(np.mean([c.snr_db for c in self.cells])), 2),
            "mean_nmse": round(float(np.mean([c.nmse for c in self.cells])), 5),
            "mean_fidelity": round(float(np.mean([c.fidelity for c in self.cells])), 4),
            "attack_cells": int(sum(c.attack_type != "normal" for c in self.cells)),
            "mean_load": round(float(np.mean([c.load for c in self.cells])), 3),
        }

    def step(self, policy: str = "hold") -> dict:
        for cell in self.cells:
            cell.snr_db = float(np.clip(cell.snr_db + self.rng.normal(0, 0.4), -4, 32))
            if policy == "reduce_pilots":
                cell.nmse *= 1.01
            elif policy.startswith("mitigate") or policy == "increase_pilots":
                cell.nmse *= 0.96
                if cell.attack_type != "normal" and self.rng.random() < 0.35:
                    cell.attack_type = "normal"
                    cell.trust = min(1.0, cell.trust + 0.08)
            cell.nmse = float(np.clip(cell.nmse * self.rng.uniform(0.97, 1.03), 0.002, 1.5))
            cell.load = float(np.clip(cell.load + self.rng.normal(0, 0.03), 0.15, 0.98))
            cell.fidelity = float(np.clip(0.9 + 0.08 * (1 - cell.nmse) - 0.05 * (cell.attack_type != "normal"), 0.7, 0.995))
        self.step_idx += 1
        state = self.observe()
        self.history.append(state)
        return state

    def visualization_data(self) -> dict:
        if not self.cells:
            self._load_seed()
        cells_out = []
        ues_out = []
        for i, c in enumerate(self.cells):
            gx = 90.0 + (i % 6) * 155.0
            gy = 120.0 + (i // 6) * 180.0
            cells_out.append({
                "id": c.cell_id,
                "x": gx,
                "y": gy,
                "snr": round(c.snr_db, 2),
                "nmse": round(c.nmse, 5),
                "load": round(c.load, 3),
                "attack": c.attack_type,
                "fidelity": round(c.fidelity, 4),
                "scenario": c.scenario,
                "ues": c.ue_count,
                "beam": c.beam_index,
                "radius": 28 + c.load * 22,
            })
            n_ue = max(4, min(10, int(c.ue_count / 4)))
            for u in range(n_ue):
                ang = (u / n_ue) * 2 * np.pi + float(self.rng.uniform(-0.2, 0.2))
                dist = 25 + float(self.rng.uniform(0, 45))
                ues_out.append({
                    "id": f"{c.cell_id}_UE{u}",
                    "x": gx + dist * np.cos(ang),
                    "y": gy + dist * np.sin(ang),
                    "cell": c.cell_id,
                    "attack": c.attack_type,
                })
        return {
            "step": self.step_idx,
            "fidelity": self.fidelity_score(),
            "cells": cells_out,
            "ues": ues_out,
            "ris": [{"id": f"RIS_{i}", "x": 140 + i * 200, "y": 820} for i in range(4)],
            "satellites": [{"id": f"LEO_{i}", "x": 180 + i * 200, "y": 48} for i in range(4)],
        }

    def fidelity_score(self) -> float:
        return round(float(np.mean([c.fidelity for c in self.cells])), 4)
