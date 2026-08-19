"""Maps agent decisions to RAN parameter updates via Digital Twin API."""

from __future__ import annotations

import uuid
from typing import Any

from src.agents.base_agent import AgentAction
from src.api.schemas import CellState, KPIStats, UEState
from src.api.state_store import RANStateStore


class RANParameterService:
    def __init__(self, store: RANStateStore):
        self.store = store

    def get_all_cells(self) -> list[CellState]:
        return [
            CellState(
                cell_id=c.cell_id, load=c.load, power_w=c.power_w, sleep=c.sleep,
                tx_power_dbm=getattr(c, "tx_power_dbm", 30),
                prb_utilization=c.load,
            )
            for c in self.store.twin.cells.values()
        ]

    def get_all_ues(self, limit: int = 50) -> list[UEState]:
        ues = []
        for u in list(self.store.twin.ues.values())[:limit]:
            ues.append(UEState(
                ue_id=u.ue_id, cell_id=u.cell_id, slice=u.slice,
                cqi=u.cqi, sinr_db=u.sinr_db, throughput_mbps=u.throughput_mbps,
                latency_ms=u.latency_ms,
                mcs=int(u.cqi * 1.5), prb_allocated=5, buffer_occupancy=u.buffer,
            ))
        return ues

    def update_cell(self, cell_id: str, params: dict) -> dict:
        if cell_id not in self.store.twin.cells:
            raise ValueError(f"Cell {cell_id} not found")
        cell = self.store.twin.cells[cell_id]
        before = {"load": cell.load, "power_w": cell.power_w, "sleep": cell.sleep}
        if "power_w" in params:
            cell.power_w = float(params["power_w"])
        if "tx_power_dbm" in params:
            cell.power_w = float(params["tx_power_dbm"]) * 10
            setattr(cell, "tx_power_dbm", params["tx_power_dbm"])
        if "sleep" in params:
            cell.sleep = bool(params["sleep"])
        if "load" in params:
            cell.load = float(params["load"])
        after = {"load": cell.load, "power_w": cell.power_w, "sleep": cell.sleep}
        change = self.store.log_parameter_change("api", cell_id, before, after, "manual")
        return {"cell_id": cell_id, "change": change}

    def update_ue(self, ue_id: str, params: dict) -> dict:
        if ue_id not in self.store.twin.ues:
            raise ValueError(f"UE {ue_id} not found")
        ue = self.store.twin.ues[ue_id]
        before = {"cqi": ue.cqi, "throughput": ue.throughput_mbps, "latency": ue.latency_ms}
        for k in ("cqi", "sinr_db", "throughput_mbps", "latency_ms", "buffer"):
            if k in params:
                setattr(ue, k, float(params[k]))
        if "prb_allocated" in params:
            ue.throughput_mbps *= (1 + int(params["prb_allocated"]) / 50)
        after = {"cqi": ue.cqi, "throughput": ue.throughput_mbps, "latency": ue.latency_ms}
        change = self.store.log_parameter_change("api", ue_id, before, after, "manual")
        return {"ue_id": ue_id, "change": change}

    def apply_manual_update(self, req) -> dict:
        kpi_before = self.store.snapshot_before()
        updates = []
        if req.cell_id:
            updates.append(self.update_cell(req.cell_id, req.parameters))
        if req.ue_id:
            updates.append(self.update_ue(req.ue_id, req.parameters))
        self.store.twin.step()
        kpi_after = self.store.record_kpi("manual_update")
        return {
            "updates": updates,
            "kpi_before": kpi_before.model_dump(),
            "kpi_after": kpi_after.model_dump(),
            "comparison": self.store.compare_kpi(kpi_before, kpi_after),
        }

    def run_agents_and_apply(self, intent: str = "", cell_id: str = "CELL_000",
                              agents: list[str] | None = None) -> dict[str, Any]:
        action_id = str(uuid.uuid4())[:8]
        kpi_before = self.store.snapshot_before()

        obs = self.store.controller.build_observation(
            context={"cell_id": cell_id, "operator_intent": intent, "slice": "eMBB"},
        )
        result = self.store.controller.run_all_agents(
            observation=obs, intent=intent, agent_filter=agents,
            kpi_before=kpi_before.model_dump(),
        )
        approved: list[AgentAction] = result["approved_actions"]
        twin_actions = [{"action_type": a.action_type, "parameters": a.parameters} for a in approved]
        param_updates = self._apply_to_twin(approved, cell_id)

        self.store.twin.apply_actions(twin_actions)
        self.store.twin.step(twin_actions)
        kpi_after = self.store.record_kpi(f"agent_run_{action_id}")

        self.store.log_action(
            action_id, [a.agent_id for a in approved],
            kpi_before, kpi_after, param_updates, result["super_agent_decision"],
        )
        return {
            "action_id": action_id,
            "agents_invoked": [a.agent_id for a in approved],
            "actions": twin_actions,
            "kpi_before": kpi_before,
            "kpi_after": kpi_after,
            "parameter_updates": param_updates,
            "super_agent_decision": result["super_agent_decision"],
        }

    def _apply_to_twin(self, actions: list[AgentAction], cell_id: str) -> list[dict]:
        updates = []
        twin = self.store.twin
        for act in actions:
            p = act.parameters
            if act.action_type == "schedule":
                for ue in list(twin.ues.values())[:10]:
                    before = {"prb": 5, "throughput": ue.throughput_mbps}
                    prb = p.get("prb_assignment", 5)
                    ue.throughput_mbps *= (1 + prb / 30)
                    ue.latency_ms = max(0.5, ue.latency_ms * (1 - p.get("scheduling_priority", 0.3) * 0.05))
                    updates.append(self.store.log_parameter_change(
                        "agent", ue.ue_id, before,
                        {"prb": prb, "throughput": ue.throughput_mbps}, act.agent_id,
                    ))
            elif act.action_type == "resource_allocation":
                if cell_id in twin.cells:
                    c = twin.cells[cell_id]
                    before = {"power_w": c.power_w, "bandwidth_mhz": c.bandwidth_mhz}
                    c.power_w = p.get("power_dbm", 30) * 10
                    c.bandwidth_mhz = p.get("bandwidth_mhz", 100)
                    c.mimo_streams = p.get("mimo_streams", 4)
                    c.spectral_efficiency = p.get("spectral_efficiency", 0.5)
                    c.carrier_aggregation = p.get("carrier_aggregation", True)
                    updates.append(self.store.log_parameter_change(
                        "agent", cell_id, before,
                        {"power_w": c.power_w, "bandwidth_mhz": c.bandwidth_mhz, "mimo": c.mimo_streams},
                        act.agent_id,
                    ))
            elif act.action_type == "carbon_reduction":
                before = {"carbon_intensity": twin.carbon_intensity_gco2_kwh, "renewable": twin.renewable_pct}
                twin.carbon_intensity_gco2_kwh = p.get("carbon_intensity_gco2_kwh", twin.carbon_intensity_gco2_kwh)
                twin.renewable_pct = p.get("renewable_routing_pct", twin.renewable_pct)
                scale = 1 - p.get("carbon_reduction_pct", 0) / 200
                for c in twin.cells.values():
                    c.power_w = max(50, c.power_w * scale)
                updates.append(self.store.log_parameter_change(
                    "agent", "network", before,
                    {"carbon_intensity": twin.carbon_intensity_gco2_kwh, "reduction_pct": p.get("carbon_reduction_pct")},
                    act.agent_id,
                ))
            elif act.action_type == "ran_sleep":
                for i, c in enumerate(twin.cells.values()):
                    if p.get("sleep_mode") and i < p.get("cells_to_sleep", 2):
                        c.sleep, c.power_w = True, 45
                updates.append(self.store.log_parameter_change("agent", cell_id, {}, p, act.agent_id))
            elif act.action_type == "renewable_energy":
                twin.renewable_pct = p.get("renewable_routing_pct", twin.renewable_pct)
                updates.append(self.store.log_parameter_change("agent", "renewable", {}, p, act.agent_id))
            elif act.action_type == "edge_inference":
                for ue in list(twin.ues.values())[:15]:
                    ue.latency_ms = max(0.3, ue.latency_ms * (1 - p.get("latency_reduction_pct", 0) / 200))
                updates.append(self.store.log_parameter_change("agent", "edge_mec", {}, p, act.agent_id))
            elif act.action_type == "green_slice":
                for c in twin.cells.values():
                    c.power_w = max(50, c.power_w * (1 - p.get("energy_saving_pct", 0) / 300))
                updates.append(self.store.log_parameter_change("agent", p.get("slice", "eMBB"), {}, p, act.agent_id))
            elif act.action_type == "traffic_optimization":
                for c in twin.cells.values():
                    c.load = max(0.1, c.load * (1 - p.get("congestion_reduction_pct", 0) / 400))
                for ue in list(twin.ues.values())[:20]:
                    ue.throughput_mbps *= 1 + p.get("peak_throughput_boost_pct", 0) / 200
                    ue.buffer = max(0.05, ue.buffer * (1 - p.get("congestion_reduction_pct", 0) / 300))
                updates.append(self.store.log_parameter_change("agent", "traffic", {}, p, act.agent_id))
            elif act.action_type == "agent_optimization":
                target = p.get("target_agent", "unknown")
                updates.append(self.store.log_parameter_change(
                    "agent_optimizer", target,
                    {"status": "degraded"},
                    {"action": p.get("optimization_action"), "recovery_pct": p.get("expected_recovery_pct")},
                    act.agent_id,
                ))
            elif act.action_type == "coordination":
                updates.append(self.store.log_parameter_change(
                    "coordination", "multi_agent",
                    {"conflicts": p.get("conflicts_detected", 0)},
                    {
                        "strategy": p.get("coordination_strategy"),
                        "resolved": p.get("conflicts_resolved", 0),
                        "agents": p.get("agents_coordinated", 0),
                    },
                    act.agent_id,
                ))
            elif act.action_type == "energy":
                for c in twin.cells.values():
                    before = {"power_w": c.power_w, "sleep": c.sleep}
                    if p.get("sleep_mode"):
                        c.sleep, c.power_w = True, 50
                    else:
                        c.sleep = False
                        c.power_w = 400 * p.get("power_scale_factor", 1)
                    updates.append(self.store.log_parameter_change(
                        "agent", c.cell_id, before,
                        {"power_w": c.power_w, "sleep": c.sleep}, act.agent_id,
                    ))
            elif act.action_type == "beamforming":
                twin.beamforming_gain_db = p.get("beamforming_gain_db", 3.0)
                for ue in twin.ues.values():
                    ue.sinr_db = min(30, ue.sinr_db + twin.beamforming_gain_db * 0.3)
            elif act.action_type == "csi":
                twin.csi_accuracy = p.get("csi_accuracy", 0.92)
            elif act.action_type == "channel_estimation":
                twin.csi_accuracy = p.get("csi_accuracy", 0.9)
            elif act.action_type == "qos":
                for ue in twin.ues.values():
                    if ue.slice == p.get("slice", "eMBB") and p.get("qos_action") == "boost_priority":
                        ue.latency_ms *= 0.85
                        ue.throughput_mbps *= 1.1
            elif act.action_type == "slice":
                target_slice = p.get("slice", "eMBB")
                action = p.get("slice_action", "maintain")
                prb_share = p.get("prb_share_pct", 0.2)
                for ue in twin.ues.values():
                    if ue.slice != target_slice:
                        continue
                    if action in ("boost_priority", "rebalance"):
                        ue.throughput_mbps *= 1 + prb_share
                        ue.latency_ms = max(0.5, ue.latency_ms * 0.9)
                    if p.get("isolation_level") == "strict":
                        ue.latency_ms = max(0.5, ue.latency_ms * 0.95)
            elif act.action_type == "air_interface":
                mcs = p.get("optimal_mcs", 15)
                for ue in list(twin.ues.values())[:10]:
                    ue.throughput_mbps *= 1 + mcs / 50
                    ue.cqi = min(15, ue.cqi + mcs / 30)
            elif act.action_type == "digital_twin" and p.get("twin_action") == "deploy_policy":
                twin.csi_accuracy = max(twin.csi_accuracy, p.get("fidelity_score", 0.9))
            elif act.action_type == "spectrum" and p.get("spectrum_action") == "reallocate":
                for c in twin.cells.values():
                    c.load = max(0.1, c.load * 0.9)
            elif act.action_type == "self_healing" and p.get("healing_required"):
                for ue in twin.ues.values():
                    ue.throughput_mbps = max(ue.throughput_mbps, ue.throughput_mbps * 1.05)
                    ue.latency_ms = max(0.5, ue.latency_ms * 0.95)
            elif act.action_type == "mobility" and p.get("handover_recommended"):
                target = p.get("target_cell", cell_id)
                for ue in list(twin.ues.values())[:3]:
                    ue.cell_id = target
        return updates
