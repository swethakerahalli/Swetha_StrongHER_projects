"""Coordination Agent — inter-agent conflict resolution and multi-domain action harmonization."""

from __future__ import annotations

import copy

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.agents.base_agent import AgentAction, AgentObservation, BaseAgent


class CoordinationAgent(BaseAgent):
    """Meta-orchestration layer: detects conflicting agent actions and harmonizes them."""

    FEATURE_KEYS = [
        "num_actions", "num_conflicts", "avg_confidence",
        "throughput_mbps", "total_power_w", "cell_utilization",
    ]

    CONFLICT_CHECKS = [
        ("energy", "schedule", "sleep_vs_high_prb"),
        ("energy", "traffic_optimization", "power_vs_peak_boost"),
        ("carbon_reduction", "traffic_optimization", "carbon_vs_traffic"),
        ("ran_sleep", "mobility", "sleep_vs_handover"),
        ("ran_sleep", "schedule", "sleep_vs_schedule"),
        ("slice", "qos", "slice_vs_qos"),
        ("green_slice", "traffic_optimization", "green_vs_traffic"),
        ("resource_allocation", "energy", "resource_vs_energy"),
    ]

    def __init__(self, model_path=None):
        super().__init__("coordination_agent", model_path)
        self.conflict_log: list[dict] = []
        self.coordination_cycles: int = 0

    def train(self, data: pd.DataFrame) -> dict[str, float]:
        df = data.copy()
        for col in self.FEATURE_KEYS:
            if col not in df.columns:
                df[col] = 0.5
        df["resolution_success"] = (
            (df["num_conflicts"] <= 3)
            | (df["avg_confidence"] > 0.75)
        ).astype(int)
        X = df[self.FEATURE_KEYS].values
        y = df["resolution_success"].values
        self.model = GradientBoostingClassifier(n_estimators=30, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        return {"accuracy": float(self.model.score(X, y)), "samples": len(df)}

    def coordinate(
        self,
        raw_actions: list[AgentAction],
        observation: AgentObservation,
    ) -> dict:
        """Detect conflicts among domain agents and return harmonized actions."""
        resolved = [copy.deepcopy(a) for a in raw_actions]
        conflicts: list[dict] = []
        action_map = {a.action_type: a for a in resolved}

        self._resolve_sleep_schedule(action_map, conflicts)
        self._resolve_energy_traffic(action_map, conflicts)
        self._resolve_carbon_traffic(action_map, conflicts)
        self._resolve_sleep_mobility(action_map, conflicts)
        self._resolve_slice_qos(action_map, conflicts)
        self._resolve_green_traffic(action_map, conflicts)
        self._resolve_resource_energy(action_map, conflicts)

        resolved = list(action_map.values())
        strategy = self._select_strategy(len(conflicts), observation)
        success_rate = self._estimate_success_rate(len(conflicts), len(resolved), observation)

        cycle = {
            "cycle": self.coordination_cycles + 1,
            "conflicts_detected": len(conflicts),
            "conflicts_resolved": len(conflicts),
            "strategy": strategy,
            "agents_coordinated": len(resolved),
            "resolution_success_pct": success_rate,
            "conflicts": conflicts,
        }
        self.conflict_log.append(cycle)
        self.coordination_cycles += 1

        return {
            "resolved_actions": resolved,
            "conflicts": conflicts,
            "coordination_strategy": strategy,
            "resolution_success_pct": success_rate,
            "agents_coordinated": len(resolved),
        }

    def predict(self, observation: AgentObservation) -> AgentAction:
        raw = observation.context.get("raw_actions", [])
        if raw and isinstance(raw[0], dict):
            raw = [
                AgentAction(
                    agent_id=r.get("agent", r.get("agent_id", "unknown")),
                    action_type=r.get("type", r.get("action_type", "")),
                    parameters=r.get("params", r.get("parameters", {})),
                    confidence=r.get("confidence", 0.8),
                )
                for r in raw
            ]
        result = self.coordinate(raw, observation)
        conf = 0.92 if result["conflicts"] else 0.98
        if self.is_trained:
            f = observation.features
            f = dict(f)
            f["num_actions"] = len(raw)
            f["num_conflicts"] = len(result["conflicts"])
            f["avg_confidence"] = (
                sum(a.confidence for a in raw) / max(len(raw), 1) if raw else 0.85
            )
            prob = float(self.model.predict_proba(self._to_array(f, self.FEATURE_KEYS))[0][1])
            conf = round(0.8 + prob * 0.18, 2)

        return AgentAction(
            agent_id=self.agent_id,
            action_type="coordination",
            parameters={
                "coordination_strategy": result["coordination_strategy"],
                "conflicts_detected": len(result["conflicts"]),
                "conflicts_resolved": len(result["conflicts"]),
                "resolution_success_pct": result["resolution_success_pct"],
                "agents_coordinated": result["agents_coordinated"],
                "conflicts": result["conflicts"][:8],
                "coordination_latency_ms": round(45 + len(result["conflicts"]) * 12, 1),
            },
            confidence=conf,
        )

    def _resolve_sleep_schedule(self, action_map: dict, conflicts: list) -> None:
        energy = action_map.get("energy")
        sched = action_map.get("schedule")
        if not energy or not sched:
            return
        if energy.parameters.get("sleep_mode") and sched.parameters.get("prb_assignment", 0) > 10:
            before = sched.parameters.get("prb_assignment")
            sched.parameters["prb_assignment"] = max(1, before // 2)
            conflicts.append({
                "pair": "energy↔schedule", "type": "sleep_vs_high_prb",
                "resolution": f"PRB {before}→{sched.parameters['prb_assignment']}",
            })

    def _resolve_energy_traffic(self, action_map: dict, conflicts: list) -> None:
        energy = action_map.get("energy")
        traffic = action_map.get("traffic_optimization")
        if not energy or not traffic:
            return
        peak = traffic.parameters.get("peak_throughput_boost_pct", 0)
        if energy.parameters.get("sleep_mode") and peak > 15:
            energy.parameters["sleep_mode"] = False
            energy.parameters["power_scale_factor"] = 0.75
            traffic.parameters["peak_throughput_boost_pct"] = round(peak * 0.7, 1)
            conflicts.append({
                "pair": "energy↔traffic", "type": "power_vs_peak_boost",
                "resolution": "Disable sleep; scale peak boost 70%",
            })

    def _resolve_carbon_traffic(self, action_map: dict, conflicts: list) -> None:
        carbon = action_map.get("carbon_reduction")
        traffic = action_map.get("traffic_optimization")
        if not carbon or not traffic:
            return
        carbon_cut = carbon.parameters.get("carbon_reduction_pct", 0)
        peak = traffic.parameters.get("peak_throughput_boost_pct", 0)
        if carbon_cut > 20 and peak > 20:
            carbon.parameters["power_scale_factor"] = max(
                0.65, carbon.parameters.get("power_scale_factor", 0.8),
            )
            traffic.parameters["reroute_pct"] = min(
                traffic.parameters.get("reroute_pct", 0.2), 0.35,
            )
            conflicts.append({
                "pair": "carbon↔traffic", "type": "carbon_vs_traffic",
                "resolution": "Balanced carbon cut + reroute cap",
            })

    def _resolve_sleep_mobility(self, action_map: dict, conflicts: list) -> None:
        sleep = action_map.get("ran_sleep")
        mob = action_map.get("mobility")
        if not sleep or not mob:
            return
        if sleep.parameters.get("sleep_mode") and mob.parameters.get("handover_recommended"):
            sleep.parameters["sleep_mode"] = False
            conflicts.append({
                "pair": "ran_sleep↔mobility", "type": "sleep_vs_handover",
                "resolution": "Wake cell for handover target",
            })

    def _resolve_slice_qos(self, action_map: dict, conflicts: list) -> None:
        sl = action_map.get("slice")
        qos = action_map.get("qos")
        if not sl or not qos:
            return
        share = sl.parameters.get("prb_share_pct", 0)
        boost = qos.parameters.get("priority_boost", 0)
        if share > 0.4 and boost > 0.5:
            sl.parameters["prb_share_pct"] = 0.38
            qos.parameters["priority_boost"] = round(boost * 0.85, 2)
            conflicts.append({
                "pair": "slice↔qos", "type": "slice_vs_qos",
                "resolution": "Cap slice share; moderate QoS boost",
            })

    def _resolve_green_traffic(self, action_map: dict, conflicts: list) -> None:
        green = action_map.get("green_slice")
        traffic = action_map.get("traffic_optimization")
        if not green or not traffic:
            return
        saving = green.parameters.get("energy_saving_pct", 0)
        peak = traffic.parameters.get("peak_throughput_boost_pct", 0)
        if saving > 15 and peak > 15:
            green.parameters["energy_saving_pct"] = round(saving * 0.8, 1)
            conflicts.append({
                "pair": "green_slice↔traffic", "type": "green_vs_traffic",
                "resolution": "Reduce green cap during peak traffic",
            })

    def _resolve_resource_energy(self, action_map: dict, conflicts: list) -> None:
        res = action_map.get("resource_allocation")
        energy = action_map.get("energy")
        if not res or not energy:
            return
        bw = res.parameters.get("bandwidth_mhz", 0)
        if energy.parameters.get("sleep_mode") and bw > 80:
            energy.parameters["sleep_mode"] = False
            conflicts.append({
                "pair": "resource↔energy", "type": "resource_vs_energy",
                "resolution": "Keep cell active for high bandwidth alloc",
            })

    @staticmethod
    def _select_strategy(num_conflicts: int, observation: AgentObservation) -> str:
        if num_conflicts == 0:
            return "consensus_pass_through"
        if num_conflicts <= 2:
            return "utility_weighted_consensus"
        intent = observation.context.get("parsed_intent", {})
        if intent.get("priority") == "energy":
            return "energy_first_arbitration"
        return "multi_objective_harmonization"

    @staticmethod
    def _estimate_success_rate(num_conflicts: int, num_actions: int, observation: AgentObservation) -> float:
        base = 96.0 if num_conflicts == 0 else max(72.0, 96.0 - num_conflicts * 4)
        if observation.features.get("avg_throughput_mbps", 0) > 100:
            base = min(99.0, base + 2)
        return round(base, 1)
