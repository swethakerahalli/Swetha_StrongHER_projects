"""Super Agent monitoring — detect degraded agents and trigger Agent Optimizer."""

from __future__ import annotations

from src.api.agent_performance_service import AGENT_KPI_PROFILE, AgentPerformanceService
from src.common.utils import load_json, project_root


class AgentMonitoringService:
    DEGRADATION_THRESHOLDS = {
        "performance_index_min": 65.0,
        "improvement_pct_min": 0.0,
        "confidence_min": 0.72,
        "validation_score_min": 0.55,
    }

    def __init__(self, perf_svc: AgentPerformanceService | None = None):
        self.perf_svc = perf_svc or AgentPerformanceService()
        self.monitoring_log: list[dict] = []
        self._constraints = self._load_constraints()

    @staticmethod
    def _load_constraints() -> dict:
        path = project_root() / "config" / "ran_constraints.json"
        return load_json(path) if path.exists() else {"constraints": []}

    def _degradation_score(self, agent: dict) -> float:
        score = 0.0
        t = self.DEGRADATION_THRESHOLDS
        pi = agent.get("performance_index", 100)
        imp = agent.get("improvement_pct", 0)
        conf = agent.get("confidence") or 0.8
        val_raw = agent.get("validation_raw") or 0.8

        if pi < t["performance_index_min"]:
            score += (t["performance_index_min"] - pi) / t["performance_index_min"] * 40
        if imp < t["improvement_pct_min"]:
            score += min(30, abs(imp) * 2)
        if conf < t["confidence_min"]:
            score += (t["confidence_min"] - conf) * 50
        if isinstance(val_raw, (int, float)) and val_raw < t["validation_score_min"]:
            score += (t["validation_score_min"] - val_raw) * 30
        return round(min(100, score), 1)

    def _status(self, agent: dict) -> str:
        deg = self._degradation_score(agent)
        if deg >= 35:
            return "degraded"
        if deg >= 15:
            return "warning"
        return "healthy"

    def monitor_all(self) -> dict:
        comparison = self.perf_svc.get_comparison()
        agents = comparison["agents"]
        monitored = []
        degraded = []
        warnings = []

        for a in agents:
            if a["agent"] in ("knowledge", "intent", "agent_optimizer"):
                continue
            deg = self._degradation_score(a)
            status = self._status(a)
            entry = {
                "agent": a["agent"],
                "agent_label": a["agent_label"],
                "role": a["role"],
                "primary_kpi": a["primary_kpi"],
                "performance_index": a["performance_index"],
                "improvement_pct": a["improvement_pct"],
                "confidence": a["confidence"],
                "validation_status": a["validation_status"],
                "validation_score": a["validation_score"],
                "degradation_score": deg,
                "status": status,
                "vs_baseline": a["vs_baseline"],
            }
            monitored.append(entry)
            if status == "degraded":
                degraded.append(entry)
            elif status == "warning":
                warnings.append(entry)

        snapshot = {
            "monitored_count": len(monitored),
            "healthy_count": sum(1 for m in monitored if m["status"] == "healthy"),
            "warning_count": len(warnings),
            "degraded_count": len(degraded),
            "agents": monitored,
            "degraded_agents": degraded,
            "warning_agents": warnings,
            "optimization_required": len(degraded) > 0 or len(warnings) > 0,
        }
        self.monitoring_log.append(snapshot)
        return snapshot

    def build_optimizer_observations(self, monitoring: dict | None = None) -> list[dict]:
        snap = monitoring or self.monitor_all()
        targets = snap["degraded_agents"] + snap["warning_agents"]
        obs_list = []
        for a in targets:
            val_raw = a.get("validation_score", "0")
            try:
                val_num = float(str(val_raw).replace("%", "")) / (100 if "%" in str(val_raw) else 1)
            except ValueError:
                val_num = 0.7
            obs_list.append({
                "features": {
                    "performance_index": a["performance_index"],
                    "improvement_pct": a["improvement_pct"],
                    "confidence": a["confidence"] or 0.8,
                    "validation_score": val_num,
                    "degradation_score": a["degradation_score"],
                },
                "context": {"target_agent": a["agent"]},
            })
        return obs_list

    def get_constraints_status(self, kpi: dict | None = None) -> dict:
        """Evaluate RAN operational constraints against current KPIs."""
        constraints = self._constraints.get("constraints", [])
        if not kpi:
            kpi = self.perf_svc.baseline_svc.autonomous_kpi().model_dump()

        results = []
        satisfied = 0
        for c in constraints:
            key = c["kpi_key"]
            val = kpi.get(key, 0)
            limit = c["limit"]
            op = c.get("operator", "<=")
            ok = self._check(val, limit, op)
            if ok:
                satisfied += 1
            margin = self._margin(val, limit, op)
            results.append({
                "id": c["id"],
                "name": c["name"],
                "category": c.get("category", "general"),
                "kpi_key": key,
                "current_value": round(float(val), 3) if isinstance(val, (int, float)) else val,
                "limit": limit,
                "operator": op,
                "unit": c.get("unit", ""),
                "satisfied": ok,
                "margin_pct": round(margin, 1),
                "severity": c.get("severity", "medium"),
                "description": c.get("description", ""),
            })

        return {
            "total_constraints": len(results),
            "satisfied_count": satisfied,
            "violated_count": len(results) - satisfied,
            "compliance_pct": round(satisfied / len(results) * 100, 1) if results else 100,
            "constraints": results,
            "by_category": self._group_by_category(results),
        }

    @staticmethod
    def _check(val: float, limit: float, op: str) -> bool:
        if op == "<=":
            return val <= limit
        if op == ">=":
            return val >= limit
        if op == "<":
            return val < limit
        if op == ">":
            return val > limit
        return True

    @staticmethod
    def _margin(val: float, limit: float, op: str) -> float:
        if limit == 0:
            return 100.0
        if op in ("<=", "<"):
            return max(0, (1 - val / limit) * 100) if limit else 0
        return max(0, (val / limit - 1) * 100) if limit else 0

    @staticmethod
    def _group_by_category(results: list[dict]) -> dict:
        groups: dict[str, list] = {}
        for r in results:
            groups.setdefault(r["category"], []).append(r)
        return {k: {"count": len(v), "satisfied": sum(1 for x in v if x["satisfied"]),
                    "items": v} for k, v in groups.items()}
