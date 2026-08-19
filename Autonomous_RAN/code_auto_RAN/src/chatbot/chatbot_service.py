"""RAN Chatbot — format-aware responses with KPI, validation, and benchmark data."""

from __future__ import annotations

import re
import uuid

from src.api.agent_performance_service import AgentPerformanceService
from src.api.ran_parameter_service import RANParameterService
from src.api.schemas import ChatResponse, KPIStats
from src.chatbot.ran_context import RANContextProvider
from src.chatbot.response_formatter import ResponseFormatter
from src.llm.llm_provider import LLMProvider


class ChatbotService:
    SYSTEM_PROMPT = (
        "You are the Autonomous Intelligent RAN operations assistant for Nokia telecom networks. "
        "Answer using ONLY the operational data provided. "
        "Format your answer as the user requested (markdown table, bullet list, or structured report). "
        "Include KPI values, benchmark comparisons, and model validation metrics when relevant. "
        "Do NOT describe internal software architecture, ML frameworks, code structure, "
        "neural network layers, or implementation details. "
        "Refer to components as 'AI-driven agents' and 'validated prediction models' only."
    )

    def __init__(self, param_svc: RANParameterService):
        self.param_svc = param_svc
        self.ctx = RANContextProvider(param_svc)
        self.fmt = ResponseFormatter()
        self.llm = LLMProvider()
        self.sessions: dict[str, list[dict]] = {}

    def handle_message(self, message: str, session_id: str = "") -> ChatResponse:
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        self.sessions.setdefault(session_id, []).append({"role": "user", "content": message})

        actions_taken = []
        reply = ""
        kpi_snapshot = None
        msg_l = message.lower()
        fmt = self.fmt.detect_format(message)

        if self._is_run_agents(msg_l):
            intent = self._extract_intent(message)
            result = self.param_svc.run_agents_and_apply(intent=intent)
            kpi_snapshot = result["kpi_after"]
            actions_taken.append({"type": "run_agents", "action_id": result["action_id"]})
            reply = self.fmt.format_agent_run(result, fmt, self.param_svc.store.compare_kpi)

        elif self._is_optimize(msg_l):
            result = self.param_svc.run_agents_and_apply(intent=message)
            kpi_snapshot = result["kpi_after"]
            actions_taken.append({"type": "optimize", "action_id": result["action_id"]})
            reply = self.fmt.format_agent_run(result, fmt, self.param_svc.store.compare_kpi)

        elif self._is_reset_baseline(msg_l):
            kpi = self.param_svc.store.reset_baseline()
            kpi_snapshot = kpi
            cmp = self.param_svc.store.get_industry_comparison()
            reply = (
                f"Industry baseline reset to conventional RAN reference values.\n\n"
                + self.fmt.format_kpi_comparison(cmp, fmt)
            )

        elif self._is_validation_query(msg_l):
            reply = self.fmt.format_validation(self.ctx.training_validation_rows(), fmt)

        elif self._is_benchmark_query(msg_l):
            reply = self.fmt.format_benchmark(self.ctx.benchmark_rows(), fmt)

        elif self._is_agent_perf_query(msg_l):
            reply = self._format_agent_performance(fmt)

        elif self._is_agent_query(msg_l):
            reply = self.fmt.format_agents(self.ctx.agent_status(), fmt)

        elif self._is_slice_query(msg_l):
            reply = self.fmt.format_slices(self.ctx.slice_kpis(), fmt)

        elif self._is_twin_query(msg_l):
            reply = self._format_twin_visualization(fmt)

        elif self._is_target_query(msg_l):
            reply = self.fmt.format_targets(self.ctx.target_kpis(), fmt)

        elif self._is_report_query(msg_l):
            reply = self.fmt.format_combined_report(self.ctx, fmt)

        elif self._is_kpi_query(msg_l):
            cmp = self.param_svc.store.get_industry_comparison()
            kpi_snapshot = cmp["after"]
            reply = self.fmt.format_kpi_comparison(cmp, fmt)

        else:
            llm_result = self._llm_reply(message, fmt)
            reply = llm_result["text"]
            llm_provider = llm_result.get("_provider", "llm")
            self.sessions[session_id].append({"role": "assistant", "content": reply})
            return ChatResponse(reply=reply, session_id=session_id, llm_provider=llm_provider)

        self.sessions[session_id].append({"role": "assistant", "content": reply})
        return ChatResponse(
            reply=reply, session_id=session_id,
            actions_taken=actions_taken, kpi_snapshot=kpi_snapshot,
            llm_provider="ran_data",
        )

  # ── Intent detectors ──────────────────────────────────────────────

    @staticmethod
    def _is_run_agents(msg: str) -> bool:
        return any(k in msg for k in ["run agent", "invoke agent", "execute agent", "apply agent"])

    @staticmethod
    def _is_reset_baseline(msg: str) -> bool:
        return "reset baseline" in msg or "set baseline" in msg

    @staticmethod
    def _is_optimize(msg: str) -> bool:
        return any(k in msg for k in ["optimize", "improve", "reduce energy", "reduce latency", "boost"])

    @staticmethod
    def _is_validation_query(msg: str) -> bool:
        return any(k in msg for k in [
            "accuracy", "validation", "validate", "trained", "training result",
            "model metric", "inference", "test result", "r2", "r²",
        ])

    @staticmethod
    def _is_benchmark_query(msg: str) -> bool:
        return any(k in msg for k in [
            "benchmark", "scheduler", "proportional fair", "round robin", "son static",
        ])

    @staticmethod
    def _is_agent_perf_query(msg: str) -> bool:
        return any(k in msg for k in ["agent performance", "agents vs baseline", "per agent", "agent comparison"])

    @staticmethod
    def _is_agent_query(msg: str) -> bool:
        return any(k in msg for k in ["agent status", "all agents", "list agents", "which agents"])

    @staticmethod
    def _is_slice_query(msg: str) -> bool:
        return any(k in msg for k in ["slice kpi", "network slice", "embb", "urllc", "mmtc", "slice status"])

    @staticmethod
    def _is_twin_query(msg: str) -> bool:
        return any(k in msg for k in [
            "digital twin", "twin visualization", "twin map", "network map",
            "topology", "cell load", "live network",
        ])

    @staticmethod
    def _is_target_query(msg: str) -> bool:
        return any(k in msg for k in ["target kpi", "goal", "progress", "3gpp target", "nokia target"])

    @staticmethod
    def _is_report_query(msg: str) -> bool:
        return any(k in msg for k in [
            "full report", "e2e", "end to end", "implementation result",
            "everything", "complete summary", "overall result",
        ])

    @staticmethod
    def _is_kpi_query(msg: str) -> bool:
        return any(k in msg for k in [
            "kpi", "throughput", "latency", "comparison", "stats", "status", "power", "security",
            "qoe", "energy", "show result", "before and after", "before vs after", "industry baseline",
        ])

    @staticmethod
    def _extract_intent(message: str) -> str:
        return message

    def _format_agent_performance(self, fmt: str) -> str:
        data = AgentPerformanceService(self.param_svc.store.baseline_svc).get_comparison()
        agents = data["agents"]
        if fmt == ResponseFormatter.FORMAT_LIST:
            items = [f"{a['agent_label']}: {a['primary_kpi']} {a['baseline_value']} → {a['autonomous_value']} ({a['improvement_pct']:+.1f}%)" for a in agents]
            return self.fmt.section("Agent Performance vs Industry Baseline", self.fmt.bullet_list(items))
        rows = [[a["agent_label"], a["primary_kpi"], a["baseline_value"], a["autonomous_value"],
                 f"{a['improvement_pct']:+.1f}%", a["validation_status"], a["validation_score"], a["performance_index"]]
                for a in agents]
        table = self.fmt.md_table(
            ["Agent", "KPI", "Baseline", "Autonomous", "Improvement", "Validation", "Metric", "Perf. Index"], rows)
        s = data["summary"]
        return (
            f"**Agent Performance vs Industry Baseline** — {s['improved_count']}/{s['total_agents']} agents improved, "
            f"avg improvement {s['avg_improvement_pct']}%.\n\n{table}"
        )

    def _format_twin_visualization(self, fmt: str) -> str:
        viz = self.param_svc.store.twin.visualization_data()
        if fmt == ResponseFormatter.FORMAT_LIST:
            items = [
                f"Fidelity: {viz['fidelity']*100:.1f}% | {viz['num_cells']} cells, {viz['num_ues']} UEs",
                f"Avg throughput: {viz['avg_throughput']:.1f} Mbps | latency: {viz['avg_latency']:.2f} ms",
                f"Power: {viz['total_power_w']:.0f} W | CSI: {viz['csi_accuracy']*100:.0f}%",
            ]
            for c in viz["cells"]:
                items.append(f"{c['cell_id']}: load {(c['load']*100):.0f}%, {c['ue_count']} UEs, {c['power_w']}W")
            return self.fmt.section("Digital Twin Live State", self.fmt.bullet_list(items))

        rows = [[
            c["cell_id"], f"{c['load']*100:.0f}%", c["ue_count"], c["power_w"],
            "Sleep" if c["sleep"] else "Active",
        ] for c in viz["cells"]]
        table = self.fmt.md_table(["Cell", "Load", "UEs", "Power (W)", "State"], rows)
        header = (
            f"**Digital Twin Visualization** — fidelity {viz['fidelity']*100:.1f}%, "
            f"{viz['num_ues']} UEs across {viz['num_cells']} gNodeB sites. "
            f"View live map on dashboard."
        )
        slice_rows = []
        slices: dict[str, list] = {}
        for u in viz["ues"]:
            slices.setdefault(u["slice"], []).append(u)
        for sl, ues in slices.items():
            tp = sum(x["throughput"] for x in ues) / len(ues)
            slice_rows.append([sl, len(ues), f"{tp:.1f}"])
        slice_table = self.fmt.md_table(["Slice", "UE Count", "Avg TP (Mbps)"], slice_rows)
        return f"{header}\n\n{table}\n\n**Slice Distribution**\n{slice_table}"

    def _llm_reply(self, message: str, fmt: str) -> dict:
        cmp = self.param_svc.store.get_industry_comparison()
        b, a = cmp["before"], cmp["after"]
        val_rows = self.ctx.training_validation_rows()[:5]
        val_summary = "; ".join(
            f"{r['agent']}: {r['primary_metric']}={r['metric_value']}" for r in val_rows
        )
        format_hint = {
            ResponseFormatter.FORMAT_TABLE: "Use a markdown table with | columns |.",
            ResponseFormatter.FORMAT_LIST: "Use bullet points (•).",
            ResponseFormatter.FORMAT_REPORT: "Use sections with **bold headers** and tables.",
        }[fmt]

        context = (
            f"User query: {message}\n"
            f"Format: {format_hint}\n"
            f"Industry baseline throughput: {b.avg_throughput_mbps} Mbps, latency: {b.avg_latency_ms} ms.\n"
            f"Autonomous RAN throughput: {a.avg_throughput_mbps} Mbps, latency: {a.avg_latency_ms} ms.\n"
            f"Throughput improvement: {cmp['delta_pct'].get('avg_throughput_mbps', 0):+.1f}%.\n"
            f"Latency reduction: {cmp['delta_pct'].get('avg_latency_ms', 0):+.1f}%.\n"
            f"Power change: {cmp['delta_pct'].get('total_power_w', 0):+.1f}%.\n"
            f"Security: {b.security_score*100:.1f}% → {a.security_score*100:.1f}%.\n"
            f"Agents active: 17 AI-driven agents, all validated.\n"
            f"Sample validation metrics: {val_summary}.\n"
            f"Super-agent validations: {self.ctx.super_agent_summary()['validations']}."
        )
        llm_out = self.llm.generate(context, system=self.SYSTEM_PROMPT, max_tokens=800)
        text = self.fmt.redact_architecture(llm_out["text"])
        if fmt == ResponseFormatter.FORMAT_TABLE and "|" not in text and self._is_kpi_query(message.lower()):
            text = self.fmt.format_kpi_comparison(cmp, fmt) + "\n\n" + text
        return {"text": text, "_provider": llm_out["provider"]}
