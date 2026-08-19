"""Format chatbot replies as tables, lists, or structured reports."""

from __future__ import annotations

import re
from typing import Any


class ResponseFormatter:
    FORMAT_TABLE = "table"
    FORMAT_LIST = "list"
    FORMAT_REPORT = "report"

    @staticmethod
    def detect_format(message: str) -> str:
        msg = message.lower()
        if any(k in msg for k in ("table", "tabular", "matrix", "csv", "grid")):
            return ResponseFormatter.FORMAT_TABLE
        if any(k in msg for k in ("list", "bullet", "enumerate", "points")):
            return ResponseFormatter.FORMAT_LIST
        if any(k in msg for k in ("report", "summary", "overview", "dashboard")):
            return ResponseFormatter.FORMAT_REPORT
        return ResponseFormatter.FORMAT_TABLE  # default rich format

    @staticmethod
    def md_table(headers: list[str], rows: list[list[Any]]) -> str:
        lines = [
            "| " + " | ".join(str(h) for h in headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        return "\n".join(lines)

    @staticmethod
    def bullet_list(items: list[str]) -> str:
        return "\n".join(f"• {item}" for item in items)

    @staticmethod
    def section(title: str, body: str) -> str:
        return f"**{title}**\n{body}"

    def format_kpi_comparison(self, cmp: dict, fmt: str) -> str:
        b, a = cmp["before"], cmp["after"]
        d = cmp["delta_pct"]
        bl = cmp.get("before_label", "Industry Baseline")
        al = cmp.get("after_label", "Autonomous RAN")

        rows = [
            ["Throughput (Mbps)", b.avg_throughput_mbps, a.avg_throughput_mbps, f"{d.get('avg_throughput_mbps', 0):+.1f}%"],
            ["Latency (ms)", b.avg_latency_ms, a.avg_latency_ms, f"{d.get('avg_latency_ms', 0):+.1f}%"],
            ["Power (W)", b.total_power_w, a.total_power_w, f"{d.get('total_power_w', 0):+.1f}%"],
            ["Security Detection", f"{b.security_score*100:.1f}%", f"{a.security_score*100:.1f}%",
             f"{d.get('security_score', 0):+.1f}%"],
            ["QoS SLA", f"{b.qos_sla_compliance*100:.1f}%", f"{a.qos_sla_compliance*100:.1f}%",
             f"{d.get('qos_sla_compliance', 0):+.1f}%"],
            ["QoE Score", b.qoe_score, a.qoe_score, f"{d.get('qoe_score', 0):+.1f}%"],
            ["Handover Success", f"{b.handover_success_rate*100:.1f}%", f"{a.handover_success_rate*100:.1f}%",
             f"{d.get('handover_success_rate', 0):+.1f}%"],
            ["Energy Efficiency", b.energy_efficiency, a.energy_efficiency,
             f"{d.get('energy_efficiency', 0):+.1f}%"],
            ["CSI Accuracy", f"{b.csi_accuracy*100:.1f}%", f"{a.csi_accuracy*100:.1f}%",
             f"{d.get('csi_accuracy', 0):+.1f}%"],
            ["Slice Efficiency", f"{b.slice_efficiency*100:.1f}%", f"{a.slice_efficiency*100:.1f}%",
             f"{d.get('slice_efficiency', 0):+.1f}%"],
        ]
        if fmt == self.FORMAT_LIST:
            items = [f"{bl} → {al}: KPI benchmark (industry vs autonomous intelligent RAN)"]
            for r in rows:
                items.append(f"{r[0]}: {r[1]} → {r[2]} ({r[3]})")
            return self.section("KPI Benchmark Results", self.bullet_list(items))

        table = self.md_table(["KPI", bl, al, "Change"], rows)
        intro = f"**{bl}** vs **{al}** — measured end-to-end optimization results."
        return f"{intro}\n\n{table}"

    def format_validation(self, rows: list[dict], fmt: str) -> str:
        if fmt == self.FORMAT_LIST:
            items = []
            for r in rows:
                conf = r.get("avg_confidence")
                conf_s = f", confidence {conf:.2f}" if isinstance(conf, (int, float)) else ""
                items.append(
                    f"{r['agent']}: {r['validation_status']} — {r['primary_metric']} {r['metric_value']}{conf_s}"
                )
            return self.section("Model Validation & Accuracy", self.bullet_list(items))

        table_rows = []
        for r in rows:
            conf = r.get("avg_confidence")
            conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
            table_rows.append([
                r["agent"],
                r["validation_status"],
                r["primary_metric"],
                r["metric_value"],
                r.get("training_samples", "—"),
                conf_s,
                "Pass" if r.get("inference_ok", True) else "Review",
            ])
        table = self.md_table(
            ["Agent", "Validation", "Metric", "Value", "Train Samples", "Confidence", "Inference"],
            table_rows,
        )
        validated = sum(1 for r in rows if r.get("validation_status") == "validated")
        intro = (
            f"**Model Validation Summary** — {validated}/{len(rows)} agents validated on held-out RAN datasets. "
            f"Metrics reflect prediction accuracy, detection rate, or fit quality (no internal model structure disclosed)."
        )
        return f"{intro}\n\n{table}"

    def format_benchmark(self, rows: list[dict], fmt: str) -> str:
        if fmt == self.FORMAT_LIST:
            items = [f"{r['scheduler'].replace('_', ' ').title()}: {r['throughput_mbps']} Mbps, "
                     f"{r['latency_ms']} ms, security {r['security_pct']}%" for r in rows]
            return self.section("Scheduler Benchmark", self.bullet_list(items))

        table_rows = [[
            r["scheduler"].replace("_", " ").title(),
            r["throughput_mbps"],
            r["latency_ms"],
            r["energy_w"],
            f"{r['security_pct']}%",
            r["qoe"],
            "Autonomous" if r.get("is_autonomous") else ("Industry" if r.get("is_industry") else "Baseline"),
        ] for r in rows]
        table = self.md_table(
            ["Scheduler", "Throughput (Mbps)", "Latency (ms)", "Energy (W)", "Security", "QoE", "Category"],
            table_rows,
        )
        ma = next((r for r in rows if r.get("is_autonomous")), None)
        pf = next((r for r in rows if r["scheduler"] == "proportional_fair"), None)
        gain = ""
        if ma and pf:
            gain_pct = round((ma["throughput_mbps"] / pf["throughput_mbps"] - 1) * 100, 1)
            gain = f" Autonomous RAN achieves **+{gain_pct}%** throughput vs Proportional Fair industry baseline."
        return f"**Scheduler Benchmark Comparison**{gain}\n\n{table}"

    def format_agents(self, agents: list[dict], fmt: str) -> str:
        if fmt == self.FORMAT_LIST:
            items = [f"{a['agent']}: {a['role']} [{a['ai_mode']}, {'Ready' if a['model_ready'] else 'Pending'}]"
                     for a in agents]
            return self.section("AI Agent Status", self.bullet_list(items))

        table_rows = [[a["agent"], a["role"], a["ai_mode"],
                       "Ready" if a["model_ready"] else "Pending",
                       "OK" if a.get("inference_ok") else "—"] for a in agents]
        table = self.md_table(["Agent", "RAN Function", "AI Mode", "Model", "Inference"], table_rows)
        return f"**17 AI-Driven RAN Agents** — operational status (architecture details not disclosed).\n\n{table}"

    def format_slices(self, slices: dict, fmt: str) -> str:
        if not slices:
            return "No active slice data in the digital twin."
        if fmt == self.FORMAT_LIST:
            items = [f"{n}: {d['avg_throughput_mbps']} Mbps, {d['avg_latency_ms']} ms, {d['ue_count']} UEs"
                     for n, d in slices.items()]
            return self.section("Network Slice KPIs", self.bullet_list(items))
        rows = [[n, d["avg_throughput_mbps"], d["avg_latency_ms"], d["ue_count"]] for n, d in slices.items()]
        return f"**Network Slice KPIs (Autonomous RAN)**\n\n{self.md_table(['Slice', 'Throughput (Mbps)', 'Latency (ms)', 'UEs'], rows)}"

    def format_targets(self, targets: list[dict], fmt: str) -> str:
        if fmt == self.FORMAT_LIST:
            items = [f"{t['label']}: {t['current_value']}{'%' if t['unit']=='percent' else ''} "
                     f"/ target {t['target_value']} ({'✓' if t['achieved'] else 'in progress'})"
                     for t in targets]
            return self.section("Target KPI Progress", self.bullet_list(items))
        rows = [[t["label"], f"{t['current_value']}{'%' if t['unit']=='percent' else ''}",
                 f"{t['target_value']}{'%' if t['unit']=='percent' else ''}",
                 f"{t['progress_pct']}%", "Achieved" if t["achieved"] else "In Progress"]
                for t in targets]
        return f"**3GPP / Nokia Target KPI Progress**\n\n{self.md_table(['KPI', 'Current', 'Target', 'Progress', 'Status'], rows)}"

    def format_agent_run(self, result: dict, fmt: str, compare_kpi_fn=None) -> str:
        b, a = result["kpi_before"], result["kpi_after"]
        cmp = (compare_kpi_fn or (lambda x, y: {"delta_pct": {}}))(b, a)
        agents = ", ".join(result["agents_invoked"])
        if fmt == self.FORMAT_LIST:
            items = [
                f"Agents approved: {agents}",
                f"Throughput: {b.avg_throughput_mbps} → {a.avg_throughput_mbps} Mbps",
                f"Latency: {b.avg_latency_ms} → {a.avg_latency_ms} ms",
                f"Power: {b.total_power_w} → {a.total_power_w} W",
                f"Parameter updates: {len(result['parameter_updates'])}",
            ]
            return self.section("Agent Optimization Result", self.bullet_list(items))

        rows = [
            ["Throughput (Mbps)", b.avg_throughput_mbps, a.avg_throughput_mbps,
             f"{cmp['delta_pct'].get('avg_throughput_mbps', 0):+.1f}%"],
            ["Latency (ms)", b.avg_latency_ms, a.avg_latency_ms,
             f"{cmp['delta_pct'].get('avg_latency_ms', 0):+.1f}%"],
            ["Power (W)", b.total_power_w, a.total_power_w,
             f"{cmp['delta_pct'].get('total_power_w', 0):+.1f}%"],
            ["Security", f"{b.security_score*100:.0f}%", f"{a.security_score*100:.0f}%",
             f"{cmp['delta_pct'].get('security_score', 0):+.1f}%"],
        ]
        table = self.md_table(["KPI", "Before Run", "After Run", "Change"], rows)
        return (
            f"**Super Agent approved {len(result['agents_invoked'])} agents:** {agents}\n"
            f"Global utility: {result['super_agent_decision'].get('global_utility', 'N/A')}\n\n{table}"
        )

    def format_combined_report(self, ctx_provider, fmt: str) -> str:
        cmp = ctx_provider.industry_comparison()
        parts = [
            self.format_kpi_comparison(cmp, fmt),
            "",
            self.format_validation(ctx_provider.training_validation_rows(), fmt),
            "",
            self.format_benchmark(ctx_provider.benchmark_rows(), fmt),
        ]
        return "\n".join(parts)

    @staticmethod
    def redact_architecture(text: str) -> str:
        """Strip framework/architecture terms from LLM output."""
        patterns = [
            r"\bsklearn\b", r"\bscikit-learn\b", r"\bpytorch\b", r"\btensorflow\b",
            r"\bneural network\b", r"\bhidden layer\b", r"\barchitecture\b",
            r"\btransformer\b", r"\bRandomForest\b", r"\bGradientBoosting\b",
            r"\bmodel\.py\b", r"\bsrc/agents\b", r"\bSuperAgentController\b",
        ]
        out = text
        for p in patterns:
            out = re.sub(p, "[operational AI model]", out, flags=re.IGNORECASE)
        return out
