"""Operations chatbot for 6G channel intelligence."""

from __future__ import annotations

import uuid
from pathlib import Path

from src.common.utils import load_json, project_root


class ChannelChatbot:
    def __init__(self, store):
        self.store = store
        self.sessions: dict[str, list] = {}
        self.kb = self._kb()

    def _kb(self) -> str:
        parts = []
        for name in ("3gpp_references.json", "nokia_insights_cache.json", "sharepoint_references.json"):
            path = project_root() / "data" / "knowledge_base" / name
            if path.exists():
                parts.append(f"{name}: {path.read_text(encoding='utf-8')[:1500]}")
        return "\n".join(parts)

    def handle(self, message: str, session_id: str = "") -> dict:
        session_id = session_id or str(uuid.uuid4())[:8]
        msg = message.lower()
        report = self.store.report
        arch = report.get("architecture", {})
        agents = report.get("agents", {})

        if any(k in msg for k in ("run agent", "optimize", "mitigate", "closed loop")):
            result = self.store.run_agents()
            coord = result.get("coordination", {})
            super_d = result.get("super", {})
            reply = (
                "Ran the multi-agent loop (domain → orchestrator → coordinator → super agent → twin).\n"
                f"- Harmonized policy: {result['policy']}\n"
                f"- Conflicts resolved: {coord.get('n_conflicts', len(coord.get('conflicts') or []))}\n"
                f"- Strategy: {coord.get('strategy')}\n"
                f"- Super approved/rejected: {super_d.get('approved_count')}/{super_d.get('rejected_count')}\n"
                f"- Twin fidelity: {result['twin']['mean_fidelity']}\n"
                f"- Mean NMSE: {result['twin']['mean_nmse']}\n"
                f"- Attack cells: {result['twin']['attack_cells']}"
            )
        elif any(k in msg for k in ("coordinator", "conflict", "harmoniz")):
            result = self.store.last_run or self.store.run_agents()
            coord = result.get("coordination", {})
            reply = (
                "Coordinator agent resolves conflicts among CSI agents.\n"
                "Priority: security/mitigation/self-healing > twin fidelity > NMSE > mobility > beam > spectrum > SE.\n"
                f"- Last strategy: {coord.get('strategy')}\n"
                f"- Harmonized policy: {coord.get('harmonized_policy') or result.get('policy')}\n"
                f"- Conflicts: {coord.get('conflicts')}\n"
                f"- Resolutions: {coord.get('resolutions')}\n"
                "Rules: keep DMRS if NMSE>0.1; security freezes beam; jamming hops carrier; "
                "twin vetoes unsafe deploy; isolate before PRB boost; MMSE fallback on poisoning."
            )
        elif any(k in msg for k in ("super agent", "super-agent", "control plane", "enable agent", "disable")):
            agent = self.store.shim.agents.get("super")
            status = agent.get_status() if agent else {}
            last = (self.store.last_run or {}).get("super", {})
            reply = (
                "Super agent is the control plane: approve/reject, weights, enable/disable, twin gate.\n"
                f"- Controlled agents: {status.get('n_controlled_agents')}\n"
                f"- Last approved: {last.get('approved_count')}  rejected: {last.get('rejected_count')}\n"
                f"- Global utility: {last.get('global_utility')}\n"
                f"- Autonomy: {last.get('autonomy_level')}\n"
                "Gates: reject CSI-prediction pilot cuts if NMSE>0.15; gate optimization/resource if twin is unsafe."
            )
        elif "nmse" in msg or "channel" in msg or "csi" in msg:
            reply = (
                "Channel estimation (held-out test):\n"
                f"- LS NMSE: {arch.get('test_nmse_ls')}\n"
                f"- MMSE NMSE: {arch.get('test_nmse_mmse')}\n"
                f"- AI ensemble NMSE: {arch.get('test_nmse_ai')}\n"
                f"- Improvement vs MMSE: {arch.get('nmse_improvement_pct')}%\n"
                f"- CSI prediction accuracy: {arch.get('csi_prediction_accuracy')}%\n"
                "Aligned with 3GPP TR 38.901 CDL/TDL and TS 38.211 DMRS/CSI-RS."
            )
        elif "attack" in msg or "security" in msg:
            sec = agents.get("security", {}).get("metrics", {})
            reply = (
                "Security agent (train/validation/test):\n"
                f"- Binary test accuracy: {sec.get('binary_test_accuracy')}\n"
                f"- Multiclass test F1: {sec.get('multiclass_test_f1')}\n"
                f"- ROC-AUC (test): {sec.get('binary_test_roc_auc')}\n"
                "Classes: normal, pilot contamination, jamming, spoofing, false CSI, poisoning, adversarial, backdoor."
            )
        elif "train" in msg or "valid" in msg or "test" in msg:
            lines = ["Per-agent train / validation / test:"]
            for name, payload in agents.items():
                m = payload.get("metrics", {})
                keys = [k for k in m if any(s in k for s in ("train", "validation", "test"))][:6]
                lines.append(f"- {name}: " + ", ".join(f"{k}={m[k]}" for k in keys))
            reply = "\n".join(lines)
        elif "twin" in msg or "digital" in msg:
            st = self.store.twin.observe()
            reply = (
                "Digital twin state:\n"
                + "\n".join(f"- {k}: {v}" for k, v in st.items())
                + f"\nFidelity score: {self.store.twin.fidelity_score()}"
            )
        elif "3gpp" in msg or "standard" in msg or "nokia" in msg:
            reply = (
                "Standards & Nokia sources used:\n"
                "- 3GPP TR 38.901, TS 38.211, TS 38.214, TS 38.101-4, TR 38.843, TR 38.811\n"
                "- Nokia System Insights CFAM RP003187 DMRS channel estimation\n"
                "- SharePoint RAN1 R1-2506757 6G AI/ML air-interface use cases\n"
                "- O-RAN near-RT RIC xApp mapping for CSI/beam/security"
            )
        elif "agent" in msg:
            reply = (
                "Channel-estimation stack: domain PHY/security/mobility agents, then orchestrator, "
                "coordinator (conflict resolution), and super agent (control).\n"
                "Agents: " + ", ".join(agents.keys() or self.store.shim.list_agents()) +
                "\nAsk about NMSE, conflicts, super agent, attacks, training, or the digital twin."
            )
        else:
            reply = (
                "I am the 6G Channel Intelligence assistant. Ask about NMSE, BER, attacks, "
                "coordinator conflicts, super-agent control, train/validation/test scores, "
                "digital twin, 3GPP alignment, or say 'run agents'."
            )
        return {"session_id": session_id, "reply": reply, "architecture": arch}
