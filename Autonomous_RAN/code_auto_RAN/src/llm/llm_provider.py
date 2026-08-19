"""Unified LLM provider: Ollama (local) with Nokia cached knowledge and rule-based fallback."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests

from src.common.utils import load_config, load_json, project_root


class LLMProvider:
    def __init__(self):
        self.cfg = load_config("llm_config.json")
        self.ollama_cfg = self.cfg["llm_providers"]["ollama"]
        self.cache_path = project_root() / self.cfg["llm_providers"]["nokia_system_insights"]["cache_path"]
        self._ollama_available: bool | None = None

    def is_ollama_available(self) -> bool:
        if self._ollama_available is not None:
            return self._ollama_available
        if not self.ollama_cfg.get("enabled", True):
            self._ollama_available = False
            return False
        try:
            r = requests.get(
                f"{self.ollama_cfg['base_url']}/api/tags",
                timeout=3,
            )
            self._ollama_available = r.status_code == 200
        except (requests.RequestException, OSError):
            self._ollama_available = False
        return self._ollama_available

    def generate(self, prompt: str, system: str = "", max_tokens: int = 512) -> dict[str, Any]:
        for provider in self.cfg["llm_providers"]["priority"]:
            if provider == "ollama" and self.is_ollama_available():
                result = self._ollama_generate(prompt, system, max_tokens)
                if result:
                    return {"text": result, "provider": "ollama", "model": self._resolve_ollama_model()}
            elif provider == "nokia_cached":
                result = self._nokia_cached_generate(prompt)
                if result:
                    return {"text": result, "provider": "nokia_cached", "model": "system_insights_cache"}
            elif provider == "rule_based":
                return {"text": self._rule_based_generate(prompt), "provider": "rule_based", "model": "template"}
        return {"text": "Unable to generate response.", "provider": "none", "model": None}

    def _resolve_ollama_model(self) -> str:
        try:
            tags = requests.get(f"{self.ollama_cfg['base_url']}/api/tags", timeout=3).json()
            available = [m["name"].split(":")[0] for m in tags.get("models", [])]
            preferred = [self.ollama_cfg["model"]] + self.ollama_cfg.get("fallback_models", [])
            for p in preferred:
                if p in available or any(p in a for a in available):
                    return p
            return available[0] if available else self.ollama_cfg["model"]
        except (requests.RequestException, OSError, KeyError, IndexError):
            return self.ollama_cfg["model"]

    def _ollama_generate(self, prompt: str, system: str, max_tokens: int) -> str | None:
        model = self._resolve_ollama_model()
        payload = {
            "model": model,
            "prompt": prompt,
            "system": system or "You are a telecom AI expert for 6G Autonomous RAN.",
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        try:
            r = requests.post(
                f"{self.ollama_cfg['base_url']}/api/generate",
                json=payload,
                timeout=self.ollama_cfg.get("timeout_seconds", 60),
            )
            if r.status_code == 200:
                return r.json().get("response", "").strip()
        except (requests.RequestException, OSError, json.JSONDecodeError):
            pass
        return None

    def _nokia_cached_generate(self, prompt: str) -> str | None:
        kb_dir = project_root() / "data" / "knowledge_base"
        parts = []
        for fname in ["nokia_cfam_references.json", "3gpp_references.json", "sharepoint_references.json"]:
            path = kb_dir / fname
            if path.exists():
                data = load_json(path)
                parts.append(json.dumps(data, indent=0)[:2000])
        if not parts:
            return None
        context = "\n".join(parts)
        return (
            f"Based on Nokia CFAM/3GPP knowledge base:\n{context[:3000]}\n\n"
            f"Query: {prompt}\n\n"
            f"Relevant features: Mobility Robustness Optimization (OSS_FC_017307), "
            f"Energy Efficiency (SR003080), Autonomous Recovery (SR001534). "
            f"Apply multi-agent coordination per O-RAN closed-loop architecture."
        )

    def _rule_based_generate(self, prompt: str) -> str:
        prompt_l = prompt.lower()
        intents = self.cfg.get("intent_templates", {})
        matched = []
        for intent, spec in intents.items():
            if any(kw in prompt_l for kw in spec["keywords"]):
                matched.append(f"{intent} -> activate {spec['primary_agent']}_agent")
        if matched:
            return f"Intent parsed: {'; '.join(matched)}. Policies will be distributed via O-RAN A1 interface."
        return f"Processed telecom query using ontology rules. Context keywords: {self._extract_keywords(prompt_l)}"

    @staticmethod
    def _extract_keywords(text: str) -> str:
        keywords = ["scheduling", "handover", "energy", "security", "throughput", "latency", "slice", "qoe"]
        found = [k for k in keywords if k in text]
        return ", ".join(found) if found else "general_ran_optimization"

    def parse_intent(self, operator_intent: str) -> dict[str, Any]:
        """Convert natural-language operator intent to agent policies."""
        response = self.generate(
            f"Parse this network operator intent into agent policies: '{operator_intent}'. "
            f"Return JSON with primary_agent, policies, kpi_targets.",
            system="Return concise structured intent for Autonomous RAN.",
        )
        text = response["text"]
        policies = []
        intents_cfg = self.cfg.get("intent_templates", {})
        intent_l = operator_intent.lower()
        primary = "scheduler"
        for name, spec in intents_cfg.items():
            if any(kw in intent_l for kw in spec["keywords"]):
                primary = spec["primary_agent"]
                policies.append({"intent": name, "agent": spec["primary_agent"]})
        return {
            "operator_intent": operator_intent,
            "primary_agent": primary,
            "policies": policies or [{"intent": "default", "agent": "scheduler"}],
            "llm_reasoning": text,
            "llm_provider": response["provider"],
        }
