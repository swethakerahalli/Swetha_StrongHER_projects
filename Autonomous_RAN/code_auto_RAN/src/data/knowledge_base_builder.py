"""Build knowledge graph from ontology and generated datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.utils import load_json, project_root, save_json


class KnowledgeBaseBuilder:
    def __init__(self):
        self.kb_dir = project_root() / "data" / "knowledge_base"
        self.dataset_dir = project_root() / "data" / "datasets"

    def build_knowledge_graph(self) -> Path:
        ontology = load_json(self.kb_dir / "telecom_ontology.json")
        nodes = []
        edges = []

        cfg = load_json(project_root() / "config" / "system_config.json")
        num_cells = cfg["simulation"]["num_cells"]
        num_ues = cfg["simulation"]["num_ues"]

        for c in range(num_cells):
            nodes.append({"id": f"CELL_{c:03d}", "type": "gNodeB", "load": 0.5})
        for u in range(num_ues):
            ue_id = f"UE_{u:04d}"
            cell_id = f"CELL_{u % num_cells:03d}"
            sl = list(cfg["network_slices"].keys())[u % 3]
            nodes.append({"id": ue_id, "type": "UserEquipment", "slice": sl})
            nodes.append({"id": f"SLICE_{sl}", "type": "NetworkSlice"})
            edges.append({"source": ue_id, "target": cell_id, "relation": "CONNECTED_TO"})
            edges.append({"source": ue_id, "target": f"SLICE_{sl}", "relation": "BELONGS_TO"})

        for agent in ["scheduler", "resource", "mobility", "security", "energy", "qos", "slice",
                      "qoe", "channel_estimation", "beamforming", "csi", "air_interface",
                      "digital_twin", "spectrum", "self_healing", "knowledge", "intent"]:
            aid = f"agent_{agent}"
            nodes.append({"id": aid, "type": "AIAgent", "agent_type": agent})
            edges.append({"source": aid, "target": f"CELL_000", "relation": "MANAGES"})

        nodes.append({"id": "twin_ran", "type": "DigitalTwin", "fidelity": 0.95})
        edges.append({"source": "twin_ran", "target": "CELL_000", "relation": "MONITORS"})

        threat_path = self.kb_dir / "telecom_ontology.json"
        threats = load_json(threat_path)["threat_taxonomy"]
        for t in threats:
            nodes.append({"id": t["id"], "type": "Threat", "threat_type": t["type"]})
            edges.append({"source": t["id"], "target": "CELL_000", "relation": "THREATENS"})

        if (self.dataset_dir / "ran_kpi_dataset.csv").exists():
            df = pd.read_csv(self.dataset_dir / "ran_kpi_dataset.csv", nrows=100)
            for _, row in df.iterrows():
                kpi_id = f"KPI_{row['ue_id']}_{row['timestamp']}"
                nodes.append({
                    "id": kpi_id, "type": "KPI",
                    "metric": "throughput_mbps", "value": row["throughput_mbps"],
                })
                edges.append({"source": row["cell_id"], "target": kpi_id, "relation": "CONSUMES"})

        kg = {
            "metadata": {"nodes": len(nodes), "edges": len(edges)},
            "ontology_ref": "telecom_ontology.json",
            "nodes": nodes,
            "edges": edges,
        }
        out = self.kb_dir / "knowledge_graph.json"
        save_json(kg, out)
        return out

    def build_external_sources_index(self) -> Path:
        """Index all external knowledge source files."""
        kb = self.kb_dir
        index = {
            "sources": {
                "3gpp": "3gpp_references.json",
                "oran": "oran_references.json",
                "nokia_cfam": "nokia_cfam_references.json",
                "sharepoint": "sharepoint_references.json",
                "confluence": "confluence_references.json",
                "ontology": "telecom_ontology.json",
            },
            "loaded": [f.name for f in kb.glob("*.json")],
        }
        out = kb / "external_sources_index.json"
        save_json(index, out)
        return out

    def build_feature_store_manifest(self) -> Path:
        manifest = {
            "feature_groups": {
                "scheduling": ["cqi", "sinr_db", "buffer_occupancy", "latency_ms", "mcs", "prb_allocated"],
                "mobility": ["velocity_mps", "direction_deg", "rsrp_dbm", "handover_pending"],
                "security": ["packet_rate_pps", "auth_failures", "spectrum_anomaly_score", "flow_entropy"],
                "energy": ["power_consumption_w", "cell_utilization", "sleep_state", "renewable_pct"],
                "slice": ["prb_utilization", "active_ues", "sla_compliance", "throughput_mbps", "latency_p99_ms"],
                "qoe": ["throughput_mbps", "latency_ms", "packet_loss", "slice"],
            },
            "datasets": {
                "ran_kpi": "ran_kpi_dataset.csv",
                "mobility": "mobility_traces.csv",
                "security": "security_events.csv",
                "energy": "energy_metrics.csv",
                "slice": "slice_utilization.csv",
                "handover": "handover_events.csv",
            },
        }
        out = self.kb_dir / "feature_store_manifest.json"
        save_json(manifest, out)
        return out


def build_knowledge_base() -> dict[str, Path]:
    builder = KnowledgeBaseBuilder()
    return {
        "knowledge_graph": builder.build_knowledge_graph(),
        "feature_store": builder.build_feature_store_manifest(),
        "external_index": builder.build_external_sources_index(),
    }
