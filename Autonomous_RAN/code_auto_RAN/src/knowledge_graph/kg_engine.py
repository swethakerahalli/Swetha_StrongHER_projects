"""Knowledge Graph engine for telecom ontology reasoning."""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from src.common.utils import load_json, project_root


class KnowledgeGraphEngine:
    def __init__(self, graph_path: Path | None = None):
        path = graph_path or project_root() / "data" / "knowledge_base" / "knowledge_graph.json"
        self.graph = nx.DiGraph()
        if path.exists():
            self._load(path)

    def _load(self, path: Path) -> None:
        data = load_json(path)
        for node in data.get("nodes", []):
            nid = node.pop("id")
            self.graph.add_node(nid, **node)
        for edge in data.get("edges", []):
            self.graph.add_edge(edge["source"], edge["target"], relation=edge["relation"])

    def query_neighbors(self, node_id: str, relation: str | None = None) -> list[str]:
        if node_id not in self.graph:
            return []
        neighbors = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if relation is None or data.get("relation") == relation:
                neighbors.append(target)
        return neighbors

    def get_cell_ues(self, cell_id: str) -> list[str]:
        ues = []
        for src, tgt, data in self.graph.edges(data=True):
            if data.get("relation") == "CONNECTED_TO" and tgt == cell_id:
                ues.append(src)
        return ues

    def get_threats_for_cell(self, cell_id: str) -> list[dict]:
        threats = []
        for src, tgt, data in self.graph.edges(data=True):
            if data.get("relation") == "THREATENS" and tgt == cell_id:
                node = self.graph.nodes.get(src, {})
                if node.get("type") == "Threat":
                    threats.append({"id": src, **node})
        return threats

    def root_cause_analysis(self, kpi_node: str, depth: int = 2) -> list[dict]:
        causes = []
        if kpi_node not in self.graph:
            return causes
        for pred in self.graph.predecessors(kpi_node):
            node = dict(self.graph.nodes[pred])
            node["id"] = pred
            causes.append(node)
            if depth > 1:
                for p2 in self.graph.predecessors(pred):
                    n2 = dict(self.graph.nodes[p2])
                    n2["id"] = p2
                    causes.append(n2)
        return causes

    def agent_context(self, cell_id: str) -> dict:
        return {
            "connected_ues": self.get_cell_ues(cell_id),
            "threats": self.get_threats_for_cell(cell_id),
            "managing_agents": [
                n for n in self.query_neighbors(f"agent_scheduler")
            ],
        }

    def stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": dict(nx.get_node_attributes(self.graph, "type")),
        }
