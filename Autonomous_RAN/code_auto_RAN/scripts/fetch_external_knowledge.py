#!/usr/bin/env python3
"""Index and document Nokia MCP knowledge sources for Autonomous Intelligent RAN."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.utils import load_json, project_root, save_json


MCP_SERVERS = {
    "system-insights": {
        "tools": ["ask", "search_specs", "read_spec_requirement", "search_tools", "run_tool"],
        "use_for": ["3GPP specs", "CFAM features", "NIDD KPIs", "O-RAN references", "troubleshooting"],
        "output_files": ["nokia_insights_cache.json", "nokia_cfam_references.json", "3gpp_references.json"],
    },
    "sharepoint": {
        "tools": ["searchSharePoint", "searchInSite", "getFileOrFolderMetadataByUrl"],
        "use_for": ["project documents", "RAN architecture decks", "internal specs"],
        "output_files": ["sharepoint_references.json"],
    },
    "confluence": {
        "tools": ["confluence_search_content", "confluence_get_page"],
        "use_for": ["team runbooks", "architecture notes", "MRO/SON documentation"],
        "output_files": ["confluence_references.json"],
    },
    "pronto-prod": {
        "tools": ["pronto search (requires auth)"],
        "use_for": ["defect patterns", "field issue RCA", "KPI regression analysis"],
        "output_files": ["pronto_references.json"],
    },
}


def main():
    kb = project_root() / "data" / "knowledge_base"
    required = [
        "3gpp_references.json", "oran_references.json", "nokia_cfam_references.json",
        "nokia_insights_cache.json", "sharepoint_references.json",
        "confluence_references.json", "telecom_ontology.json", "knowledge_graph.json",
        "feature_store_manifest.json", "external_sources_index.json",
    ]
    status = {f: (kb / f).exists() for f in required}
    cfam = load_json(kb / "nokia_cfam_references.json") if (kb / "nokia_cfam_references.json").exists() else {}
    insights = load_json(kb / "nokia_insights_cache.json") if (kb / "nokia_insights_cache.json").exists() else {}
    report = {
        "sources": status,
        "all_present": all(status.values()),
        "mcp_servers": MCP_SERVERS,
        "cfam_features": len(cfam.get("cfam_features", [])),
        "insights_kpi_groups": len(insights.get("autonomous_ran_kpis", {})),
        "agent_count": 17,
        "refresh_instructions": {
            "system_insights": "MCP: ask('Autonomous RAN KPIs 3GPP O-RAN') + search_specs(query='OSS_FC', source='cfam')",
            "sharepoint": "MCP: searchSharePoint for Autonomous RAN project docs",
            "confluence": "MCP: confluence_search_content query='Autonomous RAN'",
            "pronto": "MCP: authenticate pronto-prod then search defects",
        },
    }
    save_json(report, kb / "fetch_report.json")
    save_json({"mcp_servers": MCP_SERVERS, "last_updated": "2026-07-14"}, kb / "external_sources_index.json")
    print("Nokia MCP Knowledge Base Status:")
    for k, v in status.items():
        print(f"  {'OK' if v else 'MISSING'}: {k}")
    print(f"\nCFAM features: {report['cfam_features']}")
    print(f"MCP servers documented: {len(MCP_SERVERS)}")
    print(f"Report: {kb / 'fetch_report.json'}")


if __name__ == "__main__":
    main()
