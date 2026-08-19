const API = '';

export async function api(path, opts = {}) {
  const r = await fetch(API + path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!r.ok) throw new Error(`API ${path}: ${r.status}`);
  return r.json();
}

export async function loadDashboardData() {
  const [
    comparison, targets, agents, monitoring, constraints,
    agentPerf, carbon, greenAgents, traffic, coordination, plots, superAgent,
  ] = await Promise.all([
    api('/api/kpi/comparison'),
    api('/api/kpi/targets'),
    api('/api/agents/status'),
    api('/api/super-agent/monitoring'),
    api('/api/constraints/status'),
    api('/api/agents/performance'),
    api('/api/carbon/stats'),
    api('/api/green-agents/stats'),
    api('/api/traffic/stats'),
    api('/api/coordination/stats'),
    api('/api/plots/gallery'),
    api('/api/super-agent/status'),
  ]);
  return {
    comparison, targets, agents, monitoring, constraints,
    agentPerf, carbon, greenAgents, traffic, coordination, plots, superAgent,
  };
}
