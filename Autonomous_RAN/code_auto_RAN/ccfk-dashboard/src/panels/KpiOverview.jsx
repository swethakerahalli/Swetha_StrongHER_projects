import React from 'react';
import Card from '@nokia-csf-uxr/ccfk/Card';
import Typography from '@nokia-csf-uxr/ccfk/Typography';
import StatusIndicator from '@nokia-csf-uxr/ccfk/StatusIndicator';

function KpiCard({ label, before, after, delta, invert }) {
  const good = invert ? delta < 0 : delta > 0;
  return (
    <Card style={{ padding: 'var(--spacing-medium)', minWidth: 160, flex: '1 1 160px' }}>
      <Typography typography="CAPTION_12" style={{ color: 'var(--g-color-global-secondary-text)' }}>
        {label}
      </Typography>
      <Typography typography="TITLE_24">{after}</Typography>
      <Typography typography="CAPTION_12" style={{ color: 'var(--g-color-global-secondary-text)' }}>
        {before} → {after}
      </Typography>
      <StatusIndicator
        status={good ? 'success' : 'minor'}
        label={`${delta > 0 ? '+' : ''}${delta}%`}
      />
    </Card>
  );
}

export default function KpiOverview({ data }) {
  const cmp = data.comparison || {};
  const b = cmp.before || {};
  const a = cmp.after || {};
  const d = cmp.delta_pct || {};
  const metrics = [
    ['Throughput (Mbps)', 'avg_throughput_mbps', false],
    ['Latency (ms)', 'avg_latency_ms', true],
    ['Power (W)', 'total_power_w', true],
    ['QoS SLA', 'qos_sla_compliance', false],
    ['Security', 'security_score', false],
    ['Slice Efficiency', 'slice_efficiency', false],
    ['Renewable %', 'renewable_pct', false],
    ['Carbon (kg/h)', 'carbon_kg_co2_per_h', true],
  ];

  return (
    <div style={{ marginTop: 'var(--spacing-medium)' }}>
      <Typography typography="TITLE_16" style={{ marginBottom: 'var(--spacing-small)' }}>
        {cmp.before_label || 'Industry Baseline'} → {cmp.after_label || 'Autonomous RAN'}
      </Typography>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--spacing-small)' }}>
        {metrics.map(([label, key, invert]) => (
          <KpiCard
            key={key}
            label={label}
            before={b[key]}
            after={a[key]}
            delta={d[key] || 0}
            invert={invert}
          />
        ))}
      </div>
      <Card style={{ padding: 'var(--spacing-medium)', marginTop: 'var(--spacing-medium)' }}>
        <Typography typography="TITLE_16">Target KPIs</Typography>
        {(data.targets?.targets || []).map((t) => (
          <div key={t.label} style={{ marginTop: 'var(--spacing-small)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography typography="BODY_14">{t.label}</Typography>
              <Typography typography="BODY_14">
                {t.current_value}{t.unit === 'percent' ? '%' : ''} / {t.target_value}{t.unit === 'percent' ? '%' : ''}
              </Typography>
            </div>
            <div style={{ height: 6, background: 'var(--g-color-surface-2)', borderRadius: 3, marginTop: 4 }}>
              <div style={{
                width: `${Math.min(100, t.progress_pct)}%`,
                height: '100%',
                background: t.achieved ? 'var(--g-color-global-positive)' : 'var(--g-color-global-brand)',
                borderRadius: 3,
              }} />
            </div>
          </div>
        ))}
      </Card>
    </div>
  );
}
