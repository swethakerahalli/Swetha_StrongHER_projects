import React from 'react';
import Card from '@nokia-csf-uxr/ccfk/Card';
import Typography from '@nokia-csf-uxr/ccfk/Typography';
import StatusIndicator from '@nokia-csf-uxr/ccfk/StatusIndicator';
import ExpansionPanels, { ExpansionPanel } from '@nokia-csf-uxr/ccfk/ExpansionPanels';

export default function ConstraintsPanel({ data }) {
  const items = data?.constraints || [];
  const byCategory = data?.by_category || {};

  return (
    <div style={{ marginTop: 'var(--spacing-medium)' }}>
      <div style={{ display: 'flex', gap: 'var(--spacing-small)', flexWrap: 'wrap', marginBottom: 'var(--spacing-medium)' }}>
        {[
          ['Total', data?.total_constraints],
          ['Satisfied', data?.satisfied_count],
          ['Violated', data?.violated_count],
          ['Compliance', `${data?.compliance_pct}%`],
        ].map(([k, v]) => (
          <Card key={k} style={{ padding: 'var(--spacing-small) var(--spacing-medium)' }}>
            <Typography typography="CAPTION_12">{k}</Typography>
            <Typography typography="TITLE_16">{v}</Typography>
          </Card>
        ))}
      </div>

      <ExpansionPanels>
        {Object.entries(byCategory).map(([cat, info]) => (
          <ExpansionPanel key={cat} title={`${cat.toUpperCase()} (${info.satisfied}/${info.count} satisfied)`}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 'var(--spacing-small)' }}>
              {(info.items || []).map((c) => (
                <Card key={c.id} style={{ padding: 'var(--spacing-small)' }}>
                  <Typography typography="BODY_14_BOLD">{c.name}</Typography>
                  <Typography typography="BODY_14">
                    {c.current_value} {c.unit} {c.operator} {c.limit}
                  </Typography>
                  <StatusIndicator
                    status={c.satisfied ? 'success' : 'critical'}
                    label={c.satisfied ? `Margin ${c.margin_pct}%` : 'Violated'}
                  />
                  <Typography typography="CAPTION_12" style={{ color: 'var(--g-color-global-secondary-text)' }}>
                    {c.description}
                  </Typography>
                </Card>
              ))}
            </div>
          </ExpansionPanel>
        ))}
      </ExpansionPanels>

      <Card style={{ padding: 'var(--spacing-small)', marginTop: 'var(--spacing-medium)' }}>
        <img src="/plots/ran_constraints_dashboard.png" alt="Constraints" style={{ width: '100%', objectFit: 'contain' }} />
      </Card>
    </div>
  );
}
