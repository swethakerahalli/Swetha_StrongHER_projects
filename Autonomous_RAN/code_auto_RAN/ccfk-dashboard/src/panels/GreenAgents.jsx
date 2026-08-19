import React from 'react';
import Card from '@nokia-csf-uxr/ccfk/Card';
import Typography from '@nokia-csf-uxr/ccfk/Typography';
import Table, { TableHead, TableBody, TableRow, TableCell } from '@nokia-csf-uxr/ccfk/Table';

export default function GreenAgentsPanel({ carbon, green }) {
  const greenAgents = green?.agents || [];
  const gs = green?.summary || {};
  const ind = carbon?.industry || {};
  const auto = carbon?.autonomous || {};

  return (
    <div style={{ marginTop: 'var(--spacing-medium)' }}>
      <Typography typography="TITLE_16" style={{ marginBottom: 'var(--spacing-small)' }}>
        Carbon Emission Reduction
      </Typography>
      <div style={{ display: 'flex', gap: 'var(--spacing-small)', flexWrap: 'wrap', marginBottom: 'var(--spacing-medium)' }}>
        {[
          ['Industry CO₂', `${ind.carbon_kg_co2_per_h} kg/h`],
          ['Autonomous CO₂', `${auto.carbon_kg_co2_per_h} kg/h`],
          ['Reduction', `${auto.carbon_reduction_pct}%`],
          ['Renewable', `${auto.renewable_pct}%`],
        ].map(([k, v]) => (
          <Card key={k} style={{ padding: 'var(--spacing-small) var(--spacing-medium)' }}>
            <Typography typography="CAPTION_12">{k}</Typography>
            <Typography typography="TITLE_16">{v}</Typography>
          </Card>
        ))}
      </div>
      <Card style={{ padding: 'var(--spacing-small)', marginBottom: 'var(--spacing-medium)' }}>
        <img src="/plots/carbon_emission_reduction.png" alt="Carbon" style={{ width: '100%', objectFit: 'contain' }} />
      </Card>

      <Typography typography="TITLE_16" style={{ marginBottom: 'var(--spacing-small)' }}>
        Green &amp; Edge Agents
      </Typography>
      <div style={{ display: 'flex', gap: 'var(--spacing-small)', flexWrap: 'wrap', marginBottom: 'var(--spacing-medium)' }}>
        {[
          ['Green Agents', gs.total_green_agents],
          ['Power Reduction', `${gs.combined_power_reduction_pct}%`],
          ['Carbon Reduction', `${gs.combined_carbon_reduction_pct}%`],
          ['Renewable Gain', `+${gs.renewable_pct_gain}%`],
        ].map(([k, v]) => (
          <Card key={k} style={{ padding: 'var(--spacing-small) var(--spacing-medium)' }}>
            <Typography typography="CAPTION_12">{k}</Typography>
            <Typography typography="TITLE_16">{v}</Typography>
          </Card>
        ))}
      </div>
      <Card style={{ padding: 0, overflow: 'auto', marginBottom: 'var(--spacing-medium)' }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell as="th">Agent</TableCell>
              <TableCell as="th">Metric</TableCell>
              <TableCell as="th">Baseline</TableCell>
              <TableCell as="th">Autonomous</TableCell>
              <TableCell as="th">Improvement</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {greenAgents.map((a) => (
              <TableRow key={a.agent}>
                <TableCell>{a.label}</TableCell>
                <TableCell>{a.metric}</TableCell>
                <TableCell>{a.baseline}</TableCell>
                <TableCell>{a.autonomous}</TableCell>
                <TableCell>+{a.improvement_pct}%</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
      <Card style={{ padding: 'var(--spacing-small)' }}>
        <img src="/plots/green_edge_agents_impact.png" alt="Green edge" style={{ width: '100%', objectFit: 'contain' }} />
      </Card>
    </div>
  );
}
