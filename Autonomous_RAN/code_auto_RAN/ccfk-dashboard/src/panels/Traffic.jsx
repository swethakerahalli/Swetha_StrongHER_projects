import React from 'react';
import Card from '@nokia-csf-uxr/ccfk/Card';
import Typography from '@nokia-csf-uxr/ccfk/Typography';
import Table, { TableHead, TableBody, TableRow, TableCell } from '@nokia-csf-uxr/ccfk/Table';
import StatusIndicator from '@nokia-csf-uxr/ccfk/StatusIndicator';

export default function TrafficPanel({ data }) {
  const m = data?.metrics || {};
  const s = data?.summary || {};

  return (
    <div style={{ marginTop: 'var(--spacing-medium)' }}>
      <div style={{ display: 'flex', gap: 'var(--spacing-small)', flexWrap: 'wrap', marginBottom: 'var(--spacing-medium)' }}>
        {[
          ['Congestion ↓', `${s.congestion_reduction_pct}%`],
          ['Peak Gain', `+${s.peak_throughput_gain_pct}%`],
          ['Contribution', `${s.traffic_agent_contribution_pct}%`],
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
              <TableCell as="th">Metric</TableCell>
              <TableCell as="th">Industry</TableCell>
              <TableCell as="th">Autonomous</TableCell>
              <TableCell as="th">Improvement</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {Object.entries(m).map(([key, d]) => (
              <TableRow key={key}>
                <TableCell>{key.replace(/_/g, ' ')}</TableCell>
                <TableCell>{d.industry}</TableCell>
                <TableCell>{d.autonomous}</TableCell>
                <TableCell>
                  <StatusIndicator status="success" label={`+${d.improvement_pct}%`} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
      <Card style={{ padding: 'var(--spacing-small)' }}>
        <img src="/plots/traffic_agent_impact.png" alt="traffic" style={{ width: '100%', objectFit: 'contain' }} />
      </Card>
    </div>
  );
}
