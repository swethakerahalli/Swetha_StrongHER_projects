import React from 'react';
import Card from '@nokia-csf-uxr/ccfk/Card';
import Typography from '@nokia-csf-uxr/ccfk/Typography';
import Table, { TableHead, TableBody, TableRow, TableCell } from '@nokia-csf-uxr/ccfk/Table';
import StatusIndicator from '@nokia-csf-uxr/ccfk/StatusIndicator';

export default function AgentPerformancePanel({ data }) {
  const agents = data?.agents || [];
  const summary = data?.summary || {};

  return (
    <div style={{ marginTop: 'var(--spacing-medium)' }}>
      <div style={{ display: 'flex', gap: 'var(--spacing-small)', flexWrap: 'wrap', marginBottom: 'var(--spacing-medium)' }}>
        {[
          ['Agents', summary.total_agents],
          ['Improved', summary.improved_count],
          ['Avg Improvement', `${summary.avg_improvement_pct}%`],
          ['Avg Perf. Index', summary.avg_performance_index],
        ].map(([k, v]) => (
          <Card key={k} style={{ padding: 'var(--spacing-small) var(--spacing-medium)' }}>
            <Typography typography="CAPTION_12">{k}</Typography>
            <Typography typography="TITLE_16">{v}</Typography>
          </Card>
        ))}
      </div>
      <Card style={{ padding: 0, overflow: 'auto' }}>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell as="th">Agent</TableCell>
              <TableCell as="th">Role</TableCell>
              <TableCell as="th">Baseline</TableCell>
              <TableCell as="th">Autonomous</TableCell>
              <TableCell as="th">Improvement</TableCell>
              <TableCell as="th">Perf. Index</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {agents.map((a) => (
              <TableRow key={a.agent}>
                <TableCell><strong>{a.agent_label}</strong></TableCell>
                <TableCell>{a.role}</TableCell>
                <TableCell>{a.baseline_value}</TableCell>
                <TableCell>{a.autonomous_value}</TableCell>
                <TableCell>
                  <StatusIndicator
                    status={a.improvement_pct > 0 ? 'success' : 'minor'}
                    label={`${a.improvement_pct > 0 ? '+' : ''}${a.improvement_pct}%`}
                  />
                </TableCell>
                <TableCell>{a.performance_index}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
    </div>
  );
}
