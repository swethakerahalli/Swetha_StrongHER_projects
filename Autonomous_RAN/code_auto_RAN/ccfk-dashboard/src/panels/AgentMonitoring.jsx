import React from 'react';
import Card from '@nokia-csf-uxr/ccfk/Card';
import Typography from '@nokia-csf-uxr/ccfk/Typography';
import Table, { TableHead, TableBody, TableRow, TableCell } from '@nokia-csf-uxr/ccfk/Table';
import StatusIndicator from '@nokia-csf-uxr/ccfk/StatusIndicator';

function statusType(s) {
  if (s === 'healthy') return 'success';
  if (s === 'warning') return 'minor';
  return 'critical';
}

export default function AgentMonitoringPanel({ data, superAgent }) {
  const mon = data?.monitoring || {};
  const agents = mon.agents || [];
  const opts = data?.optimization_actions || [];

  return (
    <div style={{ marginTop: 'var(--spacing-medium)' }}>
      <div style={{ display: 'flex', gap: 'var(--spacing-small)', flexWrap: 'wrap', marginBottom: 'var(--spacing-medium)' }}>
        {[
          ['Monitored', mon.monitored_count],
          ['Healthy', mon.healthy_count],
          ['Warnings', mon.warning_count],
          ['Degraded', mon.degraded_count],
          ['Optimizations', opts.length],
          ['Cycles', superAgent?.monitoring_cycles || 0],
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
              <TableCell as="th">Status</TableCell>
              <TableCell as="th">Perf. Index</TableCell>
              <TableCell as="th">Improvement</TableCell>
              <TableCell as="th">Degradation</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {agents.map((a) => (
              <TableRow key={a.agent}>
                <TableCell>{a.agent_label}</TableCell>
                <TableCell>
                  <StatusIndicator status={statusType(a.status)} label={a.status.toUpperCase()} />
                </TableCell>
                <TableCell>{a.performance_index}</TableCell>
                <TableCell>{a.improvement_pct}%</TableCell>
                <TableCell>{a.degradation_score}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Card>
      {opts.length > 0 && (
        <Card style={{ padding: 'var(--spacing-medium)' }}>
          <Typography typography="TITLE_16">Agent Optimizer Actions</Typography>
          {opts.map((o, i) => (
            <Typography key={i} typography="BODY_14" style={{ marginTop: 'var(--spacing-xsmall)' }}>
              {o.params?.target_agent}: {o.params?.optimization_action}
              (recovery +{o.params?.expected_recovery_pct}%)
            </Typography>
          ))}
        </Card>
      )}
      <Card style={{ padding: 'var(--spacing-small)', marginTop: 'var(--spacing-medium)' }}>
        <img
          src="/plots/agent_optimizer_monitoring.png"
          alt="Agent optimizer monitoring"
          style={{ width: '100%', height: 'auto', objectFit: 'contain' }}
        />
      </Card>
    </div>
  );
}
