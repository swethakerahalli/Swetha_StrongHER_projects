import React, { useEffect, useState } from 'react';
import AdvancedTheme from '@nokia-csf-uxr/ccfk/AdvancedTheme';
import App, { AppHeader, AppBody, AppContentWrapper, AppContent } from '@nokia-csf-uxr/ccfk/App';
import AppBanner, { AppBannerLogo, AppBannerName, AppBannerNameSecondary } from '@nokia-csf-uxr/ccfk/AppBanner';
import Tabs, { Tab } from '@nokia-csf-uxr/ccfk/Tabs';
import Button from '@nokia-csf-uxr/ccfk/Button';
import Card from '@nokia-csf-uxr/ccfk/Card';
import Typography from '@nokia-csf-uxr/ccfk/Typography';
import StatusIndicator from '@nokia-csf-uxr/ccfk/StatusIndicator';
import Spinner from '@nokia-csf-uxr/ccfk/Spinner';

async function api(path, opts = {}) {
  const r = await fetch(path, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (!r.ok) throw new Error(path);
  return r.json();
}

const TABS = ['Overview', 'Agents', 'Control', 'Twin', 'Plots'];

export default function ChannelApp() {
  const [tab, setTab] = useState(0);
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  const refresh = async () => {
    try {
      const [comparison, targets, agents, twin, plots, coord, superAgent] = await Promise.all([
        api('/api/kpi/comparison'),
        api('/api/kpi/targets'),
        api('/api/agents/status'),
        api('/api/twin/state'),
        api('/api/plots/gallery'),
        api('/api/coordination/stats'),
        api('/api/super-agent/status'),
      ]);
      setData({ comparison, targets, agents, twin, plots, coord, superAgent });
      setErr(null);
    } catch (e) {
      setErr(e.message);
    }
  };

  useEffect(() => { refresh(); }, []);
  const arch = data?.comparison?.architecture || {};

  return (
    <AdvancedTheme advancedTheme="CCFK FreeForm - Dark">
      <App rtl={false} disableAnimation={false}>
        <AppHeader>
          <AppBanner>
            <AppBannerLogo />
            <AppBannerName>6G AI Channel Estimation</AppBannerName>
            <AppBannerNameSecondary>CCFK — Digital Twin, Agents, 3GPP-aligned CSI</AppBannerNameSecondary>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <StatusIndicator status="success" label="LIVE" />
              <Button variant="neutral" onClick={() => { window.location.href = '/dashboard'; }}>Classic</Button>
              <Button variant="brand-primary" onClick={refresh}>Refresh</Button>
            </div>
          </AppBanner>
        </AppHeader>
        <AppBody>
          <AppContentWrapper>
            <AppContent style={{ padding: 'var(--spacing-medium)' }}>
              <Tabs selected={tab} alignment="left">
                {TABS.map((label, i) => (
                  <Tab key={label} onClick={() => setTab(i)}>{label}</Tab>
                ))}
              </Tabs>
              {!data && !err && <Spinner />}
              {err && <Typography>API error: {err}</Typography>}
              {data && tab === 0 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginTop: 16 }}>
                  {[
                    ['AI NMSE', arch.test_nmse_ai],
                    ['MMSE NMSE', arch.test_nmse_mmse],
                    ['NMSE gain %', arch.nmse_improvement_pct],
                    ['BER reduction %', arch.ber_reduction_pct],
                    ['SE gain %', arch.spectral_efficiency_gain_pct],
                    ['CSI pred %', arch.csi_prediction_accuracy],
                  ].map(([k, v]) => (
                    <Card key={k} style={{ padding: 16, minWidth: 140 }}>
                      <Typography typography="CAPTION_12">{k}</Typography>
                      <Typography typography="TITLE_24">{v ?? '—'}</Typography>
                    </Card>
                  ))}
                </div>
              )}
              {data && tab === 1 && (
                <div style={{ marginTop: 16 }}>
                  {(data.agents.agents || []).map((a) => (
                    <Card key={a.id} style={{ padding: 12, marginBottom: 8 }}>
                      <Typography typography="TITLE_16">{a.id}</Typography>
                      <Typography typography="BODY_14">{JSON.stringify(a.metrics).slice(0, 240)}</Typography>
                    </Card>
                  ))}
                </div>
              )}
              {data && tab === 2 && (
                <div style={{ marginTop: 16 }}>
                  <Card style={{ padding: 16, marginBottom: 12 }}>
                    <Typography typography="TITLE_16">Coordinator</Typography>
                    <Typography typography="BODY_14">cycles: {data.coord?.cycles} · conflicts: {data.coord?.total_conflicts} · strategy: {data.coord?.last?.strategy || '—'}</Typography>
                    {(data.coord?.last_resolutions || []).map((r) => (
                      <Typography key={r} typography="BODY_14">{r}</Typography>
                    ))}
                    <img src="/plots/coordination_conflicts.png" alt="coordination" style={{ width: '100%', maxWidth: 720, background: '#0b1220', marginTop: 8 }} />
                  </Card>
                  <Card style={{ padding: 16 }}>
                    <Typography typography="TITLE_16">Super agent</Typography>
                    <Typography typography="BODY_14">approved: {data.superAgent?.last_run?.approved_count ?? '—'} · rejected: {data.superAgent?.last_run?.rejected_count ?? '—'} · utility: {data.superAgent?.last_run?.global_utility ?? '—'}</Typography>
                    <img src="/plots/super_agent_control.png" alt="super" style={{ width: '100%', maxWidth: 720, background: '#0b1220', marginTop: 8 }} />
                  </Card>
                </div>
              )}
              {data && tab === 3 && (
                <Card style={{ padding: 16, marginTop: 16 }}>
                  <Typography typography="TITLE_16">Digital twin</Typography>
                  {Object.entries(data.twin).map(([k, v]) => (
                    <Typography key={k} typography="BODY_14">{k}: {String(v)}</Typography>
                  ))}
                </Card>
              )}
              {data && tab === 4 && (
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 16 }}>
                  {(data.plots.plots || []).map((p) => (
                    <img key={p} src={`/plots/${p}`} alt={p} style={{ width: 220, background: '#0b1220' }} />
                  ))}
                </div>
              )}
            </AppContent>
          </AppContentWrapper>
        </AppBody>
      </App>
    </AdvancedTheme>
  );
}
