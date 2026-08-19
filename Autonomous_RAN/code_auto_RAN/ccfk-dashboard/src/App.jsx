import React, { useEffect, useState } from 'react';
import AdvancedTheme from '@nokia-csf-uxr/ccfk/AdvancedTheme';
import App, {
  AppHeader, AppBody, AppContentWrapper, AppContent,
} from '@nokia-csf-uxr/ccfk/App';
import AppBanner, {
  AppBannerLogo, AppBannerName, AppBannerNameSecondary,
} from '@nokia-csf-uxr/ccfk/AppBanner';
import Tabs, { Tab, TabsContent } from '@nokia-csf-uxr/ccfk/Tabs';
import Button from '@nokia-csf-uxr/ccfk/Button';
import Spinner from '@nokia-csf-uxr/ccfk/Spinner';
import StatusIndicator from '@nokia-csf-uxr/ccfk/StatusIndicator';
import { loadDashboardData } from './api';
import KpiOverview from './panels/KpiOverview';
import AgentMonitoringPanel from './panels/AgentMonitoring';
import TrafficPanel from './panels/Traffic';
import CoordinationPanel from './panels/Coordination';
import GreenAgentsPanel from './panels/GreenAgents';
import ConstraintsPanel from './panels/Constraints';
import PlotsGallery from './panels/PlotsGallery';
import AgentPerformancePanel from './panels/AgentPerformance';

const TABS = [
  { id: 'overview', label: 'KPI Overview' },
  { id: 'agents', label: 'Agent Performance' },
  { id: 'monitoring', label: 'Super Agent Monitoring' },
  { id: 'traffic', label: 'Traffic Agent' },
  { id: 'coordination', label: 'Coordination' },
  { id: 'green', label: 'Green & Carbon' },
  { id: 'constraints', label: 'Constraints' },
  { id: 'plots', label: 'Visualizations' },
];

export default function AutonomousRanApp() {
  const [tab, setTab] = useState('overview');
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    setLoading(true);
    setError(null);
    try {
      setData(await loadDashboardData());
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { refresh(); }, []);

  return (
    <AdvancedTheme advancedTheme="CCFK FreeForm - Dark">
      <App rtl={false} disableAnimation={false}>
        <AppHeader>
          <AppBanner>
            <AppBannerLogo />
            <AppBannerName>Autonomous Intelligent RAN</AppBannerName>
            <AppBannerNameSecondary>CCFK Dashboard — Digital Twin &amp; 24 AI Agents</AppBannerNameSecondary>
            <div style={{ marginLeft: 'auto', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
              <StatusIndicator status="success" label="LIVE" />
              <StatusIndicator status="info" label="CCFK" />
              <Button variant="neutral" onClick={() => { window.location.href = '/dashboard/classic'; }}>
                Classic Dashboard
              </Button>
              <Button variant="brand-primary" onClick={refresh}>Refresh</Button>
            </div>
          </AppBanner>
        </AppHeader>
        <AppBody>
          <AppContentWrapper>
            <AppContent style={{ padding: 'var(--spacing-medium)' }}>
              {loading && !data && (
                <div style={{ display: 'flex', justifyContent: 'center', padding: '2rem' }}>
                  <Spinner />
                </div>
              )}
              {error && (
                <div style={{ color: 'var(--g-color-global-negative)', padding: '1rem' }}>
                  Failed to load dashboard: {error}. Ensure API is running on port 8080.
                </div>
              )}
              {data && (
                <>
                  <Tabs scroll arrowKeyNavigation>
                    {TABS.map((t) => (
                      <Tab
                        key={t.id}
                        id={t.id}
                        selected={tab === t.id}
                        onSelect={() => setTab(t.id)}
                      >
                        {t.label}
                      </Tab>
                    ))}
                  </Tabs>
                  <TabsContent>
                    {tab === 'overview' && <KpiOverview data={data} />}
                    {tab === 'agents' && <AgentPerformancePanel data={data.agentPerf} />}
                    {tab === 'monitoring' && <AgentMonitoringPanel data={data.monitoring} superAgent={data.superAgent} />}
                    {tab === 'traffic' && <TrafficPanel data={data.traffic} />}
                    {tab === 'coordination' && <CoordinationPanel data={data.coordination} />}
                    {tab === 'green' && <GreenAgentsPanel carbon={data.carbon} green={data.greenAgents} />}
                    {tab === 'constraints' && <ConstraintsPanel data={data.constraints} />}
                    {tab === 'plots' && <PlotsGallery data={data.plots} />}
                  </TabsContent>
                </>
              )}
            </AppContent>
          </AppContentWrapper>
        </AppBody>
      </App>
    </AdvancedTheme>
  );
}
