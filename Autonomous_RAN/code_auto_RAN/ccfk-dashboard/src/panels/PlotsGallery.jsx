import React, { useState } from 'react';
import Card from '@nokia-csf-uxr/ccfk/Card';
import Typography from '@nokia-csf-uxr/ccfk/Typography';
import Tabs, { Tab, TabsContent } from '@nokia-csf-uxr/ccfk/Tabs';

export default function PlotsGallery({ data }) {
  const categories = data?.categories || [];
  const [cat, setCat] = useState(categories[0]?.id || 'all');
  const active = cat === 'all'
    ? categories.flatMap((c) => c.plots || [])
    : (categories.find((c) => c.id === cat)?.plots || []);

  return (
    <div style={{ marginTop: 'var(--spacing-medium)' }}>
      <Tabs scroll>
        <Tab id="all" selected={cat === 'all'} onSelect={() => setCat('all')}>All</Tab>
        {categories.map((c) => (
          <Tab key={c.id} id={c.id} selected={cat === c.id} onSelect={() => setCat(c.id)}>
            {c.label} ({c.count})
          </Tab>
        ))}
      </Tabs>
      <TabsContent>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
          gap: 'var(--spacing-small)',
          marginTop: 'var(--spacing-medium)',
        }}>
          {active.map((p) => (
            <Card
              key={p.url}
              style={{ padding: 'var(--spacing-small)', cursor: 'pointer' }}
              onClick={() => window.open(p.url, '_blank')}
            >
              <img
                src={p.url}
                alt={p.name}
                style={{ width: '100%', height: 'auto', objectFit: 'contain', minHeight: 120 }}
                onError={(e) => { e.target.style.display = 'none'; }}
              />
              <Typography typography="CAPTION_12" style={{ marginTop: 'var(--spacing-xsmall)' }}>
                {p.name}
              </Typography>
            </Card>
          ))}
        </div>
      </TabsContent>
    </div>
  );
}
