import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  base: '/dashboard/ccfk/',
  build: {
    outDir: '../static/ccfk-dashboard',
    emptyOutDir: true,
  },
  server: {
    port: 5174,
    proxy: { '/api': 'http://localhost:8090', '/plots': 'http://localhost:8090' },
  },
  resolve: {
    alias: {
      '@nokia-csf-uxr/ccfk': path.resolve(__dirname, '../../../ccfk/dist'),
    },
  },
});
