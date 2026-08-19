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
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8080',
      '/plots': 'http://localhost:8080',
    },
  },
  resolve: {
    alias: {
      '@nokia-csf-uxr/ccfk': path.resolve(__dirname, '../../ccfk/dist'),
    },
  },
  optimizeDeps: {
    include: ['react', 'react-dom', 'styled-components'],
  },
});
