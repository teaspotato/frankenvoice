import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const proxy = `http://127.0.0.1:${process.env.FV_PROXY_PORT ?? 8787}`;

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: proxy, ws: true, changeOrigin: true },
    },
  },
});
