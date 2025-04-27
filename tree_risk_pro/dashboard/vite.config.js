// vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5174,
    open: true,
    // Enable more detailed logging
    hmr: {
      overlay: true
    }
  },
  // Add clear error messages
  build: {
    sourcemap: true
  },
  // Log level
  logLevel: 'info'
});