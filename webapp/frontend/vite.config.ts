import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The API base is injected at build time (VITE_API_BASE) so the same source
// builds for local dev and for a hosted deployment. In dev the proxy below
// keeps everything same-origin, which avoids needing CORS at all while
// developing.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_API_TARGET ?? 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
