import path from 'node:path'
import { fileURLToPath } from 'node:url'

import typegpuPlugin from 'unplugin-typegpu/vite'
import { defineConfig } from 'vite'
import solidPlugin from 'vite-plugin-solid'

import { buildHashPlugin } from './src/plugins/buildHash'

const root = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  // GitHub project pages: set VITE_BASE_URL=/<repo>/ in CI (see .github/workflows)
  base: process.env.VITE_BASE_URL || '/',
  resolve: {
    alias: {
      '@': path.resolve(root, 'src'),
    },
  },
  plugins: [
    // tsover(), disabled because slow
    typegpuPlugin(),
    solidPlugin(),
    buildHashPlugin(),
  ],
  server: {
    port: 5173,
    host: '0.0.0.0',
    fs: {
      allow: ['.', '../opencv-calibration-wasm'],
    },
  },
  build: {
    target: 'esnext',
    minify: true,
    sourcemap: true,
  },
  worker: {
    format: 'es',
  },
})
