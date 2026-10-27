import path from 'node:path'
import { fileURLToPath } from 'node:url'

import typegpuPlugin from 'unplugin-typegpu/vite'
import { defineConfig, type Plugin } from 'vite'
import solidPlugin from 'vite-plugin-solid'

import { ensureModelAsset } from './scripts/modelAsset'

const root = path.dirname(fileURLToPath(import.meta.url))
function modelWeights(): Plugin {
  const log = (message: string): void => {
    console.log(`[model] ${message}`)
  }
  return {
    name: 'model-weights',
    async buildStart() {
      if (!process.env.VITEST) {
        await ensureModelAsset({ log })
      }
    },
    async configureServer() {
      if (!process.env.VITEST) {
        await ensureModelAsset({ log })
      }
    },
  }
}

export default defineConfig({
  // GitHub project pages: set VITE_BASE_URL=/<repo>/ in CI (see .github/workflows)
  base: process.env.VITE_BASE_URL || '/',
  // Git-linked TypeGPU packages ship TypeScript sources. In dev, Vite pre-bundles many
  // deps with esbuild only (no unplugin-typegpu), so `tgpu.fn` / shader metadata breaks.
  // Production uses Rollup + our plugins. Excluding these forces the transform pipeline in dev.
  optimizeDeps: {
    // ruNNtime ships TS sources straight into the Vite transform pipeline (same
    // reason as the TypeGPU entries below): esbuild pre-bundling would drop the
    // unplugin-typegpu metadata that its kernels need.
    exclude: ['typegpu', '@typegpu/color', '@typegpu/geometry', 'runntime'],
  },
  resolve: {
    alias: {
      '@': path.resolve(root, 'src'),
      // `runntime/core` is not in the package's `exports` map (only ./zoo), so
      // this alias resolves straight to the source file. Drop it — and rename
      // the imports back — once the package exports ./core.
      '@runntime/core': path.resolve(root, 'node_modules/runntime/src/core/index.ts'),
    },
  },
  plugins: [
    modelWeights(),
    // tsover(), disabled because slow
    typegpuPlugin(),
    solidPlugin(),
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
