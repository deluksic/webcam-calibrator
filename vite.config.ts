import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import typegpuPlugin from 'unplugin-typegpu/vite'
import { defineConfig, type Plugin } from 'vite'
import solidPlugin from 'vite-plugin-solid'

const root = path.dirname(fileURLToPath(import.meta.url))
const ortSourceDir = path.resolve(root, 'node_modules/onnxruntime-web/dist')
const ortAssetDir = path.resolve(root, 'public/ort')
const ORT_RUNTIME_ASSETS = [
  'ort-wasm-simd-threaded.asyncify.mjs',
  'ort-wasm-simd-threaded.asyncify.wasm',
] as const

function copyOrtRuntimeAssets(): void {
  fs.mkdirSync(ortAssetDir, { recursive: true })
  for (const filename of ORT_RUNTIME_ASSETS) {
    const source = path.join(ortSourceDir, filename)
    if (fs.existsSync(source)) {
      fs.copyFileSync(source, path.join(ortAssetDir, filename))
    }
  }
}

function ortRuntimeAssets(): Plugin {
  return {
    name: 'ort-runtime-assets',
    buildStart() {
      copyOrtRuntimeAssets()
    },
    configureServer(server) {
      copyOrtRuntimeAssets()
      server.middlewares.use((request, response, next) => {
        const pathname = (request.url ?? '').split('?')[0] ?? ''
        const marker = pathname.indexOf('/ort/')
        if (marker === -1) {
          next()
          return
        }
        const filename = path.basename(pathname)
        const filePath = path.join(ortAssetDir, filename)
        if (!filename || !fs.existsSync(filePath)) {
          next()
          return
        }
        response.statusCode = 200
        response.setHeader(
          'Content-Type',
          filename.endsWith('.wasm')
            ? 'application/wasm'
            : 'text/javascript',
        )
        response.setHeader('Cache-Control', 'no-cache')
        fs.createReadStream(filePath).pipe(response)
      })
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
    exclude: ['typegpu', '@typegpu/color', '@typegpu/geometry'],
  },
  resolve: {
    alias: {
      '@': path.resolve(root, 'src'),
    },
  },
  plugins: [
    ortRuntimeAssets(),
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
