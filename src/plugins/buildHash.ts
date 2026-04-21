import type { Plugin } from 'vite'

import packageJson from '../../package.json' with { type: 'json' }

const { abs } = Math
const BASE_VERSION = packageJson.version

// Compute a short hash from chunk content (base64 of first 12 chars, stripped of special chars)
function contentHash(code: string): string {
  let h = 0
  for (let i = 0; i < code.length; i++) {
    h = ((h << 5) - h + code.charCodeAt(i)) | 0
  }
  return abs(h).toString(16).slice(0, 8)
}

export function buildHashPlugin(): Plugin {
  return {
    name: 'vite-build-hash',

    renderChunk(code, chunk) {
      // Only inject into the main entry chunk
      if (chunk.isEntry && chunk.fileName.includes('index-')) {
        const hash = contentHash(code)
        return `window.__BUILD_HASH__ = "${BASE_VERSION}-${hash}";\n${code}`
      }
    },
  }
}
