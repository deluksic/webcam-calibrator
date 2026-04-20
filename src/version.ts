// Build hash injected by vite-build-hash plugin
export const VERSION: string = (globalThis as unknown as { __BUILD_HASH__?: string }).__BUILD_HASH__ ?? 'dev'
console.log(`[build] ${VERSION}`)
