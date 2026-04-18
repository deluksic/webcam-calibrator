// Build hash injected by vite-build-hash plugin
declare global {
  interface Window {
    __BUILD_HASH__?: string;
  }
}
export const VERSION: string = window.__BUILD_HASH__ ?? 'dev';
console.log(`[build] ${VERSION}`);
