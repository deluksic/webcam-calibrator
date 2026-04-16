import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';
import typegpuPlugin from 'unplugin-typegpu/vite';

export default defineConfig({
  plugins: [
    typegpuPlugin(),
    solidPlugin(),
  ],
  server: {
    port: 5173,
    host: '0.0.0.0',
  },
  build: {
    target: 'esnext',
    minify: false,
  },
});