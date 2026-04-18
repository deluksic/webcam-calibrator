import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';
import typegpuPlugin from 'unplugin-typegpu/vite';
import tsover from 'tsover/plugin/vite';
import { buildHashPlugin } from './src/plugins/buildHash';

export default defineConfig({
  plugins: [
    tsover(),
    typegpuPlugin(),
    solidPlugin(),
    buildHashPlugin(),
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