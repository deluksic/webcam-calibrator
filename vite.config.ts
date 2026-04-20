import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import solidPlugin from "vite-plugin-solid";
import typegpuPlugin from "unplugin-typegpu/vite";
import tsover from "tsover/plugin/vite";
import { buildHashPlugin } from "./src/plugins/buildHash";

const root = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(root, "src"),
    },
  },
  plugins: [tsover(), typegpuPlugin(), solidPlugin(), buildHashPlugin()],
  server: {
    port: 5173,
    host: "0.0.0.0",
  },
  build: {
    target: "esnext",
    minify: false,
  },
});
