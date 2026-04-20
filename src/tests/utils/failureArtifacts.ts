/**
 * Failure PNG dumps under `output/test-failures/<file>/<test-name>/`.
 *
 * Call **`attachFailureArtifacts(import.meta.url, (dir) => { ... })`** from inside an `it()` body,
 * **after** you have generated the data (same place you’d add debug visualizations). It registers
 * Vitest’s `onTestFailed` for the **current** test — no env vars, no try/catch around `expect`.
 *
 * **Must** be invoked synchronously while the test is running (e.g. from a helper called directly
 * from `it()`, not from `describe` setup or `beforeAll`).
 */
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { onTestFailed } from "vitest";
import { PNG } from "pngjs";

import type { TagPattern } from "../../lib/tag36h11";

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, "..", "..");
export const TEST_FAILURE_OUTPUT_ROOT = join(REPO_ROOT, "output", "test-failures");

export interface FailureTask {
  name: string;
}

/** Output path for a failed test; the directory is not created until a write runs. */
export function failureDirForTask(testFileUrl: string, task: FailureTask): string {
  const fileLabel = sanitizeSegment(
    testFileUrl.replace(/^.*\/src\//, "src/").replace(/\.test\.ts$/, ""),
  );
  const nameLabel = sanitizeSegment(task.name);
  return join(TEST_FAILURE_OUTPUT_ROOT, fileLabel, nameLabel);
}

function ensureParentDir(filePath: string): void {
  mkdirSync(dirname(filePath), { recursive: true });
}

function sanitizeSegment(s: string): string {
  return s
    .replace(/[^a-zA-Z0-9._/-]+/g, "_")
    .replace(/\/+/g, "/")
    .slice(0, 200);
}

/**
 * Register PNG writers for the current test if it fails. Call once per test after data exists.
 * `write(dir)` receives the output path; nothing is created on disk until a `write*Png` runs.
 */
export function attachFailureArtifacts(testFileUrl: string, write: (dir: string) => void): void {
  onTestFailed(({ task }) => {
    const dir = failureDirForTask(testFileUrl, { name: String(task.name ?? task.id ?? "test") });
    try {
      write(dir);
    } catch (e) {
      console.error("[failureArtifacts] write failed", e);
    }
  });
}

export function writeGreyPng(
  path: string,
  width: number,
  height: number,
  grey01: Float32Array,
): void {
  const data = Buffer.alloc(width * height);
  for (let i = 0; i < grey01.length; i++) {
    data[i] = Math.round(Math.min(255, Math.max(0, grey01[i]! * 255)));
  }
  const png = PNG.sync.write(
    { width, height, data },
    { colorType: 0, inputColorType: 0, inputHasAlpha: false, bitDepth: 8 },
  );
  ensureParentDir(path);
  writeFileSync(path, png);
}

export function writeSobelMagPng(
  path: string,
  width: number,
  height: number,
  sobel: Float32Array,
): void {
  let m = 0;
  const mag = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const gx = sobel[i * 2]!;
    const gy = sobel[i * 2 + 1]!;
    mag[i] = Math.hypot(gx, gy);
    m = Math.max(m, mag[i]!);
  }
  const inv = m > 1e-12 ? 1 / m : 1;
  const data = Buffer.alloc(width * height);
  for (let i = 0; i < mag.length; i++) {
    data[i] = Math.round(Math.min(255, Math.max(0, mag[i]! * inv * 255)));
  }
  const png = PNG.sync.write(
    { width, height, data },
    { colorType: 0, inputColorType: 0, inputHasAlpha: false, bitDepth: 8 },
  );
  ensureParentDir(path);
  writeFileSync(path, png);
}

export function writeCellLegendPng(
  path: string,
  decoded: TagPattern,
  truth: TagPattern,
  scale: number,
): void {
  const cw = 6 * scale;
  const ch = 6 * scale;
  const rgba = Buffer.alloc(cw * ch * 4);
  for (let row = 0; row < 6; row++) {
    for (let col = 0; col < 6; col++) {
      const i = row * 6 + col;
      const d = decoded[i]!;
      const t = truth[i]!;
      let r = 0;
      let g = 0;
      let b = 0;
      if (d === -1) {
        r = 80;
        g = 120;
        b = 255;
      } else if (d === -2) {
        r = 220;
        g = 80;
        b = 255;
      } else if (d !== t) {
        r = 255;
        g = 60;
        b = 60;
      } else {
        r = 40;
        g = 220;
        b = 80;
      }
      for (let dy = 0; dy < scale; dy++) {
        for (let dx = 0; dx < scale; dx++) {
          const x = col * scale + dx;
          const y = row * scale + dy;
          const o = (y * cw + x) * 4;
          rgba[o] = r;
          rgba[o + 1] = g;
          rgba[o + 2] = b;
          rgba[o + 3] = 255;
        }
      }
    }
  }
  const png = PNG.sync.write({ width: cw, height: ch, data: rgba });
  ensureParentDir(path);
  writeFileSync(path, png);
}
