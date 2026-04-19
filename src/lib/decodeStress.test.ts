/**
 * Characterizes `decodeTagPattern` + dictionary decode under **perspective** and **low resolution**.
 * Tightening these numbers documents the cliff; adjust if pipeline changes.
 *
 * **Supersampling:** each output pixel averages `ss×ss` tag samples on a regular subpixel grid inside the
 * pixel square (`(sx+0.5)/ss` offsets), inverse‑warped through the quad homography—so `ss` is purely a
 * nicer antialiased intensity raster, then **central-difference Sobel** runs on that grid. `ss === 2` can
 * resonate badly with FD Sobel; this file uses **4** (same spirit as other decode tests here).
 *
 * **Speckle:** after the clean raster, we add **deterministic ±amplitude** uniform noise in linear intensity,
 * clamped to `[0,1]`, then Sobel. Amplitude is `DECODE_STRESS_SPECKLE_AMP` in `decodeStressHarness.ts`, tuned
 * just below the empirical cliff (first failure ≈**0.4925** at `wh=160` for the pre–two-bin decode). Re-measure with
 * `pnpm run find:decode-stress-speckle` after pipeline changes.
 *
 * **Homography mismatch:** raster uses the fitted quad; decode uses corners offset by
 * `DECODE_STRESS_HOMOGRAPHY_MISMATCH_SCALE *` `DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX`
 * (`decodeStressStripWithHomographyMismatchOffsetsPx`); `buildTagGrid` recomputes H from the perturbed strip.
 * Re-tune scale with `pnpm run find:decode-stress-homography` (grid scan; `passes(scale)` is not
 * guaranteed monotone).
 *
 * **Debug PNGs:** `WRITE_DECODE_STRESS_PNGS=1 pnpm exec vitest run src/lib/decodeStress.test.ts`
 * writes imperfect cases to `output/decode-stress/` (intensity, Sobel magnitude, 6×6 cell RGB legend).
 *
 * **Homography diagnostic PNGs:** `pnpm run test:decode-stress-homography-3px-png` writes **imperfect**
 * cases only (same rule as `WRITE_DECODE_STRESS_PNGS`: skip when `cellErr`, Hamming `dist`, and
 * `unknowns` are all zero) under `output/decode-stress/homography-3px/` (**no speckle**; decode quad
 * offset so template **max |Δx|,|Δy|** ≈ **3px** — tune `HOMOGRAPHY_DIAG_MAX_AXIS_PX` in the test file).
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';
import { PNG } from 'pngjs';

import type { Point } from './geometry';
import { buildTagGrid, decodeTagPattern } from './grid';
import {
  DECODE_STRESS_SIZES,
  DECODE_STRESS_SPECKLE_AMP,
  decodeStressAxisStrip,
  decodeStressCornersGridOrder,
  decodeStressFitPerspectiveStrip,
  decodeStressHomographyMismatchScaleForMaxAxisPx,
  decodeStressRasterSobel,
  decodeStressStripWithHomographyMismatchOffsetsPx,
  decodeStressSynthetic,
  decodeStressSyntheticWithHomographyMismatch,
} from './decodeStressHarness';
import {
  TAG36H11_CODES,
  codeToPattern,
  decodeTag36h11AnyRotation,
  decodeTag36h11Best,
  type TagPattern,
} from './tag36h11';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = join(__dirname, '..', '..');
const STRESS_PNG_OUT = join(REPO_ROOT, 'output', 'decode-stress');
const STRESS_HOMO_DIAG_PNG_OUT = join(STRESS_PNG_OUT, 'homography-3px');
const WRITE_STRESS_PNGS = process.env.WRITE_DECODE_STRESS_PNGS === '1';
const WRITE_HOMO_DIAG_PNGS = process.env.WRITE_DECODE_STRESS_HOMOGRAPHY_DIAG_PNGS === '1';

/** See file header: avoid ss=2 + FD-Sobel resonance when characterizing decode. */
const STRESS_SUPERSAMPLE = 4;

/**
 * With the default homography corner offset + speckle, these square sizes can miss **one** data cell
 * (Hamming `dist` 1) while still resolving tag id 0; larger canvases stay exact.
 */

/** Max |Δx|,|Δy| per corner (image px) for homography diagnostic dumps. */
const HOMOGRAPHY_DIAG_MAX_AXIS_PX = 3;

/** Homography diagnostic raster: no intensity speckle. */
const HOMO_DIAG_SPECKLE_AMP = 0;

function writeGreyPng(path: string, width: number, height: number, grey01: Float32Array) {
  const data = Buffer.alloc(width * height);
  for (let i = 0; i < grey01.length; i++) {
    data[i] = Math.round(Math.min(255, Math.max(0, grey01[i]! * 255)));
  }
  const png = PNG.sync.write(
    { width, height, data },
    { colorType: 0, inputColorType: 0, inputHasAlpha: false, bitDepth: 8 },
  );
  writeFileSync(path, png);
}

function writeSobelMagPng(path: string, width: number, height: number, sobel: Float32Array) {
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
  writeFileSync(path, png);
}

/** 6×6 legend: green = match truth, red = wrong, blue = weak unknown (`-1`), magenta = tie (`-2`). */
function writeCellLegendPng(
  path: string,
  decoded: TagPattern,
  truth: TagPattern,
  scale: number,
) {
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
  const png = new PNG({ width: cw, height: ch, colorType: 6 });
  png.data = rgba;
  writeFileSync(path, PNG.sync.write({ width: cw, height: ch, data: rgba }));
}

function decodeSynthetic(
  w: number,
  h: number,
  strip: [Point, Point, Point, Point],
  tagId: number,
  supersample: number,
) {
  return decodeStressSynthetic(w, h, strip, tagId, supersample, DECODE_STRESS_SPECKLE_AMP);
}

/** Count 6×6 cells where decoded disagrees with ground-truth pattern (ignores unknowns `-1`/`-2`). */
function cellErrorsVsTruth(decoded: (0 | 1 | -1 | -2)[], truth: TagPattern): number {
  let err = 0;
  for (let i = 0; i < 36; i++) {
    const d = decoded[i];
    if (d === -1 || d === -2) continue;
    if (d !== truth[i]) err++;
  }
  return err;
}

describe('decodeTagPattern stress (perspective + low resolution)', () => {
  const tagId = 0;

  it('axis-aligned tag decodes at 48×48 (baseline low-res)', () => {
    const w = 48;
    const h = 48;
    const side = 32;
    const strip = decodeStressAxisStrip(w, h, 4, side);
    const { rot, best } = decodeSynthetic(w, h, strip, tagId, STRESS_SUPERSAMPLE);
    expect(rot).not.toBeNull();
    expect(rot!.id).toBe(tagId);
    expect(best.id).toBe(tagId);
    expect(best.dist).toBe(2);
  });

  it('strong perspective at 120×120: exact pattern + dictionary', () => {
    const w = 120;
    const h = 120;
    const strip = decodeStressFitPerspectiveStrip(w, h);
    const truth = codeToPattern(TAG36H11_CODES[tagId]);
    const { rot, best, decodedPattern } = decodeSynthetic(w, h, strip, tagId, STRESS_SUPERSAMPLE);
    expect(rot).not.toBeNull();
    expect(rot!.id).toBe(tagId);
    expect(best.dist).toBeLessThanOrEqual(1);
    expect(cellErrorsVsTruth(decodedPattern, truth)).toBe(0);
  });

  it('strong perspective at 72×72: exact pattern + dictionary', () => {
    const w = 72;
    const h = 72;
    const strip = decodeStressFitPerspectiveStrip(w, h);
    const truth = codeToPattern(TAG36H11_CODES[tagId]);
    const { rot, best, decodedPattern } = decodeSynthetic(w, h, strip, tagId, STRESS_SUPERSAMPLE);
    expect(rot).not.toBeNull();
    expect(rot!.id).toBe(tagId);
    expect(best.dist).toBe(0);
    expect(cellErrorsVsTruth(decodedPattern, truth)).toBe(0);
  });

  it.skipIf(!WRITE_HOMO_DIAG_PNGS)(
    'diagnostic PNGs: ~3px max-axis homography mismatch (imperfect cases only)',
    () => {
      mkdirSync(STRESS_HOMO_DIAG_PNG_OUT, { recursive: true });
      const truthPat = codeToPattern(TAG36H11_CODES[tagId]);
      const pattern = truthPat;
      const scaleDiag = decodeStressHomographyMismatchScaleForMaxAxisPx(HOMOGRAPHY_DIAG_MAX_AXIS_PX);
      let wrote = 0;
      for (const wh of DECODE_STRESS_SIZES) {
        const rasterStrip = decodeStressFitPerspectiveStrip(wh, wh);
        const decodeStrip = decodeStressStripWithHomographyMismatchOffsetsPx(rasterStrip, scaleDiag);
        const { intensity, sobel } = decodeStressRasterSobel(
          wh,
          wh,
          rasterStrip,
          pattern,
          STRESS_SUPERSAMPLE,
          tagId,
          HOMO_DIAG_SPECKLE_AMP,
        );
        const grid = buildTagGrid(decodeStressCornersGridOrder(decodeStrip), 6);
        const decodedPattern = decodeTagPattern(grid, sobel, wh, undefined, wh);
        if (!decodedPattern) continue;
        const unknowns = decodedPattern.filter((v) => v === -1 || v === -2).length;
        const cellErr = cellErrorsVsTruth(decodedPattern, truthPat);
        const best = decodeTag36h11Best(decodedPattern, 8);
        const rot = decodeTag36h11AnyRotation(decodedPattern, 8);
        const dist = best.dist;
        if (cellErr === 0 && dist === 0 && unknowns === 0) continue;

        const tag = `h${HOMOGRAPHY_DIAG_MAX_AXIS_PX}px-w${wh}-cellErr${cellErr}-unk${unknowns}-ham${dist}-id${rot?.id ?? -1}`;
        writeGreyPng(join(STRESS_HOMO_DIAG_PNG_OUT, `${tag}-intensity.png`), wh, wh, intensity);
        writeSobelMagPng(join(STRESS_HOMO_DIAG_PNG_OUT, `${tag}-sobelMag.png`), wh, wh, sobel);
        writeCellLegendPng(
          join(STRESS_HOMO_DIAG_PNG_OUT, `${tag}-cells-rgb.png`),
          decodedPattern,
          truthPat,
          Math.max(12, Math.floor(240 / 6)),
        );
        wrote += 1;
      }
      if (wrote > 0) {
        console.warn(`Wrote ${wrote} imperfect homography-${HOMOGRAPHY_DIAG_MAX_AXIS_PX}px case(s) under ${STRESS_HOMO_DIAG_PNG_OUT}`);
      } else {
        console.warn(
          `No imperfect homography-${HOMOGRAPHY_DIAG_MAX_AXIS_PX}px cases; no PNGs written under ${STRESS_HOMO_DIAG_PNG_OUT}`,
        );
      }
    },
  );

  describe('homography mismatch (perturbed decode corners, H recomputed)', () => {
    for (const wh of DECODE_STRESS_SIZES) {
      it(`decode at ${wh}×${wh} (id 0; exact or one-cell slack)`, () => {
        const truth = codeToPattern(TAG36H11_CODES[tagId]);
        const rasterStrip = decodeStressFitPerspectiveStrip(wh, wh);
        const decodeStrip = decodeStressStripWithHomographyMismatchOffsetsPx(rasterStrip);
        const { rot, best, decodedPattern } = decodeStressSyntheticWithHomographyMismatch(
          wh,
          wh,
          rasterStrip,
          decodeStrip,
          tagId,
          STRESS_SUPERSAMPLE,
          DECODE_STRESS_SPECKLE_AMP,
        );
        expect(rot?.id).toBe(tagId);
        expect(best.dist).toBeLessThanOrEqual(2);
        expect(cellErrorsVsTruth(decodedPattern, truth)).toBeLessThanOrEqual(2);
        expect(decodedPattern.filter((v) => v === -1 || v === -2).length).toBeLessThanOrEqual(2);
      });
    }
  });

  /**
   * Snapshot table: same **strong perspective** quad scaled into square canvases.
   * `dist` = Hamming vs best codeword (known bits); `cellErr` = wrong 6×6 cells vs raster truth (excl. `-1`);
   * `unknowns` = count of `-1` (few votes) and `-2` (tie) from `classifyModuleFromPosNeg`. Rising `unknowns` / `dist` at low `wh` marks where decode starts to slip for this raster settings.
   */
  it('matches perspective + resolution characterization table', () => {
    const truth = codeToPattern(TAG36H11_CODES[tagId]);
    const sizes = [...DECODE_STRESS_SIZES];
    const table = sizes.map((wh) => {
      const strip = decodeStressFitPerspectiveStrip(wh, wh);
      const { best, rot, decodedPattern } = decodeSynthetic(wh, wh, strip, tagId, STRESS_SUPERSAMPLE);
      const unknowns = decodedPattern.filter((v) => v === -1 || v === -2).length;
      return {
        wh,
        id: rot?.id ?? -1,
        dist: best.dist,
        rotation: rot?.rotation ?? -1,
        cellErr: cellErrorsVsTruth(decodedPattern, truth),
        unknowns,
      };
    });
    if (WRITE_STRESS_PNGS) {
      mkdirSync(STRESS_PNG_OUT, { recursive: true });
      const truthPat = codeToPattern(TAG36H11_CODES[tagId]);
      for (const wh of sizes) {
        const strip = decodeStressFitPerspectiveStrip(wh, wh);
        const pattern = truthPat;
        const { intensity, sobel } = decodeStressRasterSobel(
          wh,
          wh,
          strip,
          pattern,
          STRESS_SUPERSAMPLE,
          tagId,
          DECODE_STRESS_SPECKLE_AMP,
        );
        const grid = buildTagGrid(decodeStressCornersGridOrder(strip), 6);
        const decodedPattern = decodeTagPattern(grid, sobel, wh, undefined, wh);
        if (!decodedPattern) continue;
        const unknowns = decodedPattern.filter((v) => v === -1 || v === -2).length;
        const cellErr = cellErrorsVsTruth(decodedPattern, truthPat);
        const dist = decodeTag36h11Best(decodedPattern, 8).dist;
        if (cellErr === 0 && dist === 0 && unknowns === 0) continue;

        const tag = `w${wh}-cellErr${cellErr}-unk${unknowns}-ham${dist}`;
        writeGreyPng(join(STRESS_PNG_OUT, `${tag}-intensity.png`), wh, wh, intensity);
        writeSobelMagPng(join(STRESS_PNG_OUT, `${tag}-sobelMag.png`), wh, wh, sobel);
        writeCellLegendPng(
          join(STRESS_PNG_OUT, `${tag}-cells-rgb.png`),
          decodedPattern,
          truthPat,
          Math.max(12, Math.floor(240 / 6)),
        );
      }
      console.warn(`Wrote imperfect decode cases under ${STRESS_PNG_OUT}`);
    }

    expect(table).toMatchInlineSnapshot(`
      [
        {
          "cellErr": 1,
          "dist": 1,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 200,
        },
        {
          "cellErr": 0,
          "dist": 0,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 160,
        },
        {
          "cellErr": 0,
          "dist": 0,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 120,
        },
        {
          "cellErr": 1,
          "dist": 1,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 96,
        },
        {
          "cellErr": 0,
          "dist": 0,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 80,
        },
        {
          "cellErr": 0,
          "dist": 0,
          "id": 0,
          "rotation": 0,
          "unknowns": 1,
          "wh": 72,
        },
        {
          "cellErr": 1,
          "dist": 1,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 64,
        },
        {
          "cellErr": 0,
          "dist": 0,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 56,
        },
        {
          "cellErr": 1,
          "dist": 1,
          "id": 0,
          "rotation": 0,
          "unknowns": 1,
          "wh": 48,
        },
        {
          "cellErr": 2,
          "dist": 2,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 40,
        },
      ]
    `);
  });
});
