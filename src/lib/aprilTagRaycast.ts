// Synthetic AprilTag rendering for tests: inverse homography per pixel → UV → 6×6 module lookup,
// then central finite differences → (gx, gy) in the same interleaved layout as GPU `filteredBuffer` readback.
// Live `decodeTagPattern` (grid.ts): bbox pixels → inverse H → tag UV, 8×8 τ-votes on filtered Sobel; `decodeCell` is for tests.

import { computeHomography, type Point } from './geometry';
import type { TagPattern } from './tag36h11';

/** 3×3 row-major [m0..m8] = [r0c0, r0c1, r0c2, r1c0, ...] */
function mat3FromHomography8(h: Float32Array): number[] {
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

/** Full inverse of a 3×3 row-major matrix; returns row-major 9 floats or null if singular. */
export function invertMat3RowMajor(m: number[]): Float32Array | null {
  const a = m[0],
    b = m[1],
    c = m[2];
  const d = m[3],
    e = m[4],
    f = m[5];
  const g = m[6],
    h = m[7],
    i = m[8];

  const ei_minus_fh = e * i - f * h;
  const di_minus_fg = d * i - f * g;
  const dh_minus_eg = d * h - e * g;
  const bi_minus_ch = b * i - c * h;
  const ai_minus_cg = a * i - c * g;
  const ah_minus_bg = a * h - b * g;
  const bf_minus_ce = b * f - c * e;
  const af_minus_cd = a * f - c * d;
  const ae_minus_bd = a * e - b * d;

  const det = a * ei_minus_fh - b * di_minus_fg + c * dh_minus_eg;
  if (Math.abs(det) < 1e-14) return null;

  const invDet = 1 / det;
  return new Float32Array([
    ei_minus_fh * invDet,
    -bi_minus_ch * invDet,
    bf_minus_ce * invDet,
    -di_minus_fg * invDet,
    ai_minus_cg * invDet,
    -af_minus_cd * invDet,
    dh_minus_eg * invDet,
    -ah_minus_bg * invDet,
    ae_minus_bd * invDet,
  ]);
}

/**
 * Map image pixel (x, y) to unit-square (u, v) using the inverse of `computeHomography`’s 8-parameter map.
 * Corners must be **TL, TR, BL, BR** (same order as `computeHomography`).
 */
export function imagePixelToUnitSquareUv(
  homography8: Float32Array,
  x: number,
  y: number,
): { u: number; v: number; inside: boolean } {
  const M = mat3FromHomography8(homography8);
  const inv = invertMat3RowMajor(M);
  if (!inv) return { u: 0, v: 0, inside: false };

  const xh = inv[0] * x + inv[1] * y + inv[2];
  const yh = inv[3] * x + inv[4] * y + inv[5];
  const wh = inv[6] * x + inv[7] * y + inv[8];
  if (Math.abs(wh) < 1e-12) return { u: 0, v: 0, inside: false };

  const u = xh / wh;
  const v = yh / wh;
  const inside = u >= 0 && u <= 1 && v >= 0 && v <= 1;
  return { u, v, inside };
}

/**
 * Intensity in [0, 1] inside the tag’s unit square: **1 = white**, **0 = black**.
 * **8×8** canonical layout in UV: one **black** border cell on each side (`u`/`v` outside `(1/8, 7/8)`),
 * inner **6×6** uses `pattern` (**1 = black**, **0 = white**). Matches decode’s 8×8 module lattice.
 */
export function intensityAtTagUv(u: number, v: number, pattern: TagPattern): number {
  if (u < 0 || u > 1 || v < 0 || v > 1) return 1;

  if (u <= 1 / 8 || u >= 7 / 8 || v <= 1 / 8 || v >= 7 / 8) return 0;

  const uu = (u - 1 / 8) / (6 / 8);
  const vv = (v - 1 / 8) / (6 / 8);
  const col = Math.min(5, Math.max(0, Math.floor(uu * 6 - 1e-9)));
  const row = Math.min(5, Math.max(0, Math.floor(vv * 6 - 1e-9)));

  const v_ = pattern[row * 6 + col];
  if (v_ === 1) return 0;
  return 1;
}

export interface RenderAprilTagIntensityOptions {
  width: number;
  height: number;
  /** TL, TR, BL, BR — same as `computeHomography`. */
  corners: [Point, Point, Point, Point];
  pattern: TagPattern;
  /**
   * Supersampling factor per axis (e.g. `4` ⇒ 4×4 = 16 stratified taps per pixel, box-averaged).
   * Reduces aliasing on cell edges under perspective. Default `1` (single center sample).
   */
  supersample?: number;
}

/**
 * For each output pixel, inverse-project to (u,v) and sample the ideal 6×6 tag (piecewise constant cells).
 * Pixels outside the unit square are filled with **white** (1).
 * With `supersample` greater than 1, each pixel integrates a regular grid of subpixel samples (box filter).
 */

/** Bilinear sample of row-major `intensity` at floating `(x, y)` in pixel space. */
export function sampleIntensityBilinear(
  intensity: Float32Array,
  width: number,
  height: number,
  x: number,
  y: number,
): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(width - 1, x0 + 1);
  const y1 = Math.min(height - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const I = (xx: number, yy: number) => intensity[yy * width + xx];
  const a = I(x0, y0);
  const b = I(x1, y0);
  const c = I(x0, y1);
  const d = I(x1, y1);
  return (1 - fx) * (1 - fy) * a + fx * (1 - fy) * b + (1 - fx) * fy * c + fx * fy * d;
}

export function renderAprilTagIntensity(opts: RenderAprilTagIntensityOptions): Float32Array {
  const { width, height, corners, pattern } = opts;
  const ss = Math.max(1, Math.min(16, Math.floor(opts.supersample ?? 1)));
  const h8 = computeHomography([...corners]);
  const buf = new Float32Array(width * height);
  const inv = 1 / (ss * ss);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let sy = 0; sy < ss; sy++) {
        for (let sx = 0; sx < ss; sx++) {
          const px = x + (sx + 0.5) / ss;
          const py = y + (sy + 0.5) / ss;
          const { u, v, inside } = imagePixelToUnitSquareUv(h8, px, py);
          acc += inside ? intensityAtTagUv(u, v, pattern) : 1;
        }
      }
      buf[y * width + x] = acc * inv;
    }
  }
  return buf;
}

export interface FiniteDifferenceOptions {
  /** Multiply gradients (e.g. to align magnitudes with GPU edge output); default `1`. */
  gradientScale?: number;
}

/**
 * Central differences on `intensity` (row-major, length `width * height`).
 * Output layout matches GPU readback: index `(y * width + x) * 2` → `gx`, `gy`.
 */
export function finiteDifferenceSobelFromIntensity(
  intensity: Float32Array,
  width: number,
  height: number,
  options?: FiniteDifferenceOptions,
): Float32Array {
  const scale = options?.gradientScale ?? 1;
  const out = new Float32Array(width * height * 2);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const xm = Math.max(0, x - 1);
      const xp = Math.min(width - 1, x + 1);
      const ym = Math.max(0, y - 1);
      const yp = Math.min(height - 1, y + 1);
      const I = (xx: number, yy: number) => intensity[yy * width + xx];
      const gx = (I(xp, y) - I(xm, y)) * 0.5 * scale;
      const gy = (I(x, yp) - I(x, ym)) * 0.5 * scale;
      const o = (y * width + x) * 2;
      out[o] = gx;
      out[o + 1] = gy;
    }
  }
  return out;
}

/** Convenience: raster + Sobel in one call (for tests / future benchmarks). */
export function renderAprilTagSobelFiniteDifference(
  opts: RenderAprilTagIntensityOptions,
  fdOptions?: FiniteDifferenceOptions,
): {
  intensity: Float32Array;
  sobel: Float32Array;
} {
  const intensity = renderAprilTagIntensity(opts);
  const sobel = finiteDifferenceSobelFromIntensity(intensity, opts.width, opts.height, fdOptions);
  return { intensity, sobel };
}
