import { imagePixelToUnitSquareUv } from '@/lib/aprilTagRaycast'
/**
 * Ideal AprilTag intensity raster + finite-difference Sobel for **tests, stress harnesses, and dev scripts**.
 * Not used by live calibration / GPU decode (`grid.ts` uses `imagePixelToUnitSquareUv` from `aprilTagRaycast.ts`).
 */
import { computeHomography, type Corners } from '@/lib/geometry'
import type { TagPattern } from '@/lib/tag36h11'

const { floor, min, max } = Math

/**
 * Intensity in [0, 1] inside the tag’s unit square: **1 = white**, **0 = black**.
 * **8×8** canonical layout in UV: one **black** border cell on each side (`u`/`v` outside `(1/8, 7/8)`),
 * inner **6×6** uses `pattern` (**1 = white**, **0 = black**). Matches decode’s 8×8 module lattice.
 */
export function intensityAtTagUv(u: number, v: number, pattern: TagPattern): number {
  if (u < 0 || u > 1 || v < 0 || v > 1) {
    return 1
  }

  if (u <= 1 / 8 || u >= 7 / 8 || v <= 1 / 8 || v >= 7 / 8) {
    return 0
  }

  const uu = (u - 1 / 8) / (6 / 8)
  const vv = (v - 1 / 8) / (6 / 8)
  const col = min(5, max(0, floor(uu * 6 - 1e-9)))
  const row = min(5, max(0, floor(vv * 6 - 1e-9)))

  const v_ = pattern[row * 6 + col]
  if (v_ === 0) {
    return 0
  }
  // `-1` / `-2` (decode unknowns) render as white cell interior for synthetic views.
  return 1
}

export interface RenderAprilTagIntensityOptions {
  width: number
  height: number
  /** TL, TR, BL, BR — same as `computeHomography`. */
  corners: Corners
  pattern: TagPattern
  /**
   * Supersampling factor per axis (e.g. `4` ⇒ 4×4 = 16 stratified taps per pixel, box-averaged).
   * Reduces aliasing on cell edges under perspective. Default `1` (single center sample).
   */
  supersample?: number
}

/**
 * For each output pixel, inverse-project to (u,v) and sample the ideal 6×6 tag (piecewise constant cells).
 * Pixels outside the unit square are filled with **white** (1).
 * With `supersample` greater than 1, each pixel integrates a regular grid of subpixel samples (box filter).
 */
export function renderAprilTagIntensity(opts: RenderAprilTagIntensityOptions): Float32Array {
  const { width, height, corners, pattern } = opts
  const ss = max(1, min(16, floor(opts.supersample ?? 1)))
  const h8 = computeHomography(corners)
  const buf = new Float32Array(width * height)
  const inv = 1 / (ss * ss)

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0
      for (let sy = 0; sy < ss; sy++) {
        for (let sx = 0; sx < ss; sx++) {
          const px = x + (sx + 0.5) / ss
          const py = y + (sy + 0.5) / ss
          const { u, v, inside } = imagePixelToUnitSquareUv(h8, px, py)
          acc += inside ? intensityAtTagUv(u, v, pattern) : 1
        }
      }
      buf[y * width + x] = acc * inv
    }
  }
  return buf
}

export interface FiniteDifferenceOptions {
  /** Multiply gradients (e.g. to align magnitudes with GPU edge output); default `1`. */
  gradientScale?: number
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
  const scale = options?.gradientScale ?? 1
  const out = new Float32Array(width * height * 2)

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const xm = max(0, x - 1)
      const xp = min(width - 1, x + 1)
      const ym = max(0, y - 1)
      const yp = min(height - 1, y + 1)
      const I = (xx: number, yy: number) => intensity[yy * width + xx]!
      const gx = (I(xp, y) - I(xm, y)) * 0.5 * scale
      const gy = (I(x, yp) - I(x, ym)) * 0.5 * scale
      const o = (y * width + x) * 2
      out[o] = gx
      out[o + 1] = gy
    }
  }
  return out
}

/** Convenience: raster + Sobel in one call (for tests / benchmarks). */
export function renderAprilTagSobelFiniteDifference(
  opts: RenderAprilTagIntensityOptions,
  fdOptions?: FiniteDifferenceOptions,
): {
  intensity: Float32Array
  sobel: Float32Array
} {
  const intensity = renderAprilTagIntensity(opts)
  const sobel = finiteDifferenceSobelFromIntensity(intensity, opts.width, opts.height, fdOptions)
  return { intensity, sobel }
}
