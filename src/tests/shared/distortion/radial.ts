import type { RadialDistortionSpec } from '@/tests/shared/types'

function sampleBilinear(buf: Float32Array, width: number, height: number, x: number, y: number): number {
  const x0 = Math.floor(x)
  const y0 = Math.floor(y)
  const x1 = Math.min(width - 1, x0 + 1)
  const y1 = Math.min(height - 1, y0 + 1)
  const tx = x - x0
  const ty = y - y0
  const i00 = y0 * width + x0
  const i10 = y0 * width + x1
  const i01 = y1 * width + x0
  const i11 = y1 * width + x1
  const a = buf[i00]! * (1 - tx) + buf[i10]! * tx
  const b = buf[i01]! * (1 - tx) + buf[i11]! * tx
  return a * (1 - ty) + b * ty
}

/**
 * Mild radial distortion: inverse map from output pixel to source (undistorted) coords, then bilinear sample.
 * `k1` scales with normalized radius squared (typical small values e.g. ±0.05).
 */
export function applyRadialDistortion01(
  intensity: Float32Array,
  width: number,
  height: number,
  spec: RadialDistortionSpec,
): Float32Array {
  const cx = spec.cx ?? width * 0.5
  const cy = spec.cy ?? height * 0.5
  const ref = 0.5 * Math.min(width, height)
  const k1 = spec.k1
  const out = new Float32Array(width * height)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = x - cx
      const dy = y - cy
      const rn = Math.hypot(dx, dy) / Math.max(ref, 1e-6)
      const s = 1 + k1 * rn * rn
      const sx = cx + dx / s
      const sy = cy + dy / s
      if (sx < 0 || sx > width - 1 || sy < 0 || sy > height - 1) {
        out[y * width + x] = 1
      } else {
        out[y * width + x] = sampleBilinear(intensity, width, height, sx, sy)
      }
    }
  }
  return out
}
