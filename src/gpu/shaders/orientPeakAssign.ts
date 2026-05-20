/**
 * Signed gradient orientation → 64-bin histogram (full 360°).
 * Peaks are canonically sorted CCW at discovery time; edges 0..3 are adjacent sides.
 */
import { d, std, tgpu } from 'typegpu'
import { atan2, cos, div, dot, length, sin } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/detectedQuad'
import {
  MAX_EDGES_PER_LABEL,
  ORIENT_ASSIGN_MAX_BIN_DIST,
  ORIENT_ASSIGN_MIN_ALIGN,
  ORIENT_HIST_BINS,
} from '@/gpu/lineFitThresholds'

const PI = Math.PI

/** Map signed gradient vector to a 0..63 bin.  θ=atan2(gy,gx) → [−π,π] → bin.
 *  Bin 0 and bin 32 are opposite directions (≈π apart), not the same bucket. */
export const gradientOrientationBin = tgpu.fn(
  [d.vec2f],
  d.u32,
)((g) => {
  'use gpu'
  const theta = atan2(g.y, g.x)
  const scaled = 0.5 * (theta / PI + d.f32(1)) * ORIENT_HIST_BINS
  let bin = d.u32(std.floor(scaled))
  if (bin >= ORIENT_HIST_BINS) {
    bin = d.u32(0)
  }
  return bin
})

/** Shortest distance along the 64-bin circle (wrap-aware). Used for peak separation and
 *  assignPeakEdgeId tie-breaks. This is NOT folding direction modulo π;
 *  bin 0 and bin 32 are far apart. */
export const circularBinDist = tgpu.fn(
  [d.u32, d.u32],
  d.u32,
)((a, b) => {
  'use gpu'
  const bins = d.u32(ORIENT_HIST_BINS)
  const forward = (a + bins - b) % bins
  const backward = (b + bins - a) % bins
  return std.min(forward, backward)
})

const orientationBinToUnit = tgpu.fn(
  [d.u32],
  d.vec2f,
)((bin) => {
  'use gpu'
  const theta = (((d.f32(bin) + 0.5) / ORIENT_HIST_BINS) * 2 - 1) * PI
  return d.vec2f(cos(theta), sin(theta))
})

const PeakBins4 = d.arrayOf(d.u32, MAX_EDGES_PER_LABEL)
const PeakDirs4 = d.arrayOf(d.vec2f, MAX_EDGES_PER_LABEL)

/** 3-bin circular centroid; falls back to bin center when total weight is zero. */
export const peakDirFromLocalBins = tgpu.fn(
  [d.u32, d.f32, d.f32, d.f32],
  d.vec2f,
)((peakBin, wPrev, wCenter, wNext) => {
  'use gpu'
  const bins = d.u32(ORIENT_HIST_BINS)
  const prevB = (peakBin + bins - d.u32(1)) % bins
  const nextB = (peakBin + d.u32(1)) % bins
  const d0 = orientationBinToUnit(prevB)
  const d1 = orientationBinToUnit(peakBin)
  const d2 = orientationBinToUnit(nextB)
  const sx = d0.x * wPrev + d1.x * wCenter + d2.x * wNext
  const sy = d0.y * wPrev + d1.y * wCenter + d2.y * wNext
  const len = length(d.vec2f(sx, sy))
  if (len > d.f32(1e-6)) {
    return d.vec2f(sx / len, sy / len)
  }
  return d1
})

/** Best peak with ĝ·peakDir ≥ minAlign; prefer in-bin, then strongest align (corners steal if dist wins first).
 *  Scratch/blips on the board can orthogonally align to the wrong peak — board defect, not assigned here. */
export const assignPeakEdgeId = tgpu.fn(
  [d.u32, PeakBins4, PeakDirs4, d.vec2f, d.u32],
  d.u32,
)((peakCount, peakBins, peakDirs, g, maxBinDist) => {
  'use gpu'
  const gLen = length(g)
  if (gLen <= d.f32(0)) {
    return COMPONENT_LABEL_INVALID
  }
  const n = div(g, gLen)
  const pixelBin = gradientOrientationBin(g)
  const minAlign = d.f32(ORIENT_ASSIGN_MIN_ALIGN)

  let edgeId = d.u32(COMPONENT_LABEL_INVALID)
  let bestInBin = false
  let bestDist = d.u32(ORIENT_HIST_BINS)
  let bestDot = d.f32(-1)

  for (const k of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    if (k < peakCount) {
      const peakBin = peakBins[k]!
      if (peakBin !== COMPONENT_LABEL_INVALID) {
        const align = dot(n, peakDirs[k]!)
        if (align >= minAlign) {
          const dist = circularBinDist(pixelBin, peakBin)
          const inBin = dist <= maxBinDist
          const pick =
            edgeId === COMPONENT_LABEL_INVALID ||
            (inBin && !bestInBin) ||
            (inBin === bestInBin && align > bestDot) ||
            (inBin === bestInBin && align === bestDot && dist < bestDist)
          if (pick) {
            bestInBin = inBin
            bestDist = dist
            bestDot = align
            edgeId = k
          }
        }
      }
    }
  }

  // Corners / foreshortened sides: gradient may miss minAlign while still belonging to one peak bin.
  if (edgeId === COMPONENT_LABEL_INVALID && peakCount > d.u32(0)) {
    let nearestDist = d.u32(ORIENT_HIST_BINS)
    for (const k of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      if (k < peakCount) {
        const peakBin = peakBins[k]!
        if (peakBin !== COMPONENT_LABEL_INVALID) {
          const dist = circularBinDist(pixelBin, peakBin)
          if (dist < nearestDist) {
            nearestDist = dist
            edgeId = k
          }
        }
      }
    }
  }

  return edgeId
})
