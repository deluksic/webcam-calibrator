import { d, std, tgpu } from 'typegpu'
import { atan2, cos, div, dot, length, sin } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/contour'
import {
  MAX_EDGES_PER_LABEL,
  ORIENT_ASSIGN_MAX_BIN_DIST,
  ORIENT_ASSIGN_MIN_ALIGN,
  ORIENT_HIST_BINS,
} from '@/gpu/lineFitThresholds'

export {
  MAX_EDGES_PER_LABEL,
  ORIENT_ASSIGN_MAX_BIN_DIST,
  ORIENT_ASSIGN_MIN_ALIGN,
  ORIENT_HIST_BINS,
} from '@/gpu/lineFitThresholds'

const PI = Math.PI

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

/** Unit normal from histogram bin center. */
export const orientationBinToUnit = tgpu.fn(
  [d.u32],
  d.vec2f,
)((bin) => {
  'use gpu'
  const theta = (((d.f32(bin) + 0.5) / ORIENT_HIST_BINS) * 2 - 1) * PI
  return d.vec2f(cos(theta), sin(theta))
})

export const PeakBins4 = d.arrayOf(d.u32, MAX_EDGES_PER_LABEL)
export const PeakDirs4 = d.arrayOf(d.vec2f, MAX_EDGES_PER_LABEL)

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

/** Dot of unit gradient with peak direction (−1 if |g| = 0). */
export const gradientPeakAlign = tgpu.fn(
  [d.vec2f, d.vec2f],
  d.f32,
)((peakDir, g) => {
  'use gpu'
  const gLen = length(g)
  if (gLen <= d.f32(0)) {
    return d.f32(-1)
  }
  const n = div(g, gLen)
  return dot(n, peakDir)
})

/** Best peak with ĝ·peakDir ≥ minAlign; tie-break nearest histogram bin, then align. */
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
            (inBin === bestInBin && (dist < bestDist || (dist === bestDist && align > bestDot)))
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
  return edgeId
})
