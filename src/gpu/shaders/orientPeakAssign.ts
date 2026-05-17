import { d, std, tgpu } from 'typegpu'
import { atan2, cos, div, dot, length, sin } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/contour'

export const ORIENT_HIST_BINS = 32
export const MAX_EDGES_PER_LABEL = 6
export const ORIENT_ASSIGN_COS_THRESHOLD = 0.8

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

/** Peak bins as fixed array (invalid = COMPONENT_LABEL_INVALID). */
export const PeakBins6 = d.arrayOf(d.u32, MAX_EDGES_PER_LABEL)
export const PeakDirs6 = d.arrayOf(d.vec2f, MAX_EDGES_PER_LABEL)

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

/** Dot of unit gradient with refined peak direction (−1 if |g| = 0). */
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

/** Best-matching peak by gradient alignment (no 0.8 gate — gate pixels in line-fit passes). */
export const assignPeakEdgeId = tgpu.fn(
  [d.u32, PeakBins6, PeakDirs6, d.vec2f],
  d.u32,
)((peakCount, peakBins, peakDirs, g) => {
  'use gpu'
  const gLen = length(g)
  if (gLen <= d.f32(0)) {
    return d.u32(COMPONENT_LABEL_INVALID)
  }
  const n = div(g, gLen)
  const pixelBin = gradientOrientationBin(g)

  let edgeId = d.u32(COMPONENT_LABEL_INVALID)
  let bestDot = d.f32(-1)
  let bestDist = d.u32(ORIENT_HIST_BINS)
  for (const k of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
    if (k < peakCount) {
      const peakBin = peakBins[d.u32(k)]!
      if (peakBin !== d.u32(COMPONENT_LABEL_INVALID)) {
        const peakDir = peakDirs[d.u32(k)]!
        const align = dot(n, peakDir)
        const dist = circularBinDist(pixelBin, peakBin)
        const pick = align > bestDot || (align === bestDot && dist < bestDist)
        if (pick) {
          bestDot = align
          bestDist = dist
          edgeId = d.u32(k)
        }
      }
    }
  }
  return edgeId
})
