import { d, tgpu } from 'typegpu'
import { abs, length, max, select } from 'typegpu/std'

import { LINE_FIT_PEAKDIR_COS_MIN, LINE_MIN_INLIER_RATIO } from '@/gpu/lineFitThresholds'
import { EDGE_MIN_SPAN_PX, EdgeLineEntry } from '@/gpu/pipelines/edgeLineFitPipeline'
import { PCA_ISOTROPY_MAX, tlsNormalFromMoments } from '@/gpu/shaders/linePca'

/** i32 moment scale — must match label line-fit accum. */
const POS_FIXED_SCALE = 4

/** Mirror {@link LineFitDebugCode} in lineFitDebugPipeline (avoid circular import). */
const Reject = {
  lineInvalidRatio: 10,
  lineInvalidSpan: 11,
  tlsIsotropy: 12,
  tlsPeak: 13,
  tlsDegenerate: 14,
  tlsRefineFail: 16,
} as const

export const LabelInlierStatsReadonly = d.struct({
  count: d.u32,
  sumXFixed: d.i32,
  sumYFixed: d.i32,
  sumXXFixed: d.i32,
  sumXYFixed: d.i32,
  sumYYFixed: d.i32,
  tMinFixed: d.i32,
  tMaxFixed: d.i32,
})

export const LineFitProbe = d.struct({
  nx: d.f32,
  ny: d.f32,
  nDotMean: d.f32,
})

/** Per-slot reject reason when `line.valid === 0` (post scatter). */
export const classifyInvalidLineFit = tgpu.fn(
  [LabelInlierStatsReadonly, EdgeLineEntry, d.vec2f, d.u32],
  d.u32,
)((inlier, line, peakDir, slotCount) => {
  'use gpu'
  const refinedCount = inlier.count
  const fitCount = select(slotCount, refinedCount, refinedCount > d.u32(0))
  const invPos = d.f32(1) / d.f32(POS_FIXED_SCALE)
  const sumX = d.f32(inlier.sumXFixed) * invPos
  const sumY = d.f32(inlier.sumYFixed) * invPos
  const sumXX = d.f32(inlier.sumXXFixed) * invPos * invPos
  const sumXY = d.f32(inlier.sumXYFixed) * invPos * invPos
  const sumYY = d.f32(inlier.sumYYFixed) * invPos * invPos
  const peakLen = length(peakDir)
  const cosMin = d.f32(LINE_FIT_PEAKDIR_COS_MIN)

  if (line.inlierCount > d.u32(0)) {
    const ratio = d.f32(line.inlierCount) / d.f32(max(d.u32(1), line.count))
    const trimSpan = line.tSampleMax - line.tSampleMin
    if (ratio < d.f32(LINE_MIN_INLIER_RATIO)) {
      return d.u32(Reject.lineInvalidRatio)
    }
    if (trimSpan > d.f32(0) && trimSpan < d.f32(EDGE_MIN_SPAN_PX)) {
      return d.u32(Reject.lineInvalidSpan)
    }
    return d.u32(Reject.tlsRefineFail)
  }

  if (peakLen <= d.f32(1e-6) || fitCount < d.u32(2)) {
    return d.u32(Reject.tlsDegenerate)
  }

  const refNx = peakDir.x / peakLen
  const refNy = peakDir.y / peakLen
  const tls = tlsNormalFromMoments(fitCount, sumX, sumY, sumXX, sumXY, sumYY)
  if (tls.ok === d.u32(0)) {
    if (tls.lamMax < d.f32(1e-10)) {
      return d.u32(Reject.tlsDegenerate)
    }
    if (tls.lamMin / tls.lamMax > d.f32(PCA_ISOTROPY_MAX)) {
      return d.u32(Reject.tlsIsotropy)
    }
    return d.u32(Reject.tlsDegenerate)
  }
  const dotTls = abs(tls.nx * refNx + tls.ny * refNy)
  if (dotTls < cosMin) {
    return d.u32(Reject.tlsPeak)
  }
  return d.u32(Reject.tlsDegenerate)
})

/** Normal + offset for inlier-distance tint — matches production peakDir-first fit. */
export const lineFitProbeFromSlot = tgpu.fn(
  [LabelInlierStatsReadonly, d.vec2f, d.u32],
  LineFitProbe,
)((inlier, peakDir, slotCount) => {
  'use gpu'
  const refinedCount = inlier.count
  const fitCount = select(slotCount, refinedCount, refinedCount > d.u32(0))
  const invPos = d.f32(1) / d.f32(POS_FIXED_SCALE)
  const invN = d.f32(1) / d.f32(max(d.u32(1), fitCount))
  const sumX = d.f32(inlier.sumXFixed) * invPos
  const sumY = d.f32(inlier.sumYFixed) * invPos
  const sumXX = d.f32(inlier.sumXXFixed) * invPos * invPos
  const sumXY = d.f32(inlier.sumXYFixed) * invPos * invPos
  const sumYY = d.f32(inlier.sumYYFixed) * invPos * invPos
  const cx = sumX * invN
  const cy = sumY * invN
  const peakLen = length(peakDir)

  if (peakLen > d.f32(1e-6)) {
    const refNx = peakDir.x / peakLen
    const refNy = peakDir.y / peakLen
    let nx = refNx
    let ny = refNy
    const cosMin = d.f32(LINE_FIT_PEAKDIR_COS_MIN)

    if (fitCount >= d.u32(2)) {
      const tls = tlsNormalFromMoments(fitCount, sumX, sumY, sumXX, sumXY, sumYY)
      if (tls.ok !== d.u32(0)) {
        const dotTls = tls.nx * refNx + tls.ny * refNy
        if (abs(dotTls) >= cosMin) {
          nx = tls.nx
          ny = tls.ny
          if (dotTls < d.f32(0)) {
            nx = -nx
            ny = -ny
          }
        }
      }
    }

    return LineFitProbe({
      nx,
      ny,
      nDotMean: cx * nx + cy * ny,
    })
  }

  return LineFitProbe({ nx: d.f32(0), ny: d.f32(0), nDotMean: d.f32(0) })
})
