import { d } from 'typegpu'

import type { RoiDetectedPoint, RoiRefinedPeak } from '@/lib/roiNeighbourFilter'

export type { RoiDetectedPoint, RoiRefinedPeak }

export const ROI_PEAK_MAX_CANDIDATES = 4096
export const ROI_PEAK_MAX_CIRCLES = 4096

/** Eight 4-byte fields, so the CPU readback stride is unambiguous. */
export const ROI_REFINED_PEAK_RECORD_BYTES = 32

export const ROI_REJECT_LOW_RESPONSE = 1
export const ROI_REJECT_FIT = 2
export const ROI_REJECT_NMS = 3
export const ROI_REJECT_LOW_AREA = 4
export const ROI_REJECT_GATE = 5
export const ROI_REJECT_INDEFINITE = 6

export const RoiCircleInstance = d.struct({
  x: d.f32,
  y: d.f32,
  polarity: d.u32,
  reason: d.u32,
})
export const RoiCircleSchema = d.arrayOf(RoiCircleInstance, ROI_PEAK_MAX_CIRCLES)

export const RoiDrawIndirect = d.struct({
  vertexCount: d.u32,
  instanceCount: d.u32,
  firstVertex: d.u32,
  firstInstance: d.u32,
})

export const RoiPeakCandidateGpu = d.struct({
  x: d.u32,
  y: d.u32,
  polarity: d.u32,
})
export const RoiPeakCandidateSchema = d.arrayOf(RoiPeakCandidateGpu, ROI_PEAK_MAX_CANDIDATES)

export const RoiRefinedPeakGpu = d.struct({
  x: d.f32,
  y: d.f32,
  peak: d.f32,
  lambda1: d.f32,
  lambda2: d.f32,
  /** Physical Hessian off-diagonal term. */
  hxy: d.f32,
  polarity: d.u32,
  pad0: d.u32,
})
export const RoiRefinedPeakSchema = d.arrayOf(RoiRefinedPeakGpu, ROI_PEAK_MAX_CANDIDATES)

export type RoiRefinedPeakReadback = {
  peaks: RoiRefinedPeak[]
  count: number
  candidateCount: number
  rejectedCount: number
  bytesCopied: number
  truncated: boolean
}
