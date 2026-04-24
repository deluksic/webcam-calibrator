// Pure: planar Zhang calibration from target layout + frame observations.

import { solveHomographyDLT, type Correspondence } from '@/lib/dltHomography'
import { type Mat3, type Point } from '@/lib/geometry'
import type { CameraIntrinsics } from '@/lib/cameraModel'
import type { CalibrationFrameObservation, TagObservation } from '@/lib/calibrationTypes'
import { reprojectionStatsPooled } from '@/lib/reprojectionError'
import type { TargetLayout } from '@/lib/targetLayout'
import { extrinsicsFromHomography, solveIntrinsicsFromHomographies, type Mat3R, type Vec3 } from '@/lib/zhangCalibration'

function cornersToPlaneList(c: TargetLayout, tag: TagObservation): { plane: { x: number; y: number }; image: Point }[] | undefined {
  const p = c.get(tag.tagId)
  if (!p) {
    return undefined
  }
  return p.map((pl, j) => ({ plane: { x: pl.x, y: pl.y }, image: tag.corners[j]! }))
}

function homographyForFrame(
  layout: TargetLayout,
  f: CalibrationFrameObservation,
): { h: Mat3; frameId: number; pairs: number } | undefined {
  const all: Correspondence[] = []
  for (const t of f.tags) {
    const part = cornersToPlaneList(layout, t)
    if (part) {
      all.push(...part)
    }
  }
  if (all.length < 4) {
    return undefined
  }
  const h = solveHomographyDLT(all)
  if (!h) {
    return undefined
  }
  return { h, frameId: f.frameId, pairs: all.length }
}

export type CalibrationOk = {
  kind: 'ok'
  K: CameraIntrinsics
  homographies: { frameId: number; h: Mat3 }[]
  extrinsics: { frameId: number; R: Mat3R; t: Vec3 }[]
  rmsPx: number
  perFrameRmsPx: Map<number, number>
}

export type CalibrationErr = { kind: 'error'; reason: 'too-few-views' | 'singular' | 'non-physical' }

export type CalibrationResult = CalibrationOk | CalibrationErr

/**
 * @param frames — frames that contain at least 2 layout tags; views with <4 correspondences are skipped for H; Zhang needs ≥3 Hs.
 */
export function solveCalibration(layout: TargetLayout, frames: readonly CalibrationFrameObservation[]): CalibrationResult {
  const hs: { frameId: number; h: Mat3 }[] = []
  for (const f of frames) {
    // count tags in layout
    const nTags = f.tags.filter((t) => layout.has(t.tagId)).length
    if (nTags < 2) {
      continue
    }
    const o = homographyForFrame(layout, f)
    if (o && o.pairs >= 4) {
      hs.push({ frameId: o.frameId, h: o.h })
    }
  }
  if (hs.length < 3) {
    return { kind: 'error', reason: 'too-few-views' }
  }
  const hList = hs.map((x) => x.h)
  const k = solveIntrinsicsFromHomographies(hList)
  if (!k) {
    return { kind: 'error', reason: 'non-physical' }
  }
  if (!Number.isFinite(k.fx) || !Number.isFinite(k.fy) || k.fx <= 0 || k.fy <= 0) {
    return { kind: 'error', reason: 'non-physical' }
  }
  const extr: { frameId: number; R: Mat3R; t: Vec3 }[] = []
  for (const h of hs) {
    const e = extrinsicsFromHomography(h.h, k)
    if (!e) {
      return { kind: 'error', reason: 'singular' }
    }
    extr.push({ frameId: h.frameId, R: e.R, t: e.t })
  }
  const stats = reprojectionStatsPooled(layout, k, hs, frames)
  return { kind: 'ok', K: k, homographies: hs, extrinsics: extr, rmsPx: stats.rmsPx, perFrameRmsPx: stats.perFrameRmsPx }
}
