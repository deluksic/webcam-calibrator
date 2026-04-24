// Pure: planar Zhang calibration from labeled points + frame observations.

import { solveHomographyDLT, type Correspondence } from '@/lib/dltHomography'
import { type Mat3, type Point } from '@/lib/geometry'
import type { CameraIntrinsics } from '@/lib/cameraModel'
import type { CalibrationFrameObservation, LabeledPoint, FramePoint } from '@/lib/calibrationTypes'
import { reprojectionStatsPooled } from '@/lib/reprojectionError'
import type { TargetLayout } from '@/lib/targetLayout'
import { extrinsicsFromHomography, solveIntrinsicsFromHomographies, type Mat3R, type Vec3 } from '@/lib/zhangCalibration'

function framePointsToCorrespondences(
  framePoints: readonly FramePoint[],
  labeledPoints: readonly LabeledPoint[],
): Correspondence[] | undefined {
  const all: Correspondence[] = []
  for (const fp of framePoints) {
    const lp = labeledPoints.find((p) => p.pointId === fp.pointId)
    if (!lp) {
      continue
    }
    all.push({ plane: lp.plane, image: fp.imagePoint })
  }
  if (all.length < 8) {
    return undefined
  }
  return all
}

function homographyForFrame(
  labeledPoints: readonly LabeledPoint[],
  f: CalibrationFrameObservation,
): { h: Mat3; frameId: number; pairs: number } | undefined {
  const pairs = framePointsToCorrespondences(f.framePoints, labeledPoints)
  if (!pairs) {
    return undefined
  }
  const h = solveHomographyDLT(pairs)
  if (!h) {
    return undefined
  }
  return { h, frameId: f.frameId, pairs: pairs.length }
}

export type CalibrationOk = {
  kind: 'ok'
  K: CameraIntrinsics
  homographies: { frameId: number; h: Mat3 }[]
  extrinsics: { frameId: number; R: Mat3R; t: Vec3 }[]
  rmsPx: number
  perFrameRmsPx: Map<number, number>
}

export type CalibratedExtrinsics = ReadonlyMap<number, { R: Mat3R; t: Vec3 }>

export type CalibrationErr = { kind: 'error'; reason: 'too-few-views' | 'singular' | 'non-physical' }

export type CalibrationResult = CalibrationOk | CalibrationErr

/**
 * @param layout — used only for reprojection errors
 * @param labeledPoints — flat list of points with unique IDs, for computing homographies
 * @param frames — frames with tag observations
 */
export function solveCalibration(
  layout: TargetLayout,
  labeledPoints: LabeledPoint[],
  frames: readonly CalibrationFrameObservation[],
): CalibrationResult {
  const hs: { frameId: number; h: Mat3 }[] = []
  for (const f of frames) {
    const o = homographyForFrame(labeledPoints, f)
    if (o && o.pairs >= 8) {
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
  const stats = reprojectionStatsPooled(layout, labeledPoints, k, hs, frames)
  return { kind: 'ok', K: k, homographies: hs, extrinsics: extr, rmsPx: stats.rmsPx, perFrameRmsPx: stats.perFrameRmsPx }
}
