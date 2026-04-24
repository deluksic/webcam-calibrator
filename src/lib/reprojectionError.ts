// Reproject board plane (X,Y, z=0) with p_c = R * [x,y,0] + t; pinhole: u = fx*Xc/zc + cx, v = fy*Yc/zc + cy.

import { length, type Mat3, type Point } from '@/lib/geometry'
import type { CameraIntrinsics } from '@/lib/cameraModel'
import type { CalibrationFrameObservation, LabeledPoint } from '@/lib/calibrationTypes'
import type { TargetLayout } from '@/lib/targetLayout'
import type { Vec3 } from '@/lib/zhangCalibration'
import { extrinsicsFromHomography, type Mat3R } from '@/lib/zhangCalibration'
import { rotateRing } from '@/lib/corners'

const { sqrt } = Math

function abs(x: number) {
  return x < 0 ? -x : x
}

export function projectPlanePoint(k: CameraIntrinsics, R: Mat3R, t: Vec3, x: number, y: number): Point {
  const Xc = R[0]! * x + R[1]! * y + t.x
  const Yc = R[3]! * x + R[4]! * y + t.y
  const zc = R[6]! * x + R[7]! * y + t.z
  if (abs(zc) < 1e-15) {
    return { x: 0, y: 0 }
  }
  return { x: (k.fx * Xc) / zc + k.cx, y: (k.fy * Yc) / zc + k.cy }
}

function cornerError(pred: Point, im: Point): number {
  return length(pred.x - im.x, pred.y - im.y)
}

export function reprojectionRmsForFrame(
  layout: TargetLayout,
  labeledPoints: readonly LabeledPoint[],
  k: CameraIntrinsics,
  R: Mat3R,
  t: Vec3,
  f: CalibrationFrameObservation,
): { rms: number; n: number } {
  const sq: number[] = []
  for (const fp of f.framePoints) {
    const lp = labeledPoints.find((p) => p.pointId === fp.pointId)
    if (!lp) {
      continue
    }
    const err = cornerError(projectPlanePoint(k, R, t, lp.plane.x, lp.plane.y), fp.imagePoint)
    sq.push(err * err)
  }
  if (sq.length === 0) {
    return { rms: 0, n: 0 }
  }
  return { rms: sqrt(sq.reduce((a, b) => a + b, 0) / sq.length), n: sq.length }
}

/**
 * Pooled RMSE and per-frame map from (layout, K, H per frame, observations).
 */
export function reprojectionStatsPooled(
  layout: TargetLayout,
  labeledPoints: readonly LabeledPoint[],
  k: CameraIntrinsics,
  hs: readonly { frameId: number; h: Mat3 }[],
  frameObs: readonly CalibrationFrameObservation[],
): { rmsPx: number; perFrameRmsPx: Map<number, number> } {
  const perFrameRmsPx = new Map<number, number>()
  const allSq: number[] = []
  for (const fo of frameObs) {
    const hEnt = hs.find((x) => x.frameId === fo.frameId)
    if (!hEnt) {
      continue
    }
    const ex = extrinsicsFromHomography(hEnt.h, k)
    if (!ex) {
      continue
    }
    const { rms, n } = reprojectionRmsForFrame(layout, labeledPoints, k, ex.R, ex.t, fo)
    if (n > 0) {
      perFrameRmsPx.set(fo.frameId, rms)
    }
    for (const fp of fo.framePoints) {
      const lp = labeledPoints.find((p) => p.pointId === fp.pointId)
      if (!lp) {
        continue
      }
      const p = projectPlanePoint(k, ex.R, ex.t, lp.plane.x, lp.plane.y)
      const e = p.x - fp.imagePoint.x
      const f_ = p.y - fp.imagePoint.y
      allSq.push(e * e + f_ * f_)
    }
  }
  const rmsPx = allSq.length > 0 ? sqrt(allSq.reduce((a, b) => a + b, 0) / allSq.length) : 0
  return { rmsPx, perFrameRmsPx }
}
