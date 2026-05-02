import { d } from 'typegpu'

import type { CalibrationOk } from '@/workers/calibration.worker'
import type { Corners3, Point3 } from '@/lib/calibrationTypes'
import { WORLD_AXIS_HALF_LEN } from '@/lib/orbitOrthoMath'

export function* iterCalibrationCornerPositions(ok: CalibrationOk): Generator<Point3> {
  for (const t of ok.updatedTargets) {
    for (const p of t.corners) {
      yield p
    }
  }
}

export function calibrationDefinedCornerCount(ok: CalibrationOk): number {
  let n = 0
  for (const _ of iterCalibrationCornerPositions(ok)) {
    n++
  }
  return n
}

/** Half vertical ortho extent in world/board units (~axis half-length + padded spread from board origin). */
export function orthoExtentYForPoints(ok: CalibrationOk): number {
  let maxH = WORLD_AXIS_HALF_LEN
  for (const p of iterCalibrationCornerPositions(ok)) {
    const h = Math.hypot(p.x, p.y, p.z)
    if (h > maxH) {
      maxH = h
    }
  }
  return maxH * 1.35 + WORLD_AXIS_HALF_LEN * 0.15
}

/** TL, TR, BL, BR (matches `Corners3` ordering); shared between marker and tag-quad writers. */
export function cornersToVec3fArray(corners: Corners3): [d.v3f, d.v3f, d.v3f, d.v3f] {
  return [
    d.vec3f(corners[0].x, corners[0].y, corners[0].z),
    d.vec3f(corners[1].x, corners[1].y, corners[1].z),
    d.vec3f(corners[2].x, corners[2].y, corners[2].z),
    d.vec3f(corners[3].x, corners[3].y, corners[3].z),
  ]
}
