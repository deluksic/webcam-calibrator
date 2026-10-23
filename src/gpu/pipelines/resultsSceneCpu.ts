import type { CalibrationOk } from '@/workers/calibration.worker'
import { WORLD_AXIS_HALF_LEN } from '@/lib/orbitOrthoMath'

export function* iterCalibrationPointPositions(
  ok: CalibrationOk,
): Generator<{ x: number; y: number; z: number }> {
  for (const point of ok.updatedTargetPoints) {
    yield point
  }
}

export function calibrationPointCount(ok: CalibrationOk): number {
  return ok.updatedTargetPoints.length
}

/** Half vertical ortho extent in world/board units (~axis half-length + padded spread from board origin). */
export function orthoExtentYForPoints(ok: CalibrationOk): number {
  let maxDistance = WORLD_AXIS_HALF_LEN
  for (const point of iterCalibrationPointPositions(ok)) {
    const distance = Math.hypot(point.x, point.y, point.z)
    if (distance > maxDistance) {
      maxDistance = distance
    }
  }
  return maxDistance * 1.35 + WORLD_AXIS_HALF_LEN * 0.15
}
