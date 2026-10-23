import type { CalibrationPointObservation } from '@/lib/calibrationTypes'
import type { HexTargetMatch } from '@/lib/hexTarget/match'
import type { HexTargetModel } from '@/lib/hexTarget/target'
import {
  IDENTITY_RECOVERED,
  IDENTITY_RESOLVED,
  type MatchedIdentityPoint,
} from '@/lib/pointIdentity/match'

export type CalibrationPointConversion =
  | { points: CalibrationPointObservation[]; rejectedReason?: undefined }
  | { points: []; rejectedReason: 'no-accepted-points' | 'duplicate-design-id' }

/** Recovered identities farther than this from the predicted design position are not measurements. */
const RECOVERED_DESIGN_GATE_FACTOR = 0.5

function isAccepted(point: MatchedIdentityPoint, spacing: number): boolean {
  if (typeof point.designId !== 'number') {
    return false
  }
  if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) {
    return false
  }
  if (point.status === IDENTITY_RESOLVED) {
    return true
  }
  if (point.status !== IDENTITY_RECOVERED) {
    return false
  }
  if (point.geometryConflict === true) {
    return false
  }
  return point.designDistance === undefined || point.designDistance <= spacing * RECOVERED_DESIGN_GATE_FACTOR
}

/**
 * Turn an identity match into calibration measurements. A frame that assigns
 * the same design id twice is rejected rather than silently picking one.
 */
export function toCalibrationFramePoints(
  match: HexTargetMatch,
  design: HexTargetModel,
): CalibrationPointConversion {
  const seen = new Set<number>()
  const points: CalibrationPointObservation[] = []
  for (const point of match.points) {
    if (!isAccepted(point, design.spacing)) {
      continue
    }
    const designId = point.designId!
    if (seen.has(designId)) {
      return { points: [], rejectedReason: 'duplicate-design-id' }
    }
    seen.add(designId)
    points.push({
      designId,
      x: point.x,
      y: point.y,
      score: point.status === IDENTITY_RESOLVED ? 1 : 0.8,
    })
  }
  if (points.length === 0) {
    return { points: [], rejectedReason: 'no-accepted-points' }
  }
  return { points }
}
