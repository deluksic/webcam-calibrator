import type { Corners } from '@/lib/geometry'

/** One decoded tag in a single frame. */
export interface TagObservation {
  tagId: number
  rotation: number
  corners: Corners
  score: number
}

/** All decoded tags observed in one frame (used by calibration + solver). */
export interface CalibrationFrameObservation {
  frameId: number
  tags: readonly TagObservation[]
}

export interface Point3 {
  x: number
  y: number
  z: number
}
