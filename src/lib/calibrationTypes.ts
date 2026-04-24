import type { Corners, Point } from '@/lib/geometry'

/** One decoded tag in a single frame. */
export interface TagObservation {
  tagId: number
  rotation: number
  corners: Corners
  score: number
}

/** Point definition with unique ID (tag × corner). */
export interface LabeledPoint {
  pointId: number
  plane: { x: number; y: number }
}

/** Per-frame observation of a specific point. */
export interface FramePoint {
  pointId: number
  imagePoint: Point
}

/** All decoded tags observed in one frame. */
export interface CalibrationFrameObservation {
  frameId: number
  framePoints: readonly FramePoint[]
}

export interface Point3 {
  x: number
  y: number
  z: number
}
