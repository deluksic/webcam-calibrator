import type { Point } from '@/lib/geometry'

/** One decoded tag observation for BA (image space). */
export interface CalibrationSample {
  frameId: number
  tagId: number
  rotation: number
  innerCorners: Point[]
  score: number
}

export interface Point3 {
  x: number
  y: number
  z: number
}
