/** One decoded tag observation for BA (image space). */
export interface CalibrationSample {
  frameId: number
  tagId: number
  rotation: number
  score: number
}

export interface Point3 {
  x: number
  y: number
  z: number
}
