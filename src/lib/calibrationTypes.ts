/** One identified hex-target marker in a single frame, in camera pixels. */
export interface CalibrationPointObservation {
  /** Stable design-point index from the hex pattern. */
  designId: number
  x: number
  y: number
  /** Optional quality weight used by the frame-pool top-K policy. */
  score?: number
}

/** Pooled snapshot for calibration views. */
export interface CalibrationFrameObservation {
  frameId: number
  points: CalibrationPointObservation[]
  /** Raw frame luma values (0-1) for the same frame as `points`. */
  grayData?: Float32Array
  imageWidth?: number
  imageHeight?: number
}

/** Object-space target point, with `z = 0` on the target plane. */
export interface TargetObjectPoint {
  designId: number
  x: number
  y: number
  z: number
}
