import type { Corners } from '@/lib/geometry'

/** One decoded tag in a single frame. */
export interface TagObservation {
  tagId: number
  rotation: number
  corners: Corners
  score: number
}

/** Canonical corner slot in TL, TR, BL, BR order (`Corners` indexing). */
export type CalibrationCornerId = 0 | 1 | 2 | 3

/** AprilTag quad in **image space** (`u,v` pixels from detection). Same corner order as `Corners`. */
export interface ImageTag {
  tagId: number
  corners: Corners
}

export interface Point3 {
  x: number
  y: number
  z: number
}

/** Object-space quad in triangle-strip order: TL, TR, BL, BR (`Corners` / `ImageTag` indexing). */
export type Corners3 = [tl: Point3, tr: Point3, bl: Point3, br: Point3]

/**
 * Tags on the board in object space. Used as **`objectTags`** input to calibration (planar prior **`z = 0`**
 * from the current layout). **`CalibrationOk.updatedTargets`** uses the same shape but lists only tags that
 * were **visible in every** pooled snapshot used by the solver (same set as the intersected object points);
 * **`corners`** are the bundle-adjusted quad.
 */
export interface ObjectTag {
  tagId: number
  corners: Corners3
}

/** Pooled snapshots for calibration views. */
export interface CalibrationFrameObservation {
  frameId: number
  tags: ImageTag[]
}
