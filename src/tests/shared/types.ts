export type SceneKind = 'perspective' | 'axis'

export interface SceneSpec {
  kind: SceneKind
  width: number
  height: number
  /** Tag id for pattern selection. */
  tagId: number
  supersample: number
  margin?: number
  /** For `axis`: square side in px. For `perspective`: ignored. */
  axisSidePx?: number
  /** Scale quad so longest edge ≤ this (strict &lt; 20 for small-tag studies). */
  maxEdgePx?: number
  /** Rotate quad around centroid (radians). Default 0. */
  rotationRad?: number
  perspectiveBoost?: number
}

export type NoiseOp =
  | { type: 'speckle'; amplitude: number; seed: number }
  | { type: 'gaussian'; sigma: number; seed: number }
  | { type: 'saltPepper'; rate: number; seed: number }

export interface RadialDistortionSpec {
  /** Brown–Conrady k1 (barrel negative, pincushion positive), applied in normalized image coords. */
  k1: number
  /** Center in pixels; default image center. */
  cx?: number
  cy?: number
}

export interface BuildImageOptions {
  scene: SceneSpec
  noise?: NoiseOp[]
  /** Applied after raster, before noise (in-place remap). */
  radialDistortion?: RadialDistortionSpec
}

import type { Corners } from '@/lib/geometry'

export interface RasterPack {
  width: number
  height: number
  groundTruthStrip: Corners
  grayscale: Float32Array
  sobel: Float32Array
}
