import type { Corners, Mat3, Point } from '@/lib/geometry'
import { applyHomography, computeHomography, length } from '@/lib/geometry'
import { finiteDifferenceSobelFromIntensity } from '@/tests/utils/syntheticAprilTag'

const { min, max, floor } = Math

export interface SubpixelRefineInput {
  width: number
  height: number
  grayscale: Float32Array
  /** CPU homography: unit square → image, 8 floats + implicit h33=1. */
  homography: Mat3
  /** Optional; v1 refiner ignores (reserved for interior-aware scoring). */
  decodedTagId?: number
}

const RADIUS_PX = 5
const RADIUS_SQ = RADIUS_PX * RADIUS_PX

function stripCornersFromHomography(h: Mat3): Corners {
  return [applyHomography(h, 0, 0), applyHomography(h, 1, 0), applyHomography(h, 0, 1), applyHomography(h, 1, 1)]
}

function sobelMagnitudeAt(sobel: Float32Array, width: number, height: number, x: number, y: number): number {
  const xi = max(0, min(width - 1, floor(x)))
  const yi = max(0, min(height - 1, floor(y)))
  const o = (yi * width + xi) * 2
  const gx = sobel[o]!
  const gy = sobel[o + 1]!
  return length(gx, gy)
}

function refineCornerInDisk(sobel: Float32Array, width: number, height: number, base: Point): Point {
  let bestMag = -1
  let bestDy = 0
  let bestDx = 0
  for (let dy = -RADIUS_PX; dy <= RADIUS_PX; dy++) {
    for (let dx = -RADIUS_PX; dx <= RADIUS_PX; dx++) {
      if (dx * dx + dy * dy > RADIUS_SQ) {
        continue
      }
      const mag = sobelMagnitudeAt(sobel, width, height, base.x + dx, base.y + dy)
      if (mag > bestMag || (mag === bestMag && (dy < bestDy || (dy === bestDy && dx < bestDx)))) {
        bestMag = mag
        bestDy = dy
        bestDx = dx
      }
    }
  }
  return { x: Math.floor(base.x + bestDx) + 0.5, y: Math.floor(base.y + bestDy) + 0.5 }
}

/**
 * v1: independently refine each corner within a 5px Euclidean disk by maximizing Sobel magnitude
 * at the rounded pixel; deterministic tie-break: smallest (dy, dx) lexicographic among equal scores.
 */
export function refineSubpixelHomographyV1(input: SubpixelRefineInput): Mat3 {
  const { width, height, grayscale, homography } = input
  const sobel = finiteDifferenceSobelFromIntensity(grayscale, width, height, {
    gradientScale: 4,
  })
  const corners = stripCornersFromHomography(homography)
  const refined: Corners = [
    refineCornerInDisk(sobel, width, height, corners[0]),
    refineCornerInDisk(sobel, width, height, corners[1]),
    refineCornerInDisk(sobel, width, height, corners[2]),
    refineCornerInDisk(sobel, width, height, corners[3]),
  ]

  return computeHomography(refined)
}
