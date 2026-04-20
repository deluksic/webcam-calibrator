import { applyHomography, computeHomography, type Point } from '@/lib/geometry'
import { finiteDifferenceSobelFromIntensity } from '@/tests/utils/syntheticAprilTag'

export interface SubpixelRefineInput {
  width: number
  height: number
  grayscale: Float32Array
  /** CPU homography: unit square → image, 8 floats + implicit h33=1. */
  homography8: Float32Array
  /** Optional; v1 refiner ignores (reserved for interior-aware scoring). */
  decodedTagId?: number
}

const RADIUS_PX = 5
const RADIUS_SQ = RADIUS_PX * RADIUS_PX

function stripCornersFromHomography(h: Float32Array): [Point, Point, Point, Point] {
  return [applyHomography(h, 0, 0), applyHomography(h, 1, 0), applyHomography(h, 0, 1), applyHomography(h, 1, 1)]
}

function sobelMagnitudeAt(sobel: Float32Array, width: number, height: number, x: number, y: number): number {
  const xi = Math.max(0, Math.min(width - 1, Math.round(x)))
  const yi = Math.max(0, Math.min(height - 1, Math.round(y)))
  const o = (yi * width + xi) * 2
  const gx = sobel[o]!
  const gy = sobel[o + 1]!
  return Math.hypot(gx, gy)
}

/**
 * v1: independently refine each corner within a 5px Euclidean disk by maximizing Sobel magnitude
 * at the rounded pixel; deterministic tie-break: smallest (dy, dx) lexicographic among equal scores.
 */
export function refineSubpixelHomographyV1(input: SubpixelRefineInput): Float32Array {
  const { width, height, grayscale, homography8 } = input
  const sobel = finiteDifferenceSobelFromIntensity(grayscale, width, height, {
    gradientScale: 4,
  })
  const corners = stripCornersFromHomography(homography8)
  const refined: Point[] = []

  for (let ci = 0; ci < 4; ci++) {
    const base = corners[ci]!
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
    refined.push({ x: base.x + bestDx, y: base.y + bestDy })
  }

  return computeHomography(refined)
}
