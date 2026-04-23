import { applyHomography, type Corners, type Mat3 } from '@/lib/geometry'

/** TL, TR, BL, BR image corners from CPU homography. */
export function stripCornersFromHomography8(h: Mat3): Corners {
  return [applyHomography(h, 0, 0), applyHomography(h, 1, 0), applyHomography(h, 0, 1), applyHomography(h, 1, 1)]
}
