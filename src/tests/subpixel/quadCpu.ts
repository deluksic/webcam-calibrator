import { applyHomography, type Corners } from '@/lib/geometry'

/** TL, TR, BL, BR image corners from CPU homography8. */
export function stripCornersFromHomography8(h: Float32Array): Corners {
  return [applyHomography(h, 0, 0), applyHomography(h, 1, 0), applyHomography(h, 0, 1), applyHomography(h, 1, 1)]
}
