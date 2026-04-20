import { applyHomography, type Point } from "../../lib/geometry";

/** TL, TR, BL, BR image corners from CPU homography8. */
export function stripCornersFromHomography8(h: Float32Array): [Point, Point, Point, Point] {
  return [
    applyHomography(h, 0, 0),
    applyHomography(h, 1, 0),
    applyHomography(h, 0, 1),
    applyHomography(h, 1, 1),
  ];
}
