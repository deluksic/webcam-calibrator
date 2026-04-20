import { buildTagGrid } from "./grid";
import type { Point } from "./geometry";
import type { Point3 } from "./calibrationTypes";

/** Unit square outer quad: TL, TR, BR, BL — same order as `buildTagGrid`. */
const UNIT_OUTER: [Point, Point, Point, Point] = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];

/**
 * 7×7 inner intersections in **tag canonical** coordinates (Z = 0 plane).
 * Matches `buildTagGrid` / detection geometry; BA assigns per-tag rigid pose.
 */
export function canonicalInnerCornersTagPlane(): Point3[] {
  const grid = buildTagGrid(UNIT_OUTER);
  return grid.innerCorners.map((p) => ({ x: p.x, y: p.y, z: 0 }));
}
