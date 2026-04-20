import type { DetectedQuad } from '../gpu/contour';
import type { Point } from './geometry';

/** Stub thresholds — tune after BA exists. */
export const CALIB_MIN_MIN_R2 = 0.75;
export const CALIB_MIN_MIN_EDGE_PX = 12;
export const CALIB_MIN_AREA_PX = 400;

function quadMinEdgePx(corners: readonly Point[]): number {
  if (corners.length < 4) return 0;
  const e = (i: number, j: number) => {
    const a = corners[i]!;
    const b = corners[j]!;
    return Math.hypot(b.x - a.x, b.y - a.y);
  };
  return Math.min(e(0, 1), e(1, 2), e(2, 3), e(3, 0));
}

export function quadAreaPx(corners: readonly Point[]): number {
  if (corners.length < 4) return 0;
  const [a, b, c, d] = corners;
  return Math.abs(
    (a.x * (b.y - c.y) + b.x * (c.y - d.y) + c.x * (d.y - a.y) + d.x * (a.y - b.y)) / 2,
  );
}

export function calibrationQuadScore(q: DetectedQuad): number {
  const minR2 = q.cornerDebug?.minR2 ?? 0;
  const minEdge = quadMinEdgePx(q.corners);
  return minR2 * Math.max(minEdge, 1e-6);
}

export function acceptQuadForCalibration(q: DetectedQuad): boolean {
  if (typeof q.decodedTagId !== 'number') return false;
  if (!q.hasCorners) return false;
  if (q.cornerDebug === null || q.cornerDebug.failureCode !== 0) return false;
  if (!q.gridCells?.innerCorners || q.gridCells.innerCorners.length !== 49) return false;
  const minR2 = q.cornerDebug.minR2;
  if (minR2 < CALIB_MIN_MIN_R2) return false;
  const minEdge = quadMinEdgePx(q.corners);
  if (minEdge < CALIB_MIN_MIN_EDGE_PX) return false;
  if (quadAreaPx(q.corners) < CALIB_MIN_AREA_PX) return false;
  return true;
}

/**
 * If two+ decoded quads share the same `tagId`, reject the whole frame (likely bit flip).
 */
export function frameHasDuplicateDecodedTagIds(quads: DetectedQuad[]): boolean {
  const seen = new Set<number>();
  for (const q of quads) {
    if (typeof q.decodedTagId !== 'number') continue;
    if (seen.has(q.decodedTagId)) return true;
    seen.add(q.decodedTagId);
  }
  return false;
}
