// 2D geometry utilities for quad fitting

export interface Point {
  x: number;
  y: number;
}

export interface Line {
  a: number;
  b: number;
  c: number; // ax + by + c = 0
}

/**
 * Compute line coefficients from two points.
 * Returns null if points are coincident.
 */
export function lineFromPoints(p1: Point, p2: Point): Line | null {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  if (Math.abs(dx) < 1e-10 && Math.abs(dy) < 1e-10) {
    return null; // coincident points
  }
  // Normalize: a² + b² = 1
  const len = Math.sqrt(dx * dx + dy * dy);
  return {
    a: -dy / len,
    b: dx / len,
    c: -(p1.x * (-dy / len) + p1.y * (dx / len)),
  };
}

/**
 * Find intersection of two lines.
 * Returns null if lines are parallel.
 */
export function lineIntersection(l1: Line, l2: Line): Point | null {
  const det = l1.a * l2.b - l1.b * l2.a;
  if (Math.abs(det) < 1e-10) {
    return null; // parallel or coincident
  }
  return {
    x: (l1.c * l2.b - l1.b * l2.c) / det,
    y: (l1.a * l2.c - l1.c * l2.a) / det,
  };
}

/**
 * Compute distance from point to line.
 */
export function pointLineDistance(p: Point, l: Line): number {
  return Math.abs(l.a * p.x + l.b * p.y + l.c);
}

/**
 * Fit a line to a set of points using least squares.
 * Returns line coefficients or null if insufficient points.
 */
export function fitLine(points: Point[]): Line | null {
  if (points.length < 2) return null;

  // Use orthogonal regression
  let sumX = 0, sumY = 0;
  for (const p of points) {
    sumX += p.x;
    sumY += p.y;
  }
  const n = points.length;
  const cx = sumX / n;
  const cy = sumY / n;

  // Covariance matrix
  let xx = 0, xy = 0, yy = 0;
  for (const p of points) {
    const dx = p.x - cx;
    const dy = p.y - cy;
    xx += dx * dx;
    xy += dx * dy;
    yy += dy * dy;
  }

  // The line direction (tangent) is the eigenvector of the covariance matrix
  // For a line, we want the direction of max variance
  const theta = 0.5 * Math.atan2(2 * xy, xx - yy);
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);

  // The line coefficients (a,b,c) need the NORMAL to the line, not the direction.
  // Direction is [cosT, sinT], so normal is [-sinT, cosT]
  const a = -sinT;
  const b = cosT;
  const c = -(a * cx + b * cy);

  return { a, b, c };
}

/**
 * Subdivide a line segment into n+1 points (including endpoints).
 * Returns array of n+1 points.
 */
export function subdivideSegment(p1: Point, p2: Point, divisions: number): Point[] {
  const points: Point[] = [];
  for (let i = 0; i <= divisions; i++) {
    const t = i / divisions;
    points.push({
      x: p1.x + t * (p2.x - p1.x),
      y: p1.y + t * (p2.y - p1.y),
    });
  }
  return points;
}

/**
 * Compute angle between two vectors (in radians).
 */
export function angleBetween(v1: Point, v2: Point): number {
  const dot = v1.x * v2.x + v1.y * v2.y;
  const mag1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
  const mag2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
  if (mag1 < 1e-10 || mag2 < 1e-10) return 0;
  return Math.acos(Math.max(-1, Math.min(1, dot / (mag1 * mag2))));
}

/**
 * Check if four points form a valid quad (roughly).
 * Returns aspect ratio of the quad.
 */
export function quadAspectRatio(corners: Point[]): number {
  if (corners.length !== 4) return 0;

  // Compute approximate side lengths
  const dists = [
    Math.hypot(corners[1].x - corners[0].x, corners[1].y - corners[0].y),
    Math.hypot(corners[2].x - corners[1].x, corners[2].y - corners[1].y),
    Math.hypot(corners[3].x - corners[2].x, corners[3].y - corners[2].y),
    Math.hypot(corners[0].x - corners[3].x, corners[0].y - corners[3].y),
  ];

  // Use max/min for aspect ratio check
  const maxD = Math.max(...dists);
  const minD = Math.min(...dists);
  return maxD / minD;
}

/**
 * Check if two line segments are roughly parallel.
 */
export function areParallel(l1: Line, l2: Line, threshold: number = 0.1): boolean {
  // Parallel if normals are aligned
  const dot = Math.abs(l1.a * l2.a + l1.b * l2.b);
  return dot > (1 - threshold);
}

/**
 * Compute corner angle at vertex (middle of 3 points).
 */
export function cornerAngle(p1: Point, vertex: Point, p2: Point): number {
  const v1 = { x: p1.x - vertex.x, y: p1.y - vertex.y };
  const v2 = { x: p2.x - vertex.x, y: p2.y - vertex.y };
  return angleBetween(v1, v2);
}