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
  // Lines are a x + b y + c = 0 (same as a x + b y = −c). Cramer gives:
  // x = (b1*c2 − c1*b2) / det,  y = (c1*a2 − a1*c2) / det
  return {
    x: (l1.b * l2.c - l1.c * l2.b) / det,
    y: (l1.c * l2.a - l1.a * l2.c) / det,
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
  let sumX = 0,
    sumY = 0;
  for (const p of points) {
    sumX += p.x;
    sumY += p.y;
  }
  const n = points.length;
  const cx = sumX / n;
  const cy = sumY / n;

  // Covariance matrix
  let xx = 0,
    xy = 0,
    yy = 0;
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
  return dot > 1 - threshold;
}

/**
 * Compute corner angle at vertex (middle of 3 points).
 */
export function cornerAngle(p1: Point, vertex: Point, p2: Point): number {
  const v1 = { x: p1.x - vertex.x, y: p1.y - vertex.y };
  const v2 = { x: p2.x - vertex.x, y: p2.y - vertex.y };
  return angleBetween(v1, v2);
}

/**
 * Compute a 3x3 homography H that maps the unit square corners to 4 source points.
 *
 * Source points (dst in unit square → src in image space):
 *   (0,0) → src[0] (top-left)
 *   (1,0) → src[1] (top-right)
 *   (0,1) → src[2] (bottom-left)
 *   (1,1) → src[3] (bottom-right)
 *
 * The resulting H has h33=1 and maps (u,v) → (x,y) as:
 *   w = h7*u + h8*v + 1
 *   x = (h1*u + h2*v + h3) / w
 *   y = (h4*u + h5*v + h6) / w
 *
 * On success: 8 values `[h1, h2, h3, h4, h5, h6, h7, h8]`. Returns `null` if the 8×8 system is
 * singular (degenerate / collinear corners, or wrong point count).
 */
export function tryComputeHomography(src: Point[]): Float32Array | null {
  if (src.length !== 4) {
    return null;
  }

  // Corner order: TL(0), TR(1), BL(2), BR(3)
  // Destination (unit square): (0,0), (1,0), (0,1), (1,1)
  const dst: [number, number][] = [
    [0, 0], // TL
    [1, 0], // TR
    [0, 1], // BL
    [1, 1], // BR
  ];

  // Build the 8x8 matrix for the linear system Ah = b
  // For each correspondence (ui, vi) → (xi, yi):
  //   [ui, vi, 1, 0, 0, 0, -ui*xi, -vi*xi] [h1 h2 h3 h4 h5 h6 h7 h8]^T = xi
  //   [0, 0, 0, ui, vi, 1, -ui*yi, -vi*yi] [h1 h2 h3 h4 h5 h6 h7 h8]^T = yi
  const N = 8;
  const A: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
  const b: number[] = new Array(N).fill(0);

  for (let i = 0; i < 4; i++) {
    const ui = dst[i][0];
    const vi = dst[i][1];
    const xi = src[i].x;
    const yi = src[i].y;

    // Row 2i: xi equation
    A[2 * i][0] = ui;
    A[2 * i][1] = vi;
    A[2 * i][2] = 1;
    A[2 * i][6] = -ui * xi;
    A[2 * i][7] = -vi * xi;
    b[2 * i] = xi;

    // Row 2i+1: yi equation
    A[2 * i + 1][3] = ui;
    A[2 * i + 1][4] = vi;
    A[2 * i + 1][5] = 1;
    A[2 * i + 1][6] = -ui * yi;
    A[2 * i + 1][7] = -vi * yi;
    b[2 * i + 1] = yi;
  }

  // Gaussian elimination with partial pivoting
  const h = new Array(N).fill(0);

  for (let col = 0; col < N; col++) {
    // Find pivot
    let maxRow = col;
    for (let row = col + 1; row < N; row++) {
      if (Math.abs(A[row][col]) > Math.abs(A[maxRow][col])) {
        maxRow = row;
      }
    }

    // Swap rows
    [A[col], A[maxRow]] = [A[maxRow], A[col]];
    [b[col], b[maxRow]] = [b[maxRow], b[col]];

    const pivot = A[col][col];
    if (Math.abs(pivot) < 1e-10) {
      return null;
    }

    // Normalize pivot row
    for (let j = col; j < N; j++) {
      A[col][j] /= pivot;
    }
    b[col] /= pivot;

    // Eliminate
    for (let row = 0; row < N; row++) {
      if (row !== col) {
        const factor = A[row][col];
        if (factor !== 0) {
          for (let j = col; j < N; j++) {
            A[row][j] -= factor * A[col][j];
          }
          b[row] -= factor * b[col];
        }
      }
    }
  }

  for (let i = 0; i < N; i++) {
    h[i] = b[i];
  }

  return new Float32Array(h);
}

/**
 * Same as {@link tryComputeHomography}, but throws if the quad does not admit a unique homography.
 */
export function computeHomography(src: Point[]): Float32Array {
  if (src.length !== 4) {
    throw new Error("computeHomography requires exactly 4 source points");
  }
  const h = tryComputeHomography(src);
  if (!h) {
    throw new Error("Singular homography matrix");
  }
  return h;
}

/**
 * Apply homography to a unit-square point (u, v).
 * Returns {x, y} in image coordinates.
 */
export function applyHomography(h: Float32Array, u: number, v: number): Point {
  const h1 = h[0],
    h2 = h[1],
    h3 = h[2];
  const h4 = h[3],
    h5 = h[4],
    h6 = h[5];
  const h7 = h[6],
    h8 = h[7];

  const w = h7 * u + h8 * v + 1;
  const x = (h1 * u + h2 * v + h3) / w;
  const y = (h4 * u + h5 * v + h6) / w;
  return { x, y };
}

/**
 * Compute projective weights for perspective-correct quad rendering.
 * Maps unit square corners to detected quad via homography.
 * Corner order: TL, TR, BL, BR
 * Returns [w0, w1, w2, w3] — weights for TL, TR, BL, BR
 *
 * For homography H: (x,y) = (h1*u + h2*v + h3) / w, where w = h7*u + h8*v + 1
 * Weights at corners: w0=1, w1=h7+1, w2=h8+1, w3=h7+h8+1
 */
export function computeProjectiveWeights(corners: Point[]): [number, number, number, number] {
  if (corners.length !== 4) {
    throw new Error("computeProjectiveWeights requires exactly 4 points");
  }
  const x0 = corners[0].x,
    y0 = corners[0].y; // TL
  const x1 = corners[1].x,
    y1 = corners[1].y; // TR
  const x2 = corners[2].x,
    y2 = corners[2].y; // BL
  const x3 = corners[3].x,
    y3 = corners[3].y; // BR

  // Solve for h7, h8 from corner correspondences
  // Using: (x3-x1)*h7 + (x3-x2)*h8 = x1 + x2 - x0 - x3
  //        (y3-y1)*h7 + (y3-y2)*h8 = y1 + y2 - y0 - y3
  const A = x3 - x1;
  const B = x3 - x2;
  const C = x1 + x2 - x0 - x3;
  const D = y3 - y1;
  const E = y3 - y2;
  const F = y1 + y2 - y0 - y3;

  const det = A * E - B * D;

  if (Math.abs(det) < 1e-10) {
    // Degenerate case - parallelogram or axis-aligned
    return [1, 1, 1, 1];
  }

  const h7 = (C * E - B * F) / det;
  const h8 = (A * F - C * D) / det;

  // Weights at unit square corners
  const w0 = 1;
  const w1 = h7 + 1;
  const w2 = h8 + 1;
  const w3 = h7 + h8 + 1;

  return [w0, w1, w2, w3];
}

/**
 * Verify projective weights: blend corners by weights and check unit square maps.
 * Returns blended corner positions — should equal [0,0],[1,0],[0,1],[1,1].
 */
export function verifyProjectiveWeights(corners: Point[]): {
  blended: Point[];
  errors: number[];
} {
  const [w0, w1, w2, w3] = computeProjectiveWeights(corners);

  // At unit square corners, bilinear blend should give unit positions
  const expected = [
    { x: 0, y: 0 }, // TL → (0,0)
    { x: 1, y: 0 }, // TR → (1,0)
    { x: 0, y: 1 }, // BL → (0,1)
    { x: 1, y: 1 }, // BR → (1,1)
  ];

  // Compute what corners would be if the unit square maps through these weights
  const wSum = w0 + w1 + w2 + w3;

  // At u=0,v=0: w = w0/wSum → blended = (0 * w0 + 0 * w2 + 0 * w1 + 0 * w3) / wSum = 0 → OK
  // At u=1,v=0: w = w1/wSum → blended = (1 * w1 + 0 * w0 + ... ) / wSum = w1/wSum
  // We want blended = 1 at TR, 0 at TL/BL, and w1/wSum to normalize to 1...

  const blended = [
    { x: corners[0].x / w0, y: corners[0].y / w0 },
    { x: corners[1].x / w1, y: corners[1].y / w1 },
    { x: corners[2].x / w2, y: corners[2].y / w2 },
    { x: corners[3].x / w3, y: corners[3].y / w3 },
  ];

  const errors = blended.map((b, i) => {
    const ex = expected[i].x - b.x;
    const ey = expected[i].y - b.y;
    return Math.sqrt(ex * ex + ey * ey);
  });

  return { blended, errors };
}
