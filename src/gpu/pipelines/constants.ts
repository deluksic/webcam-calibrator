// Camera pipeline constants and shared utilities

export const WR = 0.2126;
export const WG = 0.7152;
export const WB = 0.0722;
export const HISTOGRAM_BINS = 256;
export const HIST_WIDTH = 512;
export const HIST_HEIGHT = 120;

/**
 * Must match every `computeFn` that uses `workgroupSize: [COMPUTE_WORKGROUP_SIZE, COMPUTE_WORKGROUP_SIZE, 1]`.
 * `dispatchWorkgroups(wgX, wgY)` counts workgroups, not threads: total threads in X = wgX × this value.
 */
export const COMPUTE_WORKGROUP_SIZE = 16;

/** Pointer-doubling passes: L'[i]=L[L[i]] after init (see pointerJumpPipeline). */
export const POINTER_JUMP_ITERATIONS = 10;

/**
 * Gradient dot-product threshold for edge connectivity.
 * Neighbors with dot(g_i, g_j) / (|g_i||g_j|) < this value are NOT connected.
 * -1 = all neighbors connect (old behavior)
 *  0 = perpendicular is OK, opposite is blocked
 *  cos(140°) ≈ -0.766 — blocks gradients > 140° apart
 *  cos(160°) ≈ -0.94  — aggressive, only near-parallel connects
 */
export const GRADIENT_COS_THRESHOLD = -0.8;

/**
 * Edge tangent-only dilation: closes small gaps along the edge.
 * Only dilates to neighbors that are NOT strongly gradient-aligned.
 * Lower values = more restrictive, thinner dilation.
 * 0.3 allows 3x3 dilation but blocks strongly gradient-aligned neighbors.
 */
export const EDGE_DILATE_THRESHOLD = 0.3;

/**
 * Edges display mode: true = binary white/black (threshold footprint only); false = grayscale magnitude (looks thicker).
 */
export const EDGES_VIEW_BINARY_MASK = true;

/**
 * WebGPU: `dispatchWorkgroups(wgX, wgY)` so that global_invocation_id covers every pixel in width×height
 * (threads with x ≥ width or y ≥ height exit early in the shader).
 */
export function computeDispatch2d(width: number, height: number): [number, number] {
  return [
    Math.ceil(width / COMPUTE_WORKGROUP_SIZE),
    Math.ceil(height / COMPUTE_WORKGROUP_SIZE),
  ];
}

// GPU note: keep `u32` for label storage / INVALID / bit hashes / atomics / bin indices.
// Prefer `i32` for coordinates, bounds checks, and (extent - 1); cast to `u32` only for nonnegative buffer indices.

/** Compute adaptive threshold from histogram */
export const THRESHOLD_PERCENTILE = 0.90;

export function computeThreshold(histogramData: number[], percentile: number = THRESHOLD_PERCENTILE): number {
  const totalPixels = histogramData.reduce((a, b) => a + b, 0);
  const targetCount = totalPixels * percentile;

  let cumulative = 0;
  for (let i = 0; i < histogramData.length; i++) {
    cumulative += histogramData[i];
    if (cumulative >= targetCount) {
      return i / 255.0;
    }
  }
  return 0.5;
}
