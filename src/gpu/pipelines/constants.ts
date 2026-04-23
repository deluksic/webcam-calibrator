// Camera pipeline constants and shared utilities

export const WR = 0.2126
export const WG = 0.7152
export const WB = 0.0722
export const HISTOGRAM_BINS = 256
export const HIST_WIDTH = 512
export const HIST_HEIGHT = 120
export const ALLOWED_ERROR_COUNT = 3

/**
 * Must match every `computeFn` that uses `workgroupSize: [COMPUTE_WORKGROUP_SIZE, COMPUTE_WORKGROUP_SIZE, 1]`.
 * `dispatchWorkgroups(wgX, wgY)` counts workgroups, not threads: total threads in X = wgX × this value.
 */
export const COMPUTE_WORKGROUP_SIZE = 16

/**
 * Pointer-doubling passes: L'[i]=L[L[i]] after init (see pointerJumpPipeline).
 *
 * MUST be even. After each full iteration the ping-pong index `pj` flips once,
 * so an even count guarantees `pj === 0` on exit and the compact-label pipeline
 * (whose bind groups are fixed to pointerJumpBuffer0) always reads the correct
 * converged buffer. An odd value would silently read the penultimate result.
 */
export const POINTER_JUMP_ITERATIONS = 10 // even ✓

/**
 * Gradient dot-product threshold for edge connectivity.
 * Neighbors with dot(g_i, g_j) / (|g_i||g_j|) < this value are NOT connected.
 * -1 = all neighbors connect (old behavior)
 *  0 = perpendicular is OK, opposite is blocked
 *  cos(143°) ≈ -0.8  — only near-parallel connects (was the default)
 *  cos(120°) ≈ -0.5  — allows edges up to 60° apart (good for oblique tags)
 *  cos(100°) ≈ -0.17 — allows edges up to 80° apart (very permissive)
 */
export const GRADIENT_COS_THRESHOLD = -0.5

/**
 * Edge tangent-only dilation: closes small gaps along the edge.
 * Only dilates to neighbors that are NOT strongly gradient-aligned.
 * Lower values = more restrictive, thinner dilation.
 * 0.3 allows 3x3 dilation but blocks strongly gradient-aligned neighbors.
 */
export const EDGE_DILATE_THRESHOLD = 0.3

const { ceil } = Math

/**
 * WebGPU: `dispatchWorkgroups(wgX, wgY)` so that global_invocation_id covers every pixel in width×height
 * (threads with x ≥ width or y ≥ height exit early in the shader).
 */
export function computeDispatch2d(width: number, height: number): [number, number] {
  return [ceil(width / COMPUTE_WORKGROUP_SIZE), ceil(height / COMPUTE_WORKGROUP_SIZE)]
}

// GPU note: keep `u32` for label storage / INVALID / bit hashes / atomics / bin indices.
// Prefer `i32` for coordinates, bounds checks, and (extent - 1); cast to `u32` only for nonnegative buffer indices.

/** Compute adaptive threshold from histogram */
export const THRESHOLD_PERCENTILE = 0.95

export function computeThreshold(histogramData: number[], percentile: number = THRESHOLD_PERCENTILE): number {
  const totalPixels = histogramData.reduce((a, b) => a + b, 0)
  const targetCount = totalPixels * percentile

  let cumulative = 0
  for (let i = 0; i < histogramData.length; i++) {
    cumulative += histogramData[i]!
    if (cumulative >= targetCount) {
      return i / 255.0
    }
  }
  return 0.5
}
