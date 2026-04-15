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

/** Edge dilation threshold: minimum gradient magnitude to consider for dilation. */
export const EDGE_DILATE_THRESHOLD = 0.9;

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
export function computeThreshold(histogramData: number[], percentile: number = 0.85): number {
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
