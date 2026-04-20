import type { CalibrationSample } from "./calibrationTypes";

export const DEFAULT_CALIBRATION_TOP_K = 10_000;

/** Lower score evicted first; tie-break older `frameId` first. */
function compareSamples(a: CalibrationSample, b: CalibrationSample): number {
  if (a.score !== b.score) return a.score - b.score;
  if (a.frameId !== b.frameId) return a.frameId - b.frameId;
  return a.tagId - b.tagId;
}

/**
 * Merge `incoming` into `pool`, keep at most `maxK` samples by ascending score eviction.
 * Returns a **new** array (does not mutate `pool`).
 */
export function mergeCalibrationSamplesTopK(
  pool: readonly CalibrationSample[],
  incoming: readonly CalibrationSample[],
  maxK: number,
): { next: CalibrationSample[]; evicted: number } {
  const merged = [...pool, ...incoming];
  merged.sort(compareSamples);
  let evicted = 0;
  while (merged.length > maxK) {
    merged.shift();
    evicted++;
  }
  return { next: merged, evicted };
}
