/**
 * Threshold selection for the ROI-lab blob segmentation.
 *
 * The ROI mask covers the gray target. The pipeline first removes the
 * low-frequency illumination field, so the masked flattened-luma histogram is
 * dominated by one background peak roughly in the middle of the range.
 *
 *   1. Find that peak.
 *   2. Treat a small fraction of its height as the noise floor.
 *   3. Walk outward from the peak until the histogram reaches that floor; the
 *      crossing on each side is the black/white cut.
 *
 * Marker populations can be a few percent of the gray peak, so the floor is
 * relative to the peak rather than an absolute count.
 */

export const ROI_HISTOGRAM_BINS = 256

/** Inclusive luma bin index at which a pixel stops being "black". */
export type RoiHistogramAnalysis = {
  /** True when at least one usable black/white cut was found. */
  valid: boolean
  /** Bin index (0..255) of the dominant masked flattened luma, i.e. the target gray. */
  backgroundBin: number
  /** Bins below this value are black blobs. `0` disables the black class. */
  blackCut: number
  /** Bins at or above this value are white blobs. `256` disables the white class. */
  whiteCut: number
  maskedPixels: number
  blackPixels: number
  grayPixels: number
  whitePixels: number
}

export type RoiHistogramOptions = {
  /** Below this many masked pixels the frame is not analyzed. */
  minPixels?: number
  /** A blob class may not own more than this fraction of the masked pixels. */
  maxBlobFraction?: number
  /** The target gray class must own at least this fraction of masked pixels. */
  minBackgroundFraction?: number
  /** A blob class must contain at least this fraction of the masked pixels. */
  minBlobFraction?: number
  /** Minimum mean-luma distance between a blob class and the gray mode. */
  minContrast?: number
  /**
   * Fraction of the gray peak that counts as the noise floor. Lower values
   * move both cuts away from gray and make the blob classes stricter.
   */
  noiseFloorFraction?: number
}

const DEFAULTS: Required<RoiHistogramOptions> = {
  minPixels: 256,
  maxBlobFraction: 0.35,
  minBackgroundFraction: 0.5,
  minBlobFraction: 0.001,
  minContrast: 18,
  noiseFloorFraction: 0.019,
}

const SMOOTH_KERNEL = [1, 4, 6, 4, 1] as const

function smoothHistogram(counts: ArrayLike<number>): Float64Array {
  const smoothed = new Float64Array(ROI_HISTOGRAM_BINS)
  const weightSum = SMOOTH_KERNEL.reduce((sum, weight) => sum + weight, 0)
  for (let bin = 0; bin < ROI_HISTOGRAM_BINS; bin++) {
    let sum = 0
    for (let k = 0; k < SMOOTH_KERNEL.length; k++) {
      const source = bin + k - 2
      if (source < 0 || source >= ROI_HISTOGRAM_BINS) {
        continue
      }
      sum += (counts[source] ?? 0) * SMOOTH_KERNEL[k]!
    }
    smoothed[bin] = sum / weightSum
  }
  return smoothed
}

function sumBins(counts: ArrayLike<number>, start: number, endExclusive: number): number {
  let sum = 0
  for (let bin = start; bin < endExclusive; bin++) {
    sum += Math.max(0, counts[bin] ?? 0)
  }
  return sum
}

function meanBin(counts: ArrayLike<number>, start: number, endExclusive: number): number {
  let sum = 0
  let weighted = 0
  for (let bin = start; bin < endExclusive; bin++) {
    const value = Math.max(0, counts[bin] ?? 0)
    sum += value
    weighted += value * bin
  }
  return sum > 0 ? weighted / sum : 0
}

/**
 * Analyze a masked flattened-luma histogram and return black/white bins for the shader.
 *
 * A cut of `0` (black) or `256` (white) means that class is disabled, so an
 * invalid split does not paint half the gray target as blobs.
 */
export function analyzeRoiHistogram(
  counts: ArrayLike<number>,
  options: RoiHistogramOptions = {},
): RoiHistogramAnalysis {
  const config = { ...DEFAULTS, ...options }
  let maskedPixels = 0
  for (let bin = 0; bin < ROI_HISTOGRAM_BINS; bin++) {
    maskedPixels += Math.max(0, counts[bin] ?? 0)
  }

  const disabled: RoiHistogramAnalysis = {
    valid: false,
    backgroundBin: 0,
    blackCut: 0,
    whiteCut: ROI_HISTOGRAM_BINS,
    maskedPixels,
    blackPixels: 0,
    grayPixels: maskedPixels,
    whitePixels: 0,
  }
  if (maskedPixels < config.minPixels) {
    return disabled
  }

  const smoothed = smoothHistogram(counts)
  let backgroundBin = 0
  let backgroundValue = -1
  for (let bin = 0; bin < ROI_HISTOGRAM_BINS; bin++) {
    if (smoothed[bin]! > backgroundValue) {
      backgroundValue = smoothed[bin]!
      backgroundBin = bin
    }
  }

  // The gray peak is the anchor. Both cuts are where the histogram falls to
  // the noise floor on that side.
  const noiseFloor = backgroundValue * config.noiseFloorFraction
  let blackCut = 0
  for (let bin = backgroundBin - 1; bin >= 0; bin--) {
    if (smoothed[bin]! <= noiseFloor) {
      blackCut = bin + 1
      break
    }
  }
  let whiteCut = ROI_HISTOGRAM_BINS
  for (let bin = backgroundBin + 1; bin < ROI_HISTOGRAM_BINS; bin++) {
    if (smoothed[bin]! <= noiseFloor) {
      whiteCut = bin
      break
    }
  }

  const blackPixels = sumBins(counts, 0, blackCut)
  const whitePixels = sumBins(counts, whiteCut, ROI_HISTOGRAM_BINS)
  const blackMean = meanBin(counts, 0, blackCut)
  const whiteMean = meanBin(counts, whiteCut, ROI_HISTOGRAM_BINS)
  const blackFraction = blackPixels / maskedPixels
  const whiteFraction = whitePixels / maskedPixels

  const blackOk =
    blackFraction >= config.minBlobFraction &&
    blackFraction <= config.maxBlobFraction &&
    backgroundBin - blackMean >= config.minContrast
  const whiteOk =
    whiteFraction >= config.minBlobFraction &&
    whiteFraction <= config.maxBlobFraction &&
    whiteMean - backgroundBin >= config.minContrast

  if (!blackOk && !whiteOk) {
    return disabled
  }

  const usedBlackPixels = blackOk ? blackPixels : 0
  const usedWhitePixels = whiteOk ? whitePixels : 0
  const grayPixels = maskedPixels - usedBlackPixels - usedWhitePixels
  if (grayPixels / maskedPixels < config.minBackgroundFraction) {
    return disabled
  }

  return {
    valid: true,
    backgroundBin,
    blackCut: blackOk ? blackCut : 0,
    whiteCut: whiteOk ? whiteCut : ROI_HISTOGRAM_BINS,
    maskedPixels,
    blackPixels: usedBlackPixels,
    grayPixels,
    whitePixels: usedWhitePixels,
  }
}
