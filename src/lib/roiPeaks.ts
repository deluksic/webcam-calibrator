/**
 * CPU finalization of GPU-generated ROI peak candidates.
 *
 * The GPU finds local maxima of the local-contrast response and writes the
 * 3x3 response patch around each one. Here we fit a quadratic,
 * reject non-maxima/weak peaks, run greedy NMS, and apply a per-color dynamic
 * height gate.
 */

export type RoiPeakCandidate = {
  x: number
  y: number
  /** 0 = black/dark, 1 = white/bright. */
  polarity: 0 | 1
  /** Row-major 3x3 response patch around (x, y). */
  values: ArrayLike<number>
}

export type RoiPeak = {
  x: number
  y: number
  polarity: 0 | 1
  /** Fitted peak height in response units. */
  peak: number
}

export type RoiPeakAnalysis = {
  peaks: RoiPeak[]
  blackThreshold: number
  whiteThreshold: number
  blackCount: number
  whiteCount: number
}

export type RoiPeakOptions = {
  nmsRadius?: number
  /** Height gate as a fraction of the median fitted peak for that color. */
  peakFraction?: number
  /** Absolute response floor. */
  minPeak?: number
  /** Use the combined median when a color has fewer than this many peaks. */
  minMedianSamples?: number
}

const DEFAULTS: Required<RoiPeakOptions> = {
  nmsRadius: 2,
  peakFraction: 0.2,
  minPeak: 0.005,
  minMedianSamples: 5,
}

/**
 * Least-squares fit of z = f + b x + d y + a x² + e xy + c y² over the 3x3
 * patch. Returns the fitted maximum vertex, or null if it is not a true
 * maximum (saddle/ridge) or the vertex is outside the patch.
 */
export function fitQuadraticMaximum(
  candidate: RoiPeakCandidate,
  minPeak = 0,
): RoiPeak | null {
  if (candidate.values.length < 9) {
    return null
  }
  let s0 = 0
  let sx = 0
  let sy = 0
  let sxx = 0
  let syy = 0
  let sxy = 0
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const z = Number(candidate.values[(dy + 1) * 3 + (dx + 1)] ?? 0)
      s0 += z
      sx += dx * z
      sy += dy * z
      sxx += dx * dx * z
      syy += dy * dy * z
      sxy += dx * dy * z
    }
  }

  const b = sx / 6
  const d = sy / 6
  const e = sxy / 4
  const sum = (sxx + syy) / 2 - (2 / 3) * s0
  const delta = (sxx - syy) / 2
  const a = (sum + delta) / 2
  const c = (sum - delta) / 2
  const f = (s0 - 6 * sum) / 9
  const det = 4 * a * c - e * e
  if (a >= 0 || c >= 0 || det <= 1e-9) {
    return null
  }

  const vx = (-2 * c * b + e * d) / det
  const vy = (-2 * a * d + e * b) / det
  if (!Number.isFinite(vx) || !Number.isFinite(vy) || Math.abs(vx) > 1.5 || Math.abs(vy) > 1.5) {
    return null
  }
  const peak = f + b * vx + d * vy + a * vx * vx + e * vx * vy + c * vy * vy
  if (!Number.isFinite(peak) || peak < minPeak) {
    return null
  }
  return {
    x: candidate.x + vx,
    y: candidate.y + vy,
    polarity: candidate.polarity,
    peak,
  }
}

function median(values: number[]): number {
  if (values.length === 0) {
    return 0
  }
  const sorted = [...values].sort((first, second) => first - second)
  const middle = sorted.length >> 1
  return sorted.length % 2 === 0
    ? ((sorted[middle - 1] ?? 0) + (sorted[middle] ?? 0)) / 2
    : (sorted[middle] ?? 0)
}

function greedyNms(peaks: RoiPeak[], radius: number): RoiPeak[] {
  const sorted = [...peaks].sort((first, second) => second.peak - first.peak)
  const minDistanceSq = radius * radius
  const kept: RoiPeak[] = []
  for (const peak of sorted) {
    let suppressed = false
    for (const other of kept) {
      const dx = peak.x - other.x
      const dy = peak.y - other.y
      if (dx * dx + dy * dy < minDistanceSq) {
        suppressed = true
        break
      }
    }
    if (!suppressed) {
      kept.push(peak)
    }
  }
  return kept
}

export function finalizeRoiPeaks(
  candidates: readonly RoiPeakCandidate[],
  options: RoiPeakOptions = {},
): RoiPeakAnalysis {
  const config = { ...DEFAULTS, ...options }
  const fitted: RoiPeak[] = []
  for (const candidate of candidates) {
    const peak = fitQuadraticMaximum(candidate, config.minPeak)
    if (peak !== null) {
      fitted.push(peak)
    }
  }

  const black = greedyNms(
    fitted.filter((peak) => peak.polarity === 0),
    config.nmsRadius,
  )
  const white = greedyNms(
    fitted.filter((peak) => peak.polarity === 1),
    config.nmsRadius,
  )
  const globalMedian = median(fitted.map((peak) => peak.peak))

  const gate = (peaks: RoiPeak[]): { kept: RoiPeak[]; threshold: number } => {
    if (peaks.length === 0) {
      return { kept: [], threshold: config.minPeak }
    }
    const colorMedian =
      peaks.length >= config.minMedianSamples
        ? median(peaks.map((peak) => peak.peak))
        : globalMedian
    const threshold = Math.max(config.minPeak, colorMedian * config.peakFraction)
    return {
      kept: peaks.filter((peak) => peak.peak >= threshold),
      threshold,
    }
  }

  const blackGate = gate(black)
  const whiteGate = gate(white)
  const peaks = [...blackGate.kept, ...whiteGate.kept]
  return {
    peaks,
    blackThreshold: blackGate.threshold,
    whiteThreshold: whiteGate.threshold,
    blackCount: blackGate.kept.length,
    whiteCount: whiteGate.kept.length,
  }
}
