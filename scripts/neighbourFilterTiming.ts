/** Times the CPU neighbour filter at realistic peak counts.
 *
 *  Runs several repetitions per size and reports the median, because this
 *  machine's timings drift between runs.
 *
 *  pnpm exec tsx scripts/neighbourFilterTiming.ts
 */
import { filterRefinedPeaks, type RoiRefinedPeak } from '@/lib/roiNeighbourFilter'

const { performance } = globalThis

function mulberry32(seed: number): () => number {
  let a = seed
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Gaussian via Box-Muller, so feature distributions look like real peaks
 *  rather than a uniform block that makes every z-score similar. */
function gaussian(rng: () => number, mean: number, sd: number): number {
  const u = Math.max(rng(), 1e-12)
  const v = rng()
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function makePeaks(count: number, seed: number): RoiRefinedPeak[] {
  const rng = mulberry32(seed)
  const peaks: RoiRefinedPeak[] = []
  for (let i = 0; i < count; i++) {
    const polarity = (rng() < 0.5 ? 0 : 1) as 0 | 1
    const peak = polarity === 0 ? gaussian(rng, 0.22, 0.06) : gaussian(rng, 0.82, 0.05)
    // A small minority are outliers, which is what the filter exists to reject.
    const outlier = rng() < 0.08
    const scale = outlier ? 3.5 : 1
    peaks.push({
      x: rng() * 1280,
      y: rng() * 720,
      polarity,
      peak,
      eigenvalue1: Math.max(1e-6, gaussian(rng, 0.004 * scale, 0.0012 * scale)),
      eigenvalue2: Math.max(1e-6, gaussian(rng, 0.0025 * scale, 0.0008 * scale)),
      hessianXY: gaussian(rng, 0, 0.0004 * scale),
    })
  }
  return peaks
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const mid = Math.floor(sorted.length / 2)
  return sorted.length % 2 === 0 ? (sorted[mid - 1]! + sorted[mid]!) / 2 : sorted[mid]!
}

const RUNS = 25
const SIZES = [77, 150, 300, 476, 700, 1200]

for (const size of SIZES) {
  const peaks = makePeaks(size, 11)
  // Warm up (JIT + first-run allocation).
  for (let i = 0; i < 3; i++) {
    filterRefinedPeaks(peaks)
  }
  const samples: number[] = []
  let kept = 0
  for (let i = 0; i < RUNS; i++) {
    const started = performance.now()
    const result = filterRefinedPeaks(peaks)
    samples.push(performance.now() - started)
    kept = result.kept.length
  }
  console.log(
    `peaks ${String(size).padStart(4)}  median ${median(samples).toFixed(3).padStart(7)} ms  ` +
      `min ${Math.min(...samples).toFixed(3).padStart(6)} ms  kept ${kept}/${size}`,
  )
}
