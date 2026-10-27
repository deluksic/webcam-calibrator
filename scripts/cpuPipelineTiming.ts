/** Times the whole CPU pipeline — neighbour filter + identity — for one frame at
 *  a given observation dropout.
 *
 *  Both stages run in the identity worker, back to back, and the frame loop
 *  waits for the identity result before presenting, so their sum is the frame
 *  budget this pipeline demands whenever the target is in view.
 *
 *  pnpm exec tsx scripts/cpuPipelineTiming.ts [dropoutPercent]
 */
import { createIdentityDesignBundle, runIdentityPipeline } from '@/lib/pointIdentity/identityPipeline'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { DEFAULT_VIEW, projectPattern } from '@/lib/pointIdentity/projection'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import { filterRefinedPeaks, type RoiRefinedPeak } from '@/lib/roiNeighbourFilter'

const { performance } = globalThis

const SPEC = { columns: 8, pointsPerEdge: 4, colorSeed: 1 }
const DEPTH = 2
/** Frames measured; median and min are reported because timings here are noisy. */
const RUNS = 15

function mulberry32(seed: number): () => number {
  let a = seed
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gaussian(rng: () => number, mean: number, sd: number): number {
  const u = Math.max(rng(), 1e-12)
  const v = rng()
  return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = sorted.length >> 1
  return sorted.length % 2 === 1
    ? sorted[middle]!
    : (sorted[middle - 1]! + sorted[middle]!) / 2
}

const design = createIdentityDesignBundle(SPEC)
const pattern = generateHexPattern({ columns: SPEC.columns, pointsPerEdge: SPEC.pointsPerEdge })
const colors = randomBinaryColors(pattern.points.length, SPEC.colorSeed)
const dropout = Number(process.argv[2] ?? '2') / 100

const rng = mulberry32(7)
const projected = projectPattern(pattern.points, { ...DEFAULT_VIEW, yawDeg: 6, pitchDeg: -4 })

type Observed = { x: number; y: number; color: 0 | 1 }
const observed: Observed[] = []
for (let index = 0; index < projected.length; index++) {
  if (rng() < dropout) {
    continue // dropped observation
  }
  const point = projected[index]!
  observed.push({
    x: point.x + (rng() - 0.5) * 1.5,
    y: point.y + (rng() - 0.5) * 1.5,
    color: colors[index] as 0 | 1,
  })
}

/** Plausible refined-peak features: real markers cluster tightly per polarity,
 *  with a small outlier fraction the filter is meant to reject. */
const peaks: RoiRefinedPeak[] = observed.map((point) => {
  const outlier = rng() < 0.08
  const scale = outlier ? 3.5 : 1
  return {
    x: point.x,
    y: point.y,
    polarity: point.color,
    peak: point.color === 0 ? gaussian(rng, 0.22, 0.06) : gaussian(rng, 0.82, 0.05),
    eigenvalue1: Math.max(1e-6, gaussian(rng, 0.004 * scale, 0.0012 * scale)),
    eigenvalue2: Math.max(1e-6, gaussian(rng, 0.0025 * scale, 0.0008 * scale)),
    hessianXY: gaussian(rng, 0, 0.0004 * scale),
  }
})

const filterMs: number[] = []
const identityMs: number[] = []
const totalMs: number[] = []
let keptCount = 0
let resolved = 0

for (let run = 0; run < RUNS; run++) {
  const started = performance.now()
  const filtered = filterRefinedPeaks(peaks)
  const afterFilter = performance.now()
  const output = runIdentityPipeline({
    design,
    points: filtered.kept,
    depth: DEPTH,
    recovery: true,
    frameId: run,
  })
  const finished = performance.now()

  filterMs.push(afterFilter - started)
  identityMs.push(finished - afterFilter)
  totalMs.push(finished - started)
  keptCount = filtered.kept.length
  resolved = output.metrics.resolved
}

const fmt = (value: number): string => value.toFixed(2).padStart(6)
console.log(`dropout ${(dropout * 100).toFixed(0)}%  observed ${observed.length}/${projected.length}`)
console.log(
  `neighbour filter   median ${fmt(median(filterMs))} ms   min ${fmt(Math.min(...filterMs))} ms`,
)
console.log(
  `identity           median ${fmt(median(identityMs))} ms   min ${fmt(Math.min(...identityMs))} ms`,
)
console.log(`TOTAL              median ${fmt(median(totalMs))} ms   min ${fmt(Math.min(...totalMs))} ms`)
console.log(`kept ${keptCount}/${peaks.length}   resolved ${resolved}/${keptCount}`)
