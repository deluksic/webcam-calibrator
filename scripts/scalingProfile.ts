/** Growth of each CPU-pipeline stage with the number of observed points.
 *
 *  Prints ms per stage per size, so the exponents can be read off. Run with:
 *  pnpm exec tsx scripts/scalingProfile.ts
 */
import { createIdentityDesignBundle, runIdentityPipeline } from '@/lib/pointIdentity/identityPipeline'
import { computeFingerprints } from '@/lib/pointIdentity/identity'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { DEFAULT_VIEW, projectPattern } from '@/lib/pointIdentity/projection'
import { buildProximityGraph, computeNeighborhoods } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'
import { filterRefinedPeaks, type RoiRefinedPeak } from '@/lib/roiNeighbourFilter'

const { performance } = globalThis

const SPEC = { columns: 12, pointsPerEdge: 4, colorSeed: 1 }
const DEPTH = 2
const RUNS = 7

function mulberry32(seed: number): () => number {
  let a = seed
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  const middle = sorted.length >> 1
  return sorted.length % 2 === 1 ? sorted[middle]! : (sorted[middle - 1]! + sorted[middle]!) / 2
}

function time(fn: () => void): number {
  fn()
  const samples: number[] = []
  for (let i = 0; i < RUNS; i++) {
    const started = performance.now()
    fn()
    samples.push(performance.now() - started)
  }
  return median(samples)
}

const design = createIdentityDesignBundle(SPEC)
const pattern = generateHexPattern({ columns: SPEC.columns, pointsPerEdge: SPEC.pointsPerEdge })
const colors = randomBinaryColors(pattern.points.length, SPEC.colorSeed)
const rng = mulberry32(7)
const projected = projectPattern(pattern.points, { ...DEFAULT_VIEW, yawDeg: 6, pitchDeg: -4 })

const xs = projected.map((p) => p.x)
const ys = projected.map((p) => p.y)
const cx = (Math.min(...xs) + Math.max(...xs)) / 2
const cy = (Math.min(...ys) + Math.max(...ys)) / 2
const halfSpan = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)) / 2

const rows: { points: number; filter: number; knn: number; graph: number; fps: number; identity: number }[] = []

for (const crop of [0.25, 0.35, 0.5, 0.7, 1]) {
  const keep = projected
    .map((p, i) => ({ p, i }))
    .filter(({ p }) => Math.abs(p.x - cx) <= halfSpan * crop && Math.abs(p.y - cy) <= halfSpan * crop)
  const points = keep.map(({ p, i }) => ({
    x: p.x + (rng() - 0.5) * 1.5,
    y: p.y + (rng() - 0.5) * 1.5,
    color: colors[i] as 0 | 1,
  }))
  const observedPoints = points.map((p, index) => ({ x: p.x, y: p.y, index }))
  const observedColors = points.map((p) => p.color)

  const peaks: RoiRefinedPeak[] = points.map((point) => ({
    x: point.x,
    y: point.y,
    polarity: point.color,
    peak: point.color === 0 ? 0.22 : 0.82,
    eigenvalue1: 0.004,
    eigenvalue2: 0.0025,
    hessianXY: 0,
  }))

  const filterMs = time(() => {
    filterRefinedPeaks(peaks)
  })
  let graph = buildProximityGraph(observedPoints, { pointsPerEdge: SPEC.pointsPerEdge })
  const knnMs = time(() => {
    computeNeighborhoods(observedPoints)
  })
  const graphMs = time(() => {
    graph = buildProximityGraph(observedPoints, { pointsPerEdge: SPEC.pointsPerEdge })
  })
  const fpsMs = time(() => {
    computeFingerprints(graph, observedColors, 12)
  })
  const identityMs = time(() => {
    runIdentityPipeline({ design, points, depth: DEPTH, recovery: true, frameId: 0 })
  })

  rows.push({
    points: points.length,
    filter: filterMs,
    knn: knnMs,
    graph: graphMs,
    fps: fpsMs,
    identity: identityMs,
  })
}

console.log('points  neighFilter     kNN     graph   fingerp12   identity')
for (const row of rows) {
  const f = (v: number): string => v.toFixed(2).padStart(8)
  console.log(
    String(row.points).padStart(6) +
      f(row.filter) +
      f(row.knn) +
      f(row.graph) +
      f(row.fps) +
      f(row.identity),
  )
}

// Log-log slope between the smallest and largest sizes: ~1 is linear, ~2 quadratic.
const first = rows[0]!
const last = rows[rows.length - 1]!
const exponent = (a: number, b: number): string =>
  (Math.log(b / a) / Math.log(last.points / first.points)).toFixed(2)
console.log('\nexponent (1 = linear, 2 = quadratic)')
console.log(`  neighFilter ${exponent(first.filter, last.filter)}`)
console.log(`  kNN         ${exponent(first.knn, last.knn)}`)
console.log(`  graph       ${exponent(first.graph, last.graph)}`)
console.log(`  fingerprints ${exponent(first.fps, last.fps)}`)
console.log(`  identity    ${exponent(first.identity, last.identity)}`)
