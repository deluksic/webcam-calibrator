/** Stage breakdown of the identity pipeline at full-target view.
 *
 *  Answers the question "is this CPU-bound work worth porting to WASM?" by
 *  showing which stage actually owns the 24 ms.
 *
 *  pnpm exec tsx scripts/identityProfile.ts
 */
import { createIdentityDesignBundle } from '@/lib/pointIdentity/identityPipeline'
import { computeFingerprints, truncateFingerprints } from '@/lib/pointIdentity/identity'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import {
  createLocalIcpOptions,
  MAX_FINGERPRINT_DEPTH,
  refineObservedIdentityWithLocalIcp,
} from '@/lib/pointIdentity/localIcpRefiner'
import { matchObservedIdentity } from '@/lib/pointIdentity/match'
import { DEFAULT_VIEW, projectPattern } from '@/lib/pointIdentity/projection'
import { buildProximityGraph, computeNeighborhoods } from '@/lib/pointIdentity/proximityGraph'
import { randomBinaryColors } from '@/lib/pointIdentity/random'

const { performance } = globalThis

const SPEC = { columns: 8, pointsPerEdge: 4, colorSeed: 1 }
const DEPTH = 2

function mulberry32(seed: number): () => number {
  let a = seed
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const design = createIdentityDesignBundle(SPEC)
const pattern = generateHexPattern({ columns: SPEC.columns, pointsPerEdge: SPEC.pointsPerEdge })
const colors = randomBinaryColors(pattern.points.length, SPEC.colorSeed)
const rng = mulberry32(7)

const projected = projectPattern(pattern.points, { ...DEFAULT_VIEW, yawDeg: 6, pitchDeg: -4 })
const jitter = 1.5
const points = projected.map((p, i) => ({
  x: p.x + (rng() - 0.5) * jitter,
  y: p.y + (rng() - 0.5) * jitter,
  color: colors[i] as 0 | 1,
}))

const observedPoints = points.map((p, index) => ({ x: p.x, y: p.y, index }))
const observedColors = points.map((p) => p.color)

const RUNS = 12
function time(label: string, fn: () => void): void {
  fn() // warm up
  const started = performance.now()
  for (let i = 0; i < RUNS; i++) {
    fn()
  }
  const ms = (performance.now() - started) / RUNS
  console.log(`${label.padEnd(34)} ${ms.toFixed(2).padStart(7)} ms`)
}

console.log(`points: ${points.length}  depth: ${DEPTH}`)

const observedGraph = buildProximityGraph(observedPoints, { pointsPerEdge: SPEC.pointsPerEdge })
time('computeNeighborhoods (kNN only)', () => {
  computeNeighborhoods(observedPoints)
})
time('buildProximityGraph (total)', () => {
  buildProximityGraph(observedPoints, { pointsPerEdge: SPEC.pointsPerEdge })
})
time(`computeFingerprints depth=${DEPTH}`, () => {
  computeFingerprints(observedGraph, observedColors, DEPTH)
})
time(`computeFingerprints depth=${MAX_FINGERPRINT_DEPTH}`, () => {
  computeFingerprints(observedGraph, observedColors, MAX_FINGERPRINT_DEPTH)
})

const baselineFingerprints = computeFingerprints(observedGraph, observedColors, DEPTH)
time('matchObservedIdentity (baseline)', () => {
  matchObservedIdentity(design.indexesByDepth[DEPTH - 1]!, observedGraph, observedColors, DEPTH, {
    fingerprints: baselineFingerprints,
  })
})

const baseline = matchObservedIdentity(
  design.indexesByDepth[DEPTH - 1]!,
  observedGraph,
  observedColors,
  DEPTH,
  { fingerprints: baselineFingerprints },
)
const recoveryFingerprints = computeFingerprints(observedGraph, observedColors, MAX_FINGERPRINT_DEPTH)
const options = createLocalIcpOptions({
  identityDepth: DEPTH,
  maxProofDepth: MAX_FINGERPRINT_DEPTH,
  minProofPower: 7,
  hopRadius: 8,
})
time('refineObservedIdentityWithLocalIcp', () => {
  refineObservedIdentityWithLocalIcp({
    designPattern: design.pattern,
    designColors: design.colors,
    designGraph: design.graph,
    designIndexesByDepth: design.indexesByDepth,
    observedGraph,
    observedColors,
    observedFingerprints: recoveryFingerprints,
    baseline,
    designCache: design.designCache,
    options,
  })
})

// --- A/B: two fingerprint passes vs one + truncation -------------------------
// Run interleaved so drift and GC hit both equally.
function ab(label: string, a: () => void, b: () => void, runs: number): void {
  for (let i = 0; i < 3; i++) {
    a()
    b()
  }
  let aTotal = 0
  let bTotal = 0
  for (let i = 0; i < runs; i++) {
    let t = performance.now()
    a()
    aTotal += performance.now() - t
    t = performance.now()
    b()
    bTotal += performance.now() - t
  }
  console.log(
    `${label.padEnd(34)} old ${(aTotal / runs).toFixed(2).padStart(6)} ms   ` +
      `new ${(bTotal / runs).toFixed(2).padStart(6)} ms   ` +
      `saved ${((aTotal - bTotal) / runs).toFixed(2).padStart(5)} ms`,
  )
}

ab(
  'fingerprints (depth + MAX vs MAX)',
  () => {
    computeFingerprints(observedGraph, observedColors, DEPTH)
    computeFingerprints(observedGraph, observedColors, MAX_FINGERPRINT_DEPTH)
  },
  () => {
    const full = computeFingerprints(observedGraph, observedColors, MAX_FINGERPRINT_DEPTH)
    truncateFingerprints(full, DEPTH)
  },
  20,
)
