/** Times the identity pipeline against a realistic point count.
 *
 *  The frame loop awaits identity before presenting, so this number is the
 *  frame-time floor whenever the target is in view.
 *
 *  pnpm exec tsx scripts/identityTiming.ts
 */
import { createIdentityDesignBundle, runIdentityPipeline } from '@/lib/pointIdentity/identityPipeline'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { DEFAULT_VIEW, projectPattern } from '@/lib/pointIdentity/projection'
import { randomBinaryColors } from '@/lib/pointIdentity/random'

const { performance } = globalThis

const SPEC = { columns: 8, pointsPerEdge: 4, colorSeed: 1 }

function mulberry32(seed: number): () => number {
  let a = seed
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Project through a real pose, jitter, then keep a central crop. A crop is what
 *  a partial view of the target actually looks like, and it is where matching
 *  leaves ambiguous points for the ICP recovery path. */
function observe(
  pattern: ReturnType<typeof generateHexPattern>,
  colors: readonly number[],
  cropFraction: number,
  seed: number,
): { x: number; y: number; color: 0 | 1 }[] {
  const rng = mulberry32(seed)
  const projected = projectPattern(pattern.points, {
    ...DEFAULT_VIEW,
    yawDeg: 6,
    pitchDeg: -4,
  })
  const xs = projected.map((p) => p.x)
  const ys = projected.map((p) => p.y)
  const cx = (Math.min(...xs) + Math.max(...xs)) / 2
  const cy = (Math.min(...ys) + Math.max(...ys)) / 2
  const halfW = ((Math.max(...xs) - Math.min(...xs)) * cropFraction) / 2
  const halfH = ((Math.max(...ys) - Math.min(...ys)) * cropFraction) / 2
  return projected
    .map((p, i) => ({ p, i }))
    .filter(({ p }) => Math.abs(p.x - cx) <= halfW && Math.abs(p.y - cy) <= halfH)
    .map(({ p, i }) => ({
      x: p.x + (rng() - 0.5) * 1.5,
      y: p.y + (rng() - 0.5) * 1.5,
      color: colors[i] as 0 | 1,
    }))
}

const design = createIdentityDesignBundle(SPEC)
const pattern = generateHexPattern({ columns: SPEC.columns, pointsPerEdge: SPEC.pointsPerEdge })
const colors = randomBinaryColors(pattern.points.length, SPEC.colorSeed)
console.log(`design points: ${pattern.points.length}`)

for (const crop of [1, 0.75, 0.55, 0.4, 0.28]) {
  for (const recovery of [false, true]) {
    const points = observe(pattern, colors, crop, 7)
    // Warm up (fills per-depth loose indexes).
    runIdentityPipeline({ design, points, depth: 2, recovery, frameId: 0 })
    const runs = 12
    const started = performance.now()
    let resolved = 0
    let ambiguous = 0
    for (let i = 0; i < runs; i++) {
      const out = runIdentityPipeline({ design, points, depth: 2, recovery, frameId: i })
      resolved = out.metrics.resolved
      ambiguous = out.metrics.ambiguous + out.metrics.unresolved
    }
    const ms = (performance.now() - started) / runs
    console.log(
      `crop ${crop.toFixed(2)}  points ${String(points.length).padStart(3)}  ` +
        `recovery ${recovery ? 'on ' : 'off'}  ${ms.toFixed(2).padStart(7)} ms   ` +
        `resolved ${String(resolved).padStart(3)}  ambiguous ${String(ambiguous).padStart(3)}`,
    )
  }
}
