/** Writes observed points + JS neighbour lists for the C comparison. */
import { writeFileSync } from 'node:fs'
import { generateHexPattern } from '@/lib/pointIdentity/hexPattern'
import { DEFAULT_VIEW, projectPattern } from '@/lib/pointIdentity/projection'
import { computeNeighborhoods } from '@/lib/pointIdentity/proximityGraph'

const SPEC = { columns: 12, pointsPerEdge: 4 }
const pattern = generateHexPattern(SPEC)
const projected = projectPattern(pattern.points, { ...DEFAULT_VIEW, yawDeg: 6, pitchDeg: -4 })

let seed = 7
const rnd = (): number => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff
  return seed / 0x7fffffff
}
const xs = new Float64Array(projected.length)
const ys = new Float64Array(projected.length)
projected.forEach((p, i) => {
  xs[i] = p.x + (rnd() - 0.5) * 1.5
  ys[i] = p.y + (rnd() - 0.5) * 1.5
})

const points = projected.map((_, i) => ({ x: xs[i]!, y: ys[i]!, index: i }))
const { neighborLists, localScales } = computeNeighborhoods(points)

const header = Buffer.alloc(4)
header.writeInt32LE(projected.length, 0)
writeFileSync('/tmp/cbench/points.bin', Buffer.concat([header, Buffer.from(xs.buffer), Buffer.from(ys.buffer)]))

const neigh = new Int32Array(projected.length * 12).fill(-1)
neighborLists.forEach((list, i) => list.forEach((v, j) => { neigh[i * 12 + j] = v }))
writeFileSync('/tmp/cbench/js_neighbors.bin', Buffer.from(neigh.buffer))
writeFileSync('/tmp/cbench/js_scales.bin', Buffer.from(new Float64Array(localScales).buffer))
console.log(`points ${projected.length}, wrote JS reference`)
