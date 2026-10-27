/** Times computeNeighborhoods on the same binary input the C++ benchmark reads. */
import { readFileSync } from 'node:fs'
import { computeNeighborhoods } from '@/lib/pointIdentity/proximityGraph'

const buffer = readFileSync('/tmp/cbench/points.bin')
const count = buffer.readInt32LE(0)
// Read field by field: a Float64Array view at byte offset 4 is not 8-byte
// aligned, which throws.
const xs = new Float64Array(count)
const ys = new Float64Array(count)
for (let i = 0; i < count; i++) {
  xs[i] = buffer.readDoubleLE(4 + i * 8)
  ys[i] = buffer.readDoubleLE(4 + count * 8 + i * 8)
}
const points = Array.from({ length: count }, (_, i) => ({ x: xs[i]!, y: ys[i]!, index: i }))

const reps = Number(process.argv[2] ?? 200)
computeNeighborhoods(points)
const started = performance.now()
for (let r = 0; r < reps; r++) computeNeighborhoods(points)
const ms = (performance.now() - started) / reps
console.log(`JS    points ${count}  ${ms.toFixed(3)} ms/iter`)
