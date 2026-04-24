import { solveHomographyDLT } from './src/lib/dltHomography'

const pairs = [
  { plane: { x: 0, y: 0 }, image: { x: 0, y: 0 } },
  { plane: { x: 1, y: 0 }, image: { x: 1, y: 0 } },
  { plane: { x: 0, y: 1 }, image: { x: 0, y: 1 } },
  { plane: { x: 1, y: 1 }, image: { x: 1, y: 1 } },
]

console.log('Input pairs:', pairs)
const h = solveHomographyDLT(pairs)
console.log('Computed H:', h)

import { applyHomography } from './src/lib/geometry'
for (const p of pairs) {
  const result = applyHomography(h, p.plane.x, p.plane.y)
  const error = Math.sqrt((result.x - p.image.x) ** 2 + (result.y - p.image.y) ** 2)
  console.log(`  Point (${p.plane.x},${p.plane.y}) -> (${p.image.x},${p.image.y})`)
  console.log(`    Result: (${result.x.toFixed(6)},${result.y.toFixed(6)})`)
  console.log(`    Error: ${error.toFixed(6)}`)
}
