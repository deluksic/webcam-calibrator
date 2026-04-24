import { describe, expect, it } from 'vitest'
import { applyHomography, type Mat3 } from '@/lib/geometry'
import { solveHomographyDLT, type Correspondence } from '@/lib/dltHomography'

function hErr(h: Mat3, pairs: Correspondence[]): number {
  let s = 0
  for (const c of pairs) {
    const p = applyHomography(h, c.plane.x, c.plane.y)
    s += (p.x - c.image.x) ** 2 + (p.y - c.image.y) ** 2
  }
  return Math.sqrt(s / pairs.length)
}

describe('DLT Full Test', () => {
  it('test case 1: identity', () => {
    const I: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    const plane = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
      { x: 0.2, y: 0.8 },
    ]
    const pairs: Correspondence[] = plane.map((p) => ({ plane: p, image: applyHomography(I, p.x, p.y) }))
    const h = solveHomographyDLT(pairs)
    expect(h).toBeDefined()
    if (!h) return
    const error = hErr(h, pairs)
    console.log('Identity case error:', error.toFixed(6))
    expect(error).toBeLessThan(0.01)
  })

  it('test case 2: affine+translate (matches my manual trace)', () => {
    // H: u = 2*X + 10, v = 1.5*Y + 5, w = 1
    const Htrue: Mat3 = [2, 0, 10, 0, 1.5, 5, 0, 0, 1] as const
    const plane = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
      { x: 0.3, y: 0.7 },
      { x: 1.2, y: 0.2 },
    ]
    const pairs: Correspondence[] = plane.map((p) => ({
      plane: p,
      image: applyHomography(Htrue, p.x, p.y),
    }))

    console.log('True H:', Htrue)
    console.log('Pairs:')
    for (let i = 0; i < pairs.length; i++) {
      console.log(`  Point ${i}: (${pairs[i]!.plane.x.toFixed(1)},${pairs[i]!.plane.y.toFixed(1)}) -> (${pairs[i]!.image.x.toFixed(1)},${pairs[i]!.image.y.toFixed(1)})`)
    }

    const h = solveHomographyDLT(pairs)
    console.log('Computed H:', h)

    expect(h).toBeDefined()
    if (!h) return

    const error = hErr(h, pairs)
    console.log('Error:', error.toFixed(6))

    // Check if h matches Htrue up to scale
    const hArr = Array.from(h)
    const trueArr = Array.from(Htrue)
    console.log('Comparison:')
    for (let i = 0; i < 9; i++) {
      const hi = hArr[i]
      const ti = trueArr[i]
      if (hi !== undefined && ti !== undefined && Math.abs(hi) > 1e-10) {
        const k = hi / ti
        console.log(`  h[${i}] / H[${i}] = ${k.toFixed(6)}`)
      }
    }

    expect(error).toBeLessThan(0.01)
  })
})