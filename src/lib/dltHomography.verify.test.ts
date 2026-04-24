import { describe, expect, it } from 'vitest'
import { solveHomographyDLT, type Correspondence } from '@/lib/dltHomography'
import { applyHomography, type Mat3 } from '@/lib/geometry'

describe('DLT Verify', () => {
  it('verify with simple identity case', () => {
    const pairs: Correspondence[] = [
      { plane: { x: 0, y: 0 }, image: { x: 0, y: 0 } },
      { plane: { x: 1, y: 0 }, image: { x: 1, y: 0 } },
      { plane: { x: 0, y: 1 }, image: { x: 0, y: 1 } },
      { plane: { x: 1, y: 1 }, image: { x: 1, y: 1 } },
    ]

    console.log('Input pairs:', pairs)
    const h = solveHomographyDLT(pairs)
    console.log('Computed H:', h)

    if (!h) {
      console.log('DLT returned undefined!')
      return
    }

    // Test each point
    console.log('\nTesting each point:')
    for (const p of pairs) {
      const result = applyHomography(h, p.plane.x, p.plane.y)
      const error = Math.sqrt((result.x - p.image.x) ** 2 + (result.y - p.image.y) ** 2)
      console.log(`  Point (${p.plane.x},${p.plane.y}) -> (${p.image.x},${p.image.y})`)
      console.log(`    Result: (${result.x.toFixed(6)},${result.y.toFixed(6)})`)
      console.log(`    Error: ${error.toFixed(6)}`)
    }

    // Overall error
    let totalError = 0
    for (const p of pairs) {
      const result = applyHomography(h, p.plane.x, p.plane.y)
      totalError += (result.x - p.image.x) ** 2 + (result.y - p.image.y) ** 2
    }
    const rms = Math.sqrt(totalError / pairs.length)
    console.log(`\nOverall RMS error: ${rms.toFixed(6)}`)

    expect(h).toBeDefined()
    expect(rms).toBeLessThan(0.01)
  })

  it('verify with translation homography', () => {
    // H: x = X + 10, y = Y + 5, w = 1  => row: [1, 0, 10, 0, 1, 5, 0, 0, 1]
    const Htrue: Mat3 = [1, 0, 10, 0, 1, 5, 0, 0, 1] as const

    const plane = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
    ]

    const pairs: Correspondence[] = plane.map((p) => ({
      plane: p,
      image: applyHomography(Htrue, p.x, p.y),
    }))

    console.log('True H:', Htrue)
    console.log('Pairs:', pairs)

    const h = solveHomographyDLT(pairs)
    console.log('Computed H:', h)

    if (!h) {
      console.log('DLT returned undefined!')
      return
    }

    // Test overall error
    let totalError = 0
    for (const p of pairs) {
      const result = applyHomography(h, p.plane.x, p.plane.y)
      totalError += (result.x - p.image.x) ** 2 + (result.y - p.image.y) ** 2
    }
    const rms = Math.sqrt(totalError / pairs.length)
    console.log(`\nOverall RMS error: ${rms.toFixed(6)}`)

    expect(h).toBeDefined()
    expect(rms).toBeLessThan(0.01)
  })
})