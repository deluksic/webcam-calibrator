import { describe, expect, it } from 'vitest'

import { solveHomographyDLT, type Correspondence } from './dltHomography'

describe('solveHomographyDLT', () => {
  it('succeeds on simple grid (no noise)', () => {
    const correspondences: Correspondence[] = [
      { plane: { x: 0, y: 0 }, image: { x: 0, y: 0 } },
      { plane: { x: 1, y: 0 }, image: { x: 800, y: 0 } },
      { plane: { x: 0, y: 1 }, image: { x: 0, y: 600 } },
      { plane: { x: 1, y: 1 }, image: { x: 800, y: 600 } },
    ]

    const h = solveHomographyDLT(correspondences)
    expect(h).toBeDefined()
    if (h) {
      expect(h[8]!).toBeCloseTo(1, -2)
    }
  })

  it('fails on nearly degenerate (2 points)', () => {
    const correspondences: Correspondence[] = [
      { plane: { x: 0, y: 0 }, image: { x: 0, y: 0 } },
      { plane: { x: 1, y: 0 }, image: { x: 800, y: 0 } },
    ]

    const h = solveHomographyDLT(correspondences)
    expect(h).toBeUndefined()
  })

  it('fails on co-linear points', () => {
    const correspondences: Correspondence[] = [
      { plane: { x: 0, y: 0 }, image: { x: 0, y: 0 } },
      { plane: { x: 1, y: 0 }, image: { x: 800, y: 0 } },
      { plane: { x: 2, y: 0 }, image: { x: 1600, y: 0 } },
      { plane: { x: 3, y: 0 }, image: { x: 2400, y: 0 } },
    ]

    const h = solveHomographyDLT(correspondences)
    expect(h).toBeUndefined()
  })
})
