import { describe, expect, it } from 'vitest'

import { jacobiEigenSym } from '@/lib/jacobiEigenSym'

describe('jacobiEigenSym', () => {
  it('produces ordered eigenvalues whose sum matches trace (diagonal input)', () => {
    const a = new Float64Array([3, 0, 0, 0, 1, 0, 0, 0, 2])
    const { values, vectors: _v } = jacobiEigenSym(a, 3)
    const s = values[0]! + values[1]! + values[2]!
    expect(s).toBeCloseTo(6, 5)
    expect(values[0]!).toBeLessThanOrEqual(values[1]!)
    expect(values[1]!).toBeLessThanOrEqual(values[2]!)
  })
})
