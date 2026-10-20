import { describe, expect, it } from 'vitest'

import { createRandom, randomBinaryColors } from '@/lib/pointIdentity/random'

describe('randomBinaryColors', () => {
  it('is reproducible from a seed', () => {
    expect(randomBinaryColors(32, 42)).toEqual(randomBinaryColors(32, 42))
  })

  it('produces only binary values', () => {
    const colors = randomBinaryColors(200, 7)
    expect(colors).toHaveLength(200)
    for (const color of colors) {
      expect(color === 0 || color === 1).toBe(true)
    }
  })

  it('changes with the seed', () => {
    expect(randomBinaryColors(64, 1)).not.toEqual(randomBinaryColors(64, 2))
  })
})

describe('createRandom', () => {
  it('stays in the unit interval', () => {
    const random = createRandom(123)
    for (let index = 0; index < 500; index += 1) {
      const value = random()
      expect(value).toBeGreaterThanOrEqual(0)
      expect(value).toBeLessThan(1)
    }
  })
})
