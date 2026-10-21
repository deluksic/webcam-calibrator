import { describe, expect, it } from 'vitest'

import { DEFAULT_MERGE_TUNING, grayMean, grayVariance, shouldMergeComponents } from '@/lib/hexDebug/mergePredicate'

describe('shouldMergeComponents', () => {
  it('merges two perfectly uniform matching components', () => {
    expect(
      shouldMergeComponents(
        { mean: 0.5, variance: 0, area: 10 },
        { mean: 0.51, variance: 0, area: 10 },
        DEFAULT_MERGE_TUNING,
      ),
    ).toBe(true)
  })

  it('stays strict when one component is uniform', () => {
    expect(
      shouldMergeComponents(
        { mean: 0.5, variance: 0, area: 10 },
        { mean: 0.55, variance: 0.02, area: 10 },
        DEFAULT_MERGE_TUNING,
      ),
    ).toBe(false)
  })

  it('allows more color difference when both components are noisy', () => {
    expect(
      shouldMergeComponents(
        { mean: 0.5, variance: 0.01, area: 10 },
        { mean: 0.56, variance: 0.01, area: 10 },
        DEFAULT_MERGE_TUNING,
      ),
    ).toBe(true)
  })

  it('caps the tolerance before black and white can merge', () => {
    expect(
      shouldMergeComponents(
        { mean: 0, variance: 0.25, area: 10 },
        { mean: 1, variance: 0.25, area: 10 },
        DEFAULT_MERGE_TUNING,
      ),
    ).toBe(false)
  })

  it('ignores empty components', () => {
    expect(
      shouldMergeComponents(
        { mean: 0.5, variance: 0, area: 0 },
        { mean: 0.5, variance: 0, area: 1 },
        DEFAULT_MERGE_TUNING,
      ),
    ).toBe(false)
  })
})

describe('fixed-point moments', () => {
  it('computes mean and variance from quantized sums', () => {
    const values = [0, 0, 127, 127]
    const sum = values.reduce((total, value) => total + value, 0)
    const sumSq = values.reduce((total, value) => total + value * value, 0)
    expect(grayMean(sum, values.length)).toBeCloseTo(0.5, 5)
    expect(grayVariance(sum, sumSq, values.length)).toBeCloseTo(0.25, 5)
  })
})
