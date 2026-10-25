import { describe, expect, it } from 'vitest'

import {
  CUBIC_5X5_TERM_COUNT,
  CUBIC_DESIGN_MATRIX,
  CUBIC_FIT_COEFFICIENTS,
  fitCubic5x5,
  GAUSSIAN_BLUR_WEIGHTS,
} from '@/lib/roiPeakFit'

function sampleSurface(coefficient: (u: number, v: number) => number): Float64Array {
  const values = new Float64Array(25)
  let index = 0
  for (let dy = -2; dy <= 2; dy += 1) {
    for (let dx = -2; dx <= 2; dx += 1) {
      values[index] = coefficient(dx / 2, dy / 2)
      index += 1
    }
  }
  return values
}

describe('cubic 5x5 fit', () => {
  it('precomputes a pseudo-inverse that matches the design matrix', () => {
    expect(CUBIC_FIT_COEFFICIENTS).toHaveLength(10 * 25)
    for (let first = 0; first < CUBIC_5X5_TERM_COUNT; first += 1) {
      for (let second = 0; second < CUBIC_5X5_TERM_COUNT; second += 1) {
        let sum = 0
        for (let sample = 0; sample < 25; sample += 1) {
          sum +=
            CUBIC_FIT_COEFFICIENTS[first * 25 + sample]! * CUBIC_DESIGN_MATRIX[sample * CUBIC_5X5_TERM_COUNT + second]!
        }
        expect(sum).toBeCloseTo(first === second ? 1 : 0, 9)
      }
    }
  })

  it('recovers a subpixel maximum and its physical eigenvalues', () => {
    const values = sampleSurface((u, v) => 0.5 - 0.1 * (u - 0.2) * (u - 0.2) - 0.2 * (v + 0.1) * (v + 0.1))
    const fitted = fitCubic5x5(values)
    expect(fitted).not.toBeNull()
    expect(fitted!.u).toBeCloseTo(0.2, 6)
    expect(fitted!.v).toBeCloseTo(-0.1, 6)
    expect(fitted!.peak).toBeCloseTo(0.5, 6)
    expect(fitted!.eigenvalue1).toBeCloseTo(-0.1, 6)
    expect(fitted!.eigenvalue2).toBeCloseTo(-0.05, 6)
  })

  it('returns the physical Hessian off-diagonal term', () => {
    const values = sampleSurface((u, v) => 0.5 - 0.1 * u * u - 0.05 * v * v - 0.03 * u * v)
    const fitted = fitCubic5x5(values)
    expect(fitted).not.toBeNull()
    expect(fitted!.hessianXY).toBeCloseTo(-0.03 / 4, 8)
  })

  it('returns the saddle point so the polarity test can reject it', () => {
    const fitted = fitCubic5x5(sampleSurface((u, v) => u * u - v * v))
    expect(fitted).not.toBeNull()
    expect(fitted!.u).toBeCloseTo(0, 6)
    expect(fitted!.v).toBeCloseTo(0, 6)
    expect(fitted!.eigenvalue1).toBeLessThan(0)
    expect(fitted!.eigenvalue2).toBeGreaterThan(0)
  })

  it('rejects a surface with a stationary point outside the window', () => {
    const fitted = fitCubic5x5(sampleSurface((u, v) => -(u - 1.6) * (u - 1.6) - (v - 1.6) * (v - 1.6)))
    expect(fitted).toBeNull()
  })

  it('rejects a degenerate Hessian', () => {
    const fitted = fitCubic5x5(sampleSurface((u) => u ** 3))
    expect(fitted).toBeNull()
  })
})

describe('OpenCV sigma 1.0 kernel', () => {
  it('has the expected 9 taps and unit sum', () => {
    expect(GAUSSIAN_BLUR_WEIGHTS).toHaveLength(9)
    expect(GAUSSIAN_BLUR_WEIGHTS.reduce((sum, weight) => sum + weight, 0)).toBeCloseTo(1, 6)
    expect(GAUSSIAN_BLUR_WEIGHTS[4]).toBeCloseTo(0.39894348, 8)
  })
})
