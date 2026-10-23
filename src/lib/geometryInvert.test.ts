import { describe, expect, it } from 'vitest'

import { invertMat3RowMajor, type Mat3 } from '@/lib/geometry'

function multiply(a: Mat3, b: Mat3): Mat3 {
  const out: Mat3 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      out[row * 3 + column] =
        a[row * 3]! * b[column]! +
        a[row * 3 + 1]! * b[3 + column]! +
        a[row * 3 + 2]! * b[6 + column]!
    }
  }
  return out
}

describe('invertMat3RowMajor', () => {
  it('inverts a non-singular matrix', () => {
    const matrix: Mat3 = [2, 0, 1, 0, 3, 0, 1, 0, 2]
    const inverse = invertMat3RowMajor(matrix)
    expect(inverse).toBeDefined()
    const product = multiply(matrix, inverse!)
    expect(product[0]).toBeCloseTo(1)
    expect(product[4]).toBeCloseTo(1)
    expect(product[8]).toBeCloseTo(1)
    expect(product[1]).toBeCloseTo(0)
    expect(product[3]).toBeCloseTo(0)
  })

  it('returns undefined for a singular matrix', () => {
    expect(invertMat3RowMajor([1, 2, 3, 2, 4, 6, 0, 0, 1])).toBeUndefined()
  })
})
