import { describe, expect, it } from 'vitest'

import { invertMat3RowMajor } from '@/lib/aprilTagRaycast'
import { applyHomography, type Mat3 } from '@/lib/geometry'
import {
  designResiduals,
  fitHomography,
  hasAnchorSpread,
  isHomographyOrientationValid,
  robustHomography,
  type HomographyCorrespondence,
} from '@/lib/pointIdentity/homographyFit'

function multiplyMat3(first: Mat3, second: Mat3): Mat3 {
  const result: number[] = []
  for (let row = 0; row < 3; row += 1) {
    for (let column = 0; column < 3; column += 1) {
      let sum = 0
      for (let index = 0; index < 3; index += 1) {
        sum += first[row * 3 + index]! * second[index * 3 + column]!
      }
      result.push(sum)
    }
  }
  return result as Mat3
}

const PERSPECTIVE: Mat3 = [1.2, 0.1, 30, -0.05, 1.1, 40, 0.0005, 0.0002, 1]

describe('homographyFit', () => {
  it('rejects a transposed inverse with a round-trip check', () => {
    const inverse = invertMat3RowMajor(PERSPECTIVE)
    expect(inverse).toBeDefined()
    const product = multiplyMat3(PERSPECTIVE, inverse!)
    expect(product[0]).toBeCloseTo(1, 10)
    expect(product[1]).toBeCloseTo(0, 10)
    expect(product[2]).toBeCloseTo(0, 10)
    expect(product[3]).toBeCloseTo(0, 10)
    expect(product[4]).toBeCloseTo(1, 10)
    expect(product[5]).toBeCloseTo(0, 10)
    expect(product[6]).toBeCloseTo(0, 10)
    expect(product[7]).toBeCloseTo(0, 10)
    expect(product[8]).toBeCloseTo(1, 10)
  })

  it('recovers a perspective transform from four or more points', () => {
    const designPoints = [
      { x: 0, y: 0 },
      { x: 100, y: 0 },
      { x: 0, y: 100 },
      { x: 100, y: 100 },
      { x: 50, y: 50 },
      { x: 30, y: 70 },
    ]
    const correspondences: HomographyCorrespondence[] = designPoints.map((design) => {
      const projected = applyHomography(PERSPECTIVE, design.x, design.y)
      return { design, observed: { x: projected.x, y: projected.y } }
    })
    const fitted = fitHomography(correspondences)
    expect(fitted).toBeDefined()
    const heldOut = { x: 42, y: 17 }
    const expected = applyHomography(PERSPECTIVE, heldOut.x, heldOut.y)
    const actual = applyHomography(fitted!, heldOut.x, heldOut.y)
    expect(actual.x).toBeCloseTo(expected.x, 5)
    expect(actual.y).toBeCloseTo(expected.y, 5)
    expect(designResiduals(fitted!, correspondences).every((residual) => residual < 1e-5)).toBe(true)
  })

  it('trims outliers in design space', () => {
    const designPoints: Array<{ x: number; y: number }> = []
    for (let row = 0; row < 5; row += 1) {
      for (let column = 0; column < 5; column += 1) {
        designPoints.push({ x: column * 10, y: row * 10 })
      }
    }
    const trueTransform: Mat3 = [1, 0, 200, 0, 1, 150, 0, 0, 1]
    const correspondences: HomographyCorrespondence[] = designPoints.map((design, index) => {
      const projected = applyHomography(trueTransform, design.x, design.y)
      const outlier = index % 7 === 0
      return {
        design,
        observed: outlier
          ? { x: projected.x + 40, y: projected.y - 35 }
          : { x: projected.x, y: projected.y },
      }
    })
    const fitted = robustHomography(correspondences, {
      designSpacing: 10,
      minInliers: 6,
      minInlierRatio: 0.5,
      seed: 7,
    })
    expect(fitted).toBeDefined()
    expect(fitted!.inlierIndices.length).toBeGreaterThanOrEqual(18)
    expect(fitted!.rms).toBeLessThan(1)
    const expected = applyHomography(trueTransform, 12, 34)
    const actual = applyHomography(fitted!.matrix, 12, 34)
    expect(actual.x).toBeCloseTo(expected.x, 4)
    expect(actual.y).toBeCloseTo(expected.y, 4)
  })

  it('rejects a reflected transform', () => {
    const mirrored: Mat3 = [-1.2, 0.1, 30, -0.05, 1.1, 40, 0.0005, 0.0002, 1]
    expect(isHomographyOrientationValid(PERSPECTIVE, { x: 50, y: 50 })).toBe(true)
    expect(isHomographyOrientationValid(mirrored, { x: 50, y: 50 })).toBe(false)
  })

  it('checks anchor spread and collinearity', () => {
    expect(
      hasAnchorSpread(
        [
          { x: 0, y: 0 },
          { x: 100, y: 0 },
          { x: 0, y: 100 },
          { x: 100, y: 100 },
        ],
        10,
      ),
    ).toBe(true)
    expect(
      hasAnchorSpread(
        [
          { x: 0, y: 0 },
          { x: 10, y: 0 },
          { x: 20, y: 0 },
          { x: 30, y: 0 },
        ],
        10,
      ),
    ).toBe(false)
  })
})
