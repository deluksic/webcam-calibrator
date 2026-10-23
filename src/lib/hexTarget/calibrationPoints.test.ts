import { describe, expect, it } from 'vitest'

import { toCalibrationFramePoints } from '@/lib/hexTarget/calibrationPoints'
import type { HexTargetMatch } from '@/lib/hexTarget/match'
import { defaultHexTargetModel } from '@/lib/hexTarget/target'
import {
  IDENTITY_AMBIGUOUS,
  IDENTITY_RECOVERED,
  IDENTITY_RESOLVED,
} from '@/lib/pointIdentity/match'

function matchWith(points: HexTargetMatch['points']): HexTargetMatch {
  return {
    points,
    edges: [],
    summary: {
      total: points.length,
      resolved: 0,
      recovered: 0,
      ambiguous: 0,
      unresolved: 0,
      conflicts: 0,
    },
  }
}

describe('toCalibrationFramePoints', () => {
  const design = defaultHexTargetModel()

  it('accepts resolved points and rejects ambiguous ones', () => {
    const match = matchWith([
      { x: 10, y: 20, color: 0, designId: 3, status: IDENTITY_RESOLVED },
      { x: 30, y: 40, color: 1, designId: undefined, status: IDENTITY_AMBIGUOUS },
    ])
    const result = toCalibrationFramePoints(match, design)
    expect(result.rejectedReason).toBeUndefined()
    expect(result.points).toEqual([{ designId: 3, x: 10, y: 20, score: 1 }])
  })

  it('rejects duplicate design ids instead of choosing one', () => {
    const match = matchWith([
      { x: 10, y: 20, color: 0, designId: 3, status: IDENTITY_RESOLVED },
      { x: 30, y: 40, color: 1, designId: 3, status: IDENTITY_RESOLVED },
    ])
    expect(toCalibrationFramePoints(match, design).rejectedReason).toBe('duplicate-design-id')
  })

  it('accepts safe recovered points and rejects conflicts or far predictions', () => {
    const match = matchWith([
      {
        x: 10,
        y: 20,
        color: 0,
        designId: 4,
        status: IDENTITY_RECOVERED,
        designDistance: design.spacing * 0.1,
      },
      {
        x: 30,
        y: 40,
        color: 1,
        designId: 5,
        status: IDENTITY_RECOVERED,
        designDistance: design.spacing * 0.1,
        geometryConflict: true,
      },
      {
        x: 50,
        y: 60,
        color: 1,
        designId: 6,
        status: IDENTITY_RECOVERED,
        designDistance: design.spacing,
      },
    ])
    const result = toCalibrationFramePoints(match, design)
    expect(result.points).toEqual([{ designId: 4, x: 10, y: 20, score: 0.8 }])
  })
})
