import { describe, expect, it } from 'vitest'

import { outOfPlaneTiltDeg, projectPattern, projectPoint } from '@/lib/pointIdentity/projection'
import type { PatternPoint, ViewPose } from '@/lib/pointIdentity/types'

function points(coords: Array<readonly [number, number]>): PatternPoint[] {
  return coords.map(([x, y], index) => ({ x, y, index }))
}

const FRONT: ViewPose = {
  yawDeg: 0,
  pitchDeg: 0,
  cameraDistance: 2000,
  focalLength: 1000,
  sphericalDistortion: 0,
}

describe('projectPoint', () => {
  it('projects a front-facing plane with a uniform scale', () => {
    const projected = projectPoint({ x: 100, y: -50 }, FRONT)
    expect(projected.x).toBeCloseTo(50, 9)
    expect(projected.y).toBeCloseTo(-25, 9)
  })

  it('moves closer points farther from the image centre', () => {
    const near = projectPoint({ x: 100, y: 0 }, { ...FRONT, cameraDistance: 1400 })
    const far = projectPoint({ x: 100, y: 0 }, { ...FRONT, cameraDistance: 2800 })
    expect(near.x).toBeGreaterThan(far.x)
  })

  it('collapses the x axis at 90 degrees of yaw', () => {
    const projected = projectPoint({ x: 100, y: 40 }, { ...FRONT, yawDeg: 90 })
    expect(projected.x).toBeCloseTo(0, 9)
    expect(projected.y).toBeGreaterThan(0)
  })

  it('collapses the y axis at 90 degrees of pitch', () => {
    const projected = projectPoint({ x: 100, y: 40 }, { ...FRONT, pitchDeg: 90 })
    expect(projected.y).toBeCloseTo(0, 9)
    expect(projected.x).toBeGreaterThan(0)
  })

  it('compresses the outer field like a fisheye while keeping the principal point fixed', () => {
    const field = points([
      [0, 0],
      [100, 0],
    ])
    const straight = projectPattern(field, FRONT)
    const warped = projectPattern(field, { ...FRONT, sphericalDistortion: 0.5 })
    expect(warped[0]).toEqual(straight[0])
    expect(Math.hypot(warped[1]!.x, warped[1]!.y)).toBeLessThan(Math.hypot(straight[1]!.x, straight[1]!.y))
  })

  it('leaves the projection unchanged at zero distortion', () => {
    const field = points([
      [10, 20],
      [-30, 15],
    ])
    expect(projectPattern(field, { ...FRONT, sphericalDistortion: 0 })).toEqual(projectPattern(field, FRONT))
  })

  it('preserves point identity when projecting a field', () => {
    const field = points([
      [0, 0],
      [10, 0],
      [0, 10],
    ])
    const projected = projectPattern(field, { ...FRONT, yawDeg: 24, pitchDeg: -19 })
    expect(projected.map((point) => point.index)).toEqual([0, 1, 2])
  })
})

describe('outOfPlaneTiltDeg', () => {
  it('is zero for a fronto-parallel view', () => {
    expect(outOfPlaneTiltDeg(FRONT)).toBeCloseTo(0, 9)
  })

  it('matches the yaw angle when pitch is zero', () => {
    expect(outOfPlaneTiltDeg({ ...FRONT, yawDeg: 60 })).toBeCloseTo(60, 9)
  })

  it('combines yaw and pitch', () => {
    const expected = (Math.acos(Math.cos((45 * Math.PI) / 180) ** 2) * 180) / Math.PI
    expect(outOfPlaneTiltDeg({ ...FRONT, yawDeg: 45, pitchDeg: 45 })).toBeCloseTo(expected, 9)
  })
})
