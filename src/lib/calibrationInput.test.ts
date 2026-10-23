import { describe, expect, it } from 'vitest'

import { buildCalibrationInput, CALIBRATION_MIN_SHARED_POINTS } from '@/lib/calibrationInput'
import type { CalibrationFrameObservation, TargetObjectPoint } from '@/lib/calibrationTypes'

function target(count: number): TargetObjectPoint[] {
  return Array.from({ length: count }, (_, index) => ({
    designId: index,
    x: index,
    y: index * 2,
    z: 0,
  }))
}

function frame(frameId: number, designIds: number[]): CalibrationFrameObservation {
  return {
    frameId,
    points: designIds.map((designId) => ({ designId, x: designId + 0.5, y: designId + 1.5 })),
  }
}

const ids = Array.from({ length: CALIBRATION_MIN_SHARED_POINTS }, (_, index) => index)

describe('buildCalibrationInput', () => {
  it('builds a shared object template from three overlapping frames', () => {
    const result = buildCalibrationInput(target(12), [frame(1, ids), frame(2, ids), frame(3, ids)])
    expect(result.kind).toBe('ok')
    if (result.kind !== 'ok') {
      return
    }
    expect(result.input.frameIds).toEqual([1, 2, 3])
    expect(result.input.sharedDesignIds).toEqual(ids)
    expect(result.input.objectPoints).toHaveLength(3)
    expect(result.input.objectPoints[0]).toHaveLength(ids.length)
    expect(result.input.imagePoints[0]![0]).toEqual([0.5, 1.5])
  })

  it('requires at least three usable frames', () => {
    const result = buildCalibrationInput(target(12), [frame(1, ids), frame(2, ids)])
    expect(result.kind).toBe('error')
  })

  it('drops frames with fewer than eight usable points', () => {
    const result = buildCalibrationInput(target(12), [
      frame(1, ids),
      frame(2, ids),
      frame(3, ids.slice(0, 7)),
    ])
    expect(result.kind).toBe('error')
  })

  it('requires enough shared design ids across views', () => {
    const result = buildCalibrationInput(target(20), [
      frame(1, ids),
      frame(2, ids),
      frame(3, ids.map((designId) => designId + 8)),
    ])
    expect(result.kind).toBe('error')
  })

  it('rejects a frame that assigns one design id twice', () => {
    const result = buildCalibrationInput(target(20), [
      frame(1, ids),
      frame(2, ids),
      frame(3, [...ids, ids[0]!]),
    ])
    expect(result.kind).toBe('error')
  })
})
