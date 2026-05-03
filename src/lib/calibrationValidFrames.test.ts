import { describe, expect, it } from 'vitest'

import type { CalibrationFrameObservation, Corners3, ImageTag } from '@/lib/calibrationTypes'
import type { Corners } from '@/lib/geometry'
import { countValidSolveFrames, filterFramesForLayout } from '@/lib/calibrationValidFrames'
import type { TargetLayout } from '@/lib/targetLayout'

function layoutFromTagIds(ids: number[]): TargetLayout {
  const m = new Map<number, Corners3>()
  for (const id of ids) {
    m.set(id, [
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 0, z: 0 },
      { x: 1, y: 1, z: 0 },
      { x: 0, y: 1, z: 0 },
    ])
  }
  return m
}

function frameWithTags(tagIds: number[], frameId = 1): CalibrationFrameObservation {
  const corners: Corners = [
    { x: 0, y: 0, score: 1 },
    { x: 10, y: 0, score: 1 },
    { x: 10, y: 10, score: 1 },
    { x: 0, y: 10, score: 1 },
  ]
  const tags: ImageTag[] = tagIds.map((tagId) => ({ tagId, corners }))
  return { frameId, tags }
}

describe('calibrationValidFrames', () => {
  it('returns empty when layout missing', () => {
    const pool: CalibrationFrameObservation[] = [frameWithTags([1, 2])]
    expect(countValidSolveFrames(pool, undefined)).toBe(0)
    expect(filterFramesForLayout(pool, undefined)).toEqual([])
  })

  it('drops frames with fewer than two layout tags after filter', () => {
    const lay = layoutFromTagIds([10, 20])
    const pool: CalibrationFrameObservation[] = [frameWithTags([10])]
    expect(countValidSolveFrames(pool, lay)).toBe(0)
  })

  it('keeps frames with two layout tags (8 corners)', () => {
    const lay = layoutFromTagIds([10, 20])
    const pool: CalibrationFrameObservation[] = [frameWithTags([10, 20])]
    expect(countValidSolveFrames(pool, lay)).toBe(1)
    const f = filterFramesForLayout(pool, lay)
    expect(f).toHaveLength(1)
    expect(f[0]!.tags.map((t) => t.tagId).sort((a, b) => a - b)).toEqual([10, 20])
  })

  it('counts multiple qualifying frames', () => {
    const lay = layoutFromTagIds([1, 2])
    const pool: CalibrationFrameObservation[] = [frameWithTags([1, 2], 1), frameWithTags([1, 2], 2)]
    expect(countValidSolveFrames(pool, lay)).toBe(2)
  })
})
