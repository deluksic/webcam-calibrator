import { describe, expect, it } from 'vitest'

import { mergeCalibrationFramesTopK } from '@/lib/calibrationTopK'
import type { ImageTag, TagObservation } from '@/lib/calibrationTypes'
import type { Corners } from '@/lib/geometry'
import { learnLayoutFromFrame } from '@/lib/targetLayout'

const unitSq: Corners = [
  { x: 0, y: 0 },
  { x: 100, y: 0 },
  { x: 0, y: 100 },
  { x: 100, y: 100 },
]

const shifted: Corners = [
  { x: 200, y: 0 },
  { x: 300, y: 0 },
  { x: 200, y: 100 },
  { x: 300, y: 100 },
]

describe('calibration first snapshot → pool', () => {
  it('learns layout and merge adds exactly one frame (two-tag invariant)', () => {
    const uniqueTags: TagObservation[] = [
      { tagId: 10, rotation: 0, corners: unitSq, score: 1 },
      { tagId: 20, rotation: 0, corners: shifted, score: 1 },
    ]
    const L = learnLayoutFromFrame(uniqueTags)
    expect(L).toBeDefined()

    const frameTagsModel: ImageTag[] = uniqueTags.map((t) => ({
      tagId: t.tagId,
      corners: [
        { ...t.corners[0]!, score: t.score },
        { ...t.corners[1]!, score: t.score },
        { ...t.corners[2]!, score: t.score },
        { ...t.corners[3]!, score: t.score },
      ],
    }))

    const { next, evicted } = mergeCalibrationFramesTopK([], [{ frameId: 42, tags: frameTagsModel }], 1000)
    expect(evicted).toBe(0)
    expect(next).toHaveLength(1)
    expect(next[0]!.tags.map((x) => x.tagId).sort((a, b) => a - b)).toEqual([10, 20])
  })
})
