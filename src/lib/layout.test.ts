import { describe, expect, it } from 'vitest'

import { tryComputeHomography, applyHomography, type Mat3, type Point } from '@/lib/geometry'
import { invertMat3RowMajor } from '@/lib/aprilTagRaycast'
import type { Corners } from '@/lib/geometry'

describe('Layout learning', () => {
  const TAG_SIZE = 1.0

  it('recomputes exact layout from perfect tag observations', () => {
    // Tag 1 at origin
    const tag1: Corners = [
      { x: 0, y: 0 },
      { x: TAG_SIZE, y: 0 },
      { x: 0, y: TAG_SIZE },
      { x: TAG_SIZE, y: TAG_SIZE },
    ]

    // Tag 2 offset by (3, 2)
    const tag2: Corners = [
      { x: 3, y: 2 },
      { x: 3 + TAG_SIZE, y: 2 },
      { x: 3, y: 2 + TAG_SIZE },
      { x: 3 + TAG_SIZE, y: 2 + TAG_SIZE },
    ]

    // Sort by tagId
    const sorted = [tag1, tag2].sort((a, b) => 0) // won't sort without tagId, but order doesn't matter
    const tags = [{ tagId: 1, corners: tag1 }, { tagId: 2, corners: tag2 }]

    // Anchor is lowest tagId (tag1)
    const anchor = tags[0]!
    console.log('[test] Anchor corners:', anchor.corners)

    // Compute homography from anchor corners
    const hAnchor = tryComputeHomography(anchor.corners) as Mat3
    console.log('[test] hAnchor:', hAnchor)
    expect(hAnchor).toBeDefined()
    if (!hAnchor) return

    const hInv = invertMat3RowMajor(hAnchor)
    expect(hInv).toBeDefined()
    if (!hInv) return

    console.log('[test] hInv:', hInv)

    // Expected: UNIT_SQUARE corners
    const expected = [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 0, y: 1 },
      { x: 1, y: 1 },
    ]
    console.log('[test] Expected anchor corners:', expected)

    // Map tag2 corners
    const tag2Mapped: Corners = [
      applyHomography(hInv, tag2[0].x, tag2[0].y),
      applyHomography(hInv, tag2[1].x, tag2[1].y),
      applyHomography(hInv, tag2[2].x, tag2[2].y),
      applyHomography(hInv, tag2[3].x, tag2[3].y),
    ]
    console.log('[test] Tag2 mapped corners:', tag2Mapped)

    const layout = new Map([
      [1, expected],
      [2, tag2Mapped],
    ])

    // Verify tag2 corners - tag2 spans from (3,2) to (4,3)
    expect(tag2Mapped[0]!.x).toBeCloseTo(3, 5)
    expect(tag2Mapped[0]!.y).toBeCloseTo(2, 5)
    expect(tag2Mapped[1]!.x).toBeCloseTo(4, 5)
    expect(tag2Mapped[1]!.y).toBeCloseTo(2, 5)
    expect(tag2Mapped[2]!.x).toBeCloseTo(3, 5)
    expect(tag2Mapped[2]!.y).toBeCloseTo(3, 5)
    expect(tag2Mapped[3]!.x).toBeCloseTo(4, 5)
    expect(tag2Mapped[3]!.y).toBeCloseTo(3, 5)
  })
})
