import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { computeHomography } from '@/lib/geometry'
import { refineSubpixelHomographyV1 } from '@/lib/subpixelRefinement'
import { codeToPattern, TAG36H11_CODES } from '@/lib/tag36h11'
import { attachFailureArtifacts, writeGreyPng } from '@/tests/utils/failureArtifacts'
import { renderAprilTagIntensity } from '@/tests/utils/syntheticAprilTag'

const THIS_FILE = import.meta.url

describe('refineSubpixelHomographyV1', () => {
  it('returns a valid homography for axis-aligned tag', () => {
    const w = 64
    const h = 64
    const strip: [
      { x: number; y: number },
      { x: number; y: number },
      { x: number; y: number },
      { x: number; y: number },
    ] = [
      { x: 8, y: 8 },
      { x: 40, y: 8 },
      { x: 8, y: 40 },
      { x: 40, y: 40 },
    ]
    const pattern = codeToPattern(TAG36H11_CODES[0]!)
    const grayscale = renderAprilTagIntensity({
      width: w,
      height: h,
      corners: strip,
      pattern,
      supersample: 2,
    })
    const H = computeHomography([...strip])

    attachFailureArtifacts(THIS_FILE, (dir) => {
      writeGreyPng(join(dir, 'grayscale.png'), w, h, grayscale)
    })
    const out = refineSubpixelHomographyV1({
      width: w,
      height: h,
      grayscale,
      homography8: H,
    })
    expect(out.length).toBe(8)
    expect(out.every((x) => Number.isFinite(x))).toBe(true)
  })
})
