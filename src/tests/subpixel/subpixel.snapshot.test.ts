/**
 * Subpixel refinement harness — accuracy metrics only. `vitest --update` when refiner or scene changes.
 * On failure, PNGs via `attachFailureArtifacts`.
 */
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { refineHomographyV1, refineIdentity } from '@/tests/subpixel/refiners'
import { runSubpixelAlignmentCase } from '@/tests/subpixel/runCase'
import { attachFailureArtifacts, writeGreyPng, writeSobelMagPng } from '@/tests/utils/failureArtifacts'
import { finiteDifferenceSobelFromIntensity } from '@/tests/utils/syntheticAprilTag'

const baseScene = {
  kind: 'perspective' as const,
  width: 96,
  height: 96,
  tagId: 0,
  supersample: 4,
  maxEdgePx: 18,
}

const THIS_FILE = import.meta.url

describe('subpixel alignment', () => {
  it('v1 refiner metrics snapshot (mismatch + speckle)', () => {
    const args = {
      scene: baseScene,
      noise: [{ type: 'speckle' as const, amplitude: 0.1, seed: 42 }],
      initial: { kind: 'mismatchTemplate' as const, scale: 0.5 },
      refiner: refineHomographyV1,
      decodedTagId: 0,
    }
    const result = runSubpixelAlignmentCase(args)

    attachFailureArtifacts(THIS_FILE, (dir) => {
      const { width, height, grayscale } = result
      writeGreyPng(join(dir, 'grayscale.png'), width, height, grayscale)
      const sobel = finiteDifferenceSobelFromIntensity(grayscale, width, height, {
        gradientScale: 4,
      })
      writeSobelMagPng(join(dir, 'sobelMag.png'), width, height, sobel)
    })
    expect(result.metrics).toMatchSnapshot()
  })

  it('identity refiner leaves geometry unchanged vs init', () => {
    const args = {
      scene: baseScene,
      noise: [{ type: 'speckle' as const, amplitude: 0.05, seed: 99 }],
      initial: { kind: 'mismatchTemplate' as const, scale: 0.5 },
      refiner: refineIdentity,
      decodedTagId: 0,
    }
    const result = runSubpixelAlignmentCase(args)

    attachFailureArtifacts(THIS_FILE, (dir) => {
      const { width, height, grayscale } = result
      writeGreyPng(join(dir, 'grayscale.png'), width, height, grayscale)
      const sobel = finiteDifferenceSobelFromIntensity(grayscale, width, height, {
        gradientScale: 4,
      })
      writeSobelMagPng(join(dir, 'sobelMag.png'), width, height, sobel)
    })
    for (let i = 0; i < result.H_init.length; i++) {
      expect(result.H_refined[i]).toBeCloseTo(result.H_init[i]!, 6)
    }
  })
})
