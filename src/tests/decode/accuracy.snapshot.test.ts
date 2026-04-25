/**
 * Accuracy-only snapshot: same speckle + perspective family as decode stress (`DECODE_STRESS_SIZES`).
 * Run `vitest --update` when decode pipeline changes intentionally.
 * On failure, PNGs are written under `output/test-failures/` via `attachFailureArtifacts`.
 */
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { DECODE_STRESS_SIZES, DECODE_STRESS_SPECKLE_AMP, decodeStressSpeckleSeed } from '@/lib/decodeStressHarness'
import { decodeTagPattern } from '@/lib/grid'
import { codeToPattern, decodeTag36h11Best, TAG36H11_CODES } from '@/lib/tag36h11'
import { buildPerspectiveAccuracyTable, cellErrorsVsTruth } from '@/tests/decode/accuracyTable'
import { buildRasterPack } from '@/tests/shared/pipeline'
import {
  attachFailureArtifacts,
  writeCellLegendPng,
  writeGreyPng,
  writeSobelMagPng,
} from '@/tests/utils/failureArtifacts'

const { max, floor } = Math
const STRESS_SUPERSAMPLE = 4
const THIS_FILE = import.meta.url

describe('decode accuracy snapshots', () => {
  it('perspective + speckle characterization table (tag 0)', () => {
    const tagId = 0
    const truthPat = codeToPattern(TAG36H11_CODES[tagId]!)
    const sizes = [...DECODE_STRESS_SIZES]

    const table = buildPerspectiveAccuracyTable([...DECODE_STRESS_SIZES], tagId, STRESS_SUPERSAMPLE)

    attachFailureArtifacts(THIS_FILE, (dir) => {
      for (const wh of sizes) {
        const pack = buildRasterPack({
          scene: {
            kind: 'perspective',
            width: wh,
            height: wh,
            tagId,
            supersample: STRESS_SUPERSAMPLE,
          },
          noise: [
            {
              type: 'speckle',
              amplitude: DECODE_STRESS_SPECKLE_AMP,
              seed: decodeStressSpeckleSeed(wh, wh, tagId),
            },
          ],
        })
        const decodedPattern = decodeTagPattern(pack.groundTruthStrip, pack.sobel, wh, undefined, wh)
        if (!decodedPattern) {
          continue
        }
        const unknowns = decodedPattern.filter((v) => v === -1 || v === -2).length
        const cellErr = cellErrorsVsTruth(decodedPattern, truthPat)
        const dist = decodeTag36h11Best(decodedPattern, 8).dist
        if (cellErr === 0 && dist === 0 && unknowns === 0) {
          continue
        }

        const tag = `w${wh}-cellErr${cellErr}-unk${unknowns}-ham${dist}`
        writeGreyPng(join(dir, `${tag}-intensity.png`), wh, wh, pack.grayscale)
        writeSobelMagPng(join(dir, `${tag}-sobelMag.png`), wh, wh, pack.sobel)
        writeCellLegendPng(join(dir, `${tag}-cells-rgb.png`), decodedPattern, truthPat, max(12, floor(240 / 6)))
      }
    })
    expect(table).toMatchSnapshot()
  })
})
