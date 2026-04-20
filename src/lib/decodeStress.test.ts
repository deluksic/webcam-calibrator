/**
 * Characterizes `decodeTagPattern` + dictionary decode under **perspective** and **low resolution**.
 * Tightening these numbers documents the cliff; adjust if pipeline changes.
 *
 * **Supersampling:** each output pixel averages `ss×ss` tag samples on a regular subpixel grid inside the
 * pixel square (`(sx+0.5)/ss` offsets), inverse‑warped through the quad homography—so `ss` is purely a
 * nicer antialiased intensity raster, then **central-difference Sobel** runs on that grid. `ss === 2` can
 * resonate badly with FD Sobel; this file uses **4** (same spirit as other decode tests here).
 *
 * **Speckle:** deterministic ±`DECODE_STRESS_SPECKLE_AMP` uniform noise on intensity before Sobel
 * (`decodeStressHarness.ts`). **`decodeStressSuite`** allows modest Hamming / cell-error / unknown slack; re-measure
 * speckle headroom with `pnpm run find:decode-stress-speckle` after pipeline changes.
 *
 * **Homography mismatch:** raster uses the fitted quad; decode uses corners offset by
 * `DECODE_STRESS_HOMOGRAPHY_MISMATCH_SCALE *` `DECODE_STRESS_HOMOGRAPHY_MISMATCH_OFFSET_TEMPLATE_PX`
 * (`decodeStressStripWithHomographyMismatchOffsetsPx`); `buildTagGrid` recomputes H from the perturbed strip.
 * Re-tune scale with `pnpm run find:decode-stress-homography` (grid scan; `passes(scale)` is not
 * guaranteed monotone).
 *
 * **Failed tests** write PNGs under `output/test-failures/` (intensity, Sobel magnitude, 6×6 cell legend when applicable).
 * Topic harness: `src/tests/decode/` — `pnpm test:decode`.
 */
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import {
  DECODE_STRESS_SIZES,
  DECODE_STRESS_SPECKLE_AMP,
  decodeStressAxisStrip,
  decodeStressCornersGridOrder,
  decodeStressFitPerspectiveStrip,
  decodeStressRasterSobel,
  decodeStressStripWithHomographyMismatchOffsetsPx,
  decodeStressSynthetic,
  decodeStressSyntheticWithHomographyMismatch,
} from '@/lib/decodeStressHarness'
import type { Point } from '@/lib/geometry'
import { buildTagGrid, decodeTagPattern } from '@/lib/grid'
import { TAG36H11_CODES, codeToPattern, decodeTag36h11Best, type TagPattern } from '@/lib/tag36h11'
import {
  attachFailureArtifacts,
  writeCellLegendPng,
  writeGreyPng,
  writeSobelMagPng,
} from '@/tests/utils/failureArtifacts'

const THIS_FILE = import.meta.url

/** See file header: avoid ss=2 + FD-Sobel resonance when characterizing decode. */
const STRESS_SUPERSAMPLE = 4

function decodeSynthetic(
  w: number,
  h: number,
  strip: [Point, Point, Point, Point],
  tagId: number,
  supersample: number,
) {
  return decodeStressSynthetic(w, h, strip, tagId, supersample, DECODE_STRESS_SPECKLE_AMP)
}

/** Count 6×6 cells where decoded disagrees with ground-truth pattern (ignores unknowns `-1`/`-2`). */
function cellErrorsVsTruth(decoded: (0 | 1 | -1 | -2)[], truth: TagPattern): number {
  let err = 0
  for (let i = 0; i < 36; i++) {
    const d = decoded[i]
    if (d === -1 || d === -2) {
      continue
    }
    if (d !== truth[i]) {
      err++
    }
  }
  return err
}

describe('decodeTagPattern stress (perspective + low resolution)', () => {
  const tagId = 0

  it('axis-aligned tag decodes at 48×48 (baseline low-res)', () => {
    const w = 48
    const h = 48
    const side = 32
    const strip = decodeStressAxisStrip(w, h, 4, side)
    const truthPat = codeToPattern(TAG36H11_CODES[tagId])
    const { intensity, sobel } = decodeStressRasterSobel(
      w,
      h,
      strip,
      truthPat,
      STRESS_SUPERSAMPLE,
      tagId,
      DECODE_STRESS_SPECKLE_AMP,
    )
    const grid = buildTagGrid(decodeStressCornersGridOrder(strip), 6)
    const decodedPattern = decodeTagPattern(grid, sobel, w, undefined, h)

    attachFailureArtifacts(THIS_FILE, (dir) => {
      writeGreyPng(join(dir, 'intensity.png'), w, h, intensity)
      writeSobelMagPng(join(dir, 'sobelMag.png'), w, h, sobel)
      if (decodedPattern) {
        writeCellLegendPng(join(dir, 'cells-rgb.png'), decodedPattern, truthPat, Math.max(12, Math.floor(240 / 6)))
      }
    })
    const { rot, best } = decodeSynthetic(w, h, strip, tagId, STRESS_SUPERSAMPLE)
    expect(rot).not.toBeNull()
    expect(rot!.id).toBe(tagId)
    expect(best.id).toBe(tagId)
    expect(best.dist).toBe(1)
  })

  it('strong perspective at 120×120: exact pattern + dictionary', () => {
    const w = 120
    const h = 120
    const strip = decodeStressFitPerspectiveStrip(w, h)
    const truth = codeToPattern(TAG36H11_CODES[tagId])
    const { intensity, sobel } = decodeStressRasterSobel(
      w,
      h,
      strip,
      truth,
      STRESS_SUPERSAMPLE,
      tagId,
      DECODE_STRESS_SPECKLE_AMP,
    )
    const grid = buildTagGrid(decodeStressCornersGridOrder(strip), 6)
    const decodedPattern = decodeTagPattern(grid, sobel, w, undefined, h)

    attachFailureArtifacts(THIS_FILE, (dir) => {
      writeGreyPng(join(dir, 'intensity.png'), w, h, intensity)
      writeSobelMagPng(join(dir, 'sobelMag.png'), w, h, sobel)
      if (decodedPattern) {
        writeCellLegendPng(join(dir, 'cells-rgb.png'), decodedPattern, truth, Math.max(12, Math.floor(240 / 6)))
      }
    })
    const { rot, best, decodedPattern: dp } = decodeSynthetic(w, h, strip, tagId, STRESS_SUPERSAMPLE)
    expect(rot).not.toBeNull()
    expect(rot!.id).toBe(tagId)
    expect(best.dist).toBeLessThanOrEqual(1)
    expect(cellErrorsVsTruth(dp, truth)).toBe(0)
  })

  it('strong perspective at 72×72: exact pattern + dictionary', () => {
    const w = 72
    const h = 72
    const strip = decodeStressFitPerspectiveStrip(w, h)
    const truth = codeToPattern(TAG36H11_CODES[tagId])
    const { intensity, sobel } = decodeStressRasterSobel(
      w,
      h,
      strip,
      truth,
      STRESS_SUPERSAMPLE,
      tagId,
      DECODE_STRESS_SPECKLE_AMP,
    )
    const grid = buildTagGrid(decodeStressCornersGridOrder(strip), 6)
    const decodedPattern = decodeTagPattern(grid, sobel, w, undefined, h)

    attachFailureArtifacts(THIS_FILE, (dir) => {
      writeGreyPng(join(dir, 'intensity.png'), w, h, intensity)
      writeSobelMagPng(join(dir, 'sobelMag.png'), w, h, sobel)
      if (decodedPattern) {
        writeCellLegendPng(join(dir, 'cells-rgb.png'), decodedPattern, truth, Math.max(12, Math.floor(240 / 6)))
      }
    })
    const { rot, best, decodedPattern: dp } = decodeSynthetic(w, h, strip, tagId, STRESS_SUPERSAMPLE)
    expect(rot).not.toBeNull()
    expect(rot!.id).toBe(tagId)
    expect(best.dist).toBe(1)
    expect(cellErrorsVsTruth(dp, truth)).toBe(1)
  })

  describe('homography mismatch (perturbed decode corners, H recomputed)', () => {
    for (const wh of DECODE_STRESS_SIZES) {
      it(`decode at ${wh}×${wh} (id 0; exact or one-cell slack)`, () => {
        const truth = codeToPattern(TAG36H11_CODES[tagId])
        const rasterStrip = decodeStressFitPerspectiveStrip(wh, wh)
        const decodeStrip = decodeStressStripWithHomographyMismatchOffsetsPx(rasterStrip)
        const { intensity, sobel } = decodeStressRasterSobel(
          wh,
          wh,
          rasterStrip,
          truth,
          STRESS_SUPERSAMPLE,
          tagId,
          DECODE_STRESS_SPECKLE_AMP,
        )
        const grid = buildTagGrid(decodeStressCornersGridOrder(decodeStrip), 6)
        const decodedPattern = decodeTagPattern(grid, sobel, wh, undefined, wh)

        attachFailureArtifacts(THIS_FILE, (dir) => {
          writeGreyPng(join(dir, 'intensity.png'), wh, wh, intensity)
          writeSobelMagPng(join(dir, 'sobelMag.png'), wh, wh, sobel)
          if (decodedPattern) {
            writeCellLegendPng(join(dir, 'cells-rgb.png'), decodedPattern, truth, Math.max(12, Math.floor(240 / 6)))
          }
        })
        const {
          rot,
          best,
          decodedPattern: dp,
        } = decodeStressSyntheticWithHomographyMismatch(
          wh,
          wh,
          rasterStrip,
          decodeStrip,
          tagId,
          STRESS_SUPERSAMPLE,
          DECODE_STRESS_SPECKLE_AMP,
        )
        expect(rot?.id).toBe(tagId)
        expect(best.dist).toBeLessThanOrEqual(2)
        expect(cellErrorsVsTruth(dp, truth)).toBeLessThanOrEqual(2)
        expect(dp.filter((v) => v === -1 || v === -2).length).toBeLessThanOrEqual(2)
      })
    }
  })

  /**
   * Snapshot table: same **strong perspective** quad scaled into square canvases.
   * `dist` = Hamming vs best codeword (known bits); `cellErr` = wrong 6×6 cells vs raster truth (excl. `-1`);
   * `unknowns` = count of `-1` (few votes) and `-2` (tie) from `classifyModuleFromPosNeg`. Rising `unknowns` / `dist` at low `wh` marks where decode starts to slip for this raster settings.
   */
  it('matches perspective + resolution characterization table', () => {
    const truth = codeToPattern(TAG36H11_CODES[tagId])
    const sizes = [...DECODE_STRESS_SIZES]
    const truthPat = codeToPattern(TAG36H11_CODES[tagId])

    const table = sizes.map((wh) => {
      const strip = decodeStressFitPerspectiveStrip(wh, wh)
      const { best, rot, decodedPattern } = decodeSynthetic(wh, wh, strip, tagId, STRESS_SUPERSAMPLE)
      const unknowns = decodedPattern.filter((v) => v === -1 || v === -2).length
      return {
        wh,
        id: rot?.id ?? -1,
        dist: best.dist,
        rotation: rot?.rotation ?? -1,
        cellErr: cellErrorsVsTruth(decodedPattern, truth),
        unknowns,
      }
    })

    attachFailureArtifacts(THIS_FILE, (dir) => {
      for (const wh of sizes) {
        const strip = decodeStressFitPerspectiveStrip(wh, wh)
        const pattern = truthPat
        const { intensity, sobel } = decodeStressRasterSobel(
          wh,
          wh,
          strip,
          pattern,
          STRESS_SUPERSAMPLE,
          tagId,
          DECODE_STRESS_SPECKLE_AMP,
        )
        const grid = buildTagGrid(decodeStressCornersGridOrder(strip), 6)
        const decodedPattern = decodeTagPattern(grid, sobel, wh, undefined, wh)
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
        writeGreyPng(join(dir, `${tag}-intensity.png`), wh, wh, intensity)
        writeSobelMagPng(join(dir, `${tag}-sobelMag.png`), wh, wh, sobel)
        writeCellLegendPng(
          join(dir, `${tag}-cells-rgb.png`),
          decodedPattern,
          truthPat,
          Math.max(12, Math.floor(240 / 6)),
        )
      }
    })
    expect(table).toMatchInlineSnapshot(`
      [
        {
          "cellErr": 1,
          "dist": 1,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 200,
        },
        {
          "cellErr": 0,
          "dist": 0,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 160,
        },
        {
          "cellErr": 0,
          "dist": 0,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 120,
        },
        {
          "cellErr": 1,
          "dist": 1,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 96,
        },
        {
          "cellErr": 0,
          "dist": 0,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 80,
        },
        {
          "cellErr": 1,
          "dist": 1,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 72,
        },
        {
          "cellErr": 1,
          "dist": 1,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 64,
        },
        {
          "cellErr": 0,
          "dist": 0,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 56,
        },
        {
          "cellErr": 0,
          "dist": 0,
          "id": 0,
          "rotation": 0,
          "unknowns": 1,
          "wh": 48,
        },
        {
          "cellErr": 2,
          "dist": 2,
          "id": 0,
          "rotation": 0,
          "unknowns": 0,
          "wh": 40,
        },
      ]
    `)
  })
})
