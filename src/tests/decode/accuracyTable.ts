import {
  DECODE_STRESS_SPECKLE_AMP,
  decodeStressCornersGridOrder,
  decodeStressSpeckleSeed,
} from '@/lib/decodeStressHarness'
import { buildTagGrid, decodeTagPattern } from '@/lib/grid'
import {
  codeToPattern,
  decodeTag36h11AnyRotation,
  decodeTag36h11Best,
  TAG36H11_CODES,
  type TagPattern,
} from '@/lib/tag36h11'
import { buildRasterPack } from '@/tests/shared/pipeline'
import type { SceneSpec } from '@/tests/shared/types'

export function cellErrorsVsTruth(decoded: (0 | 1 | -1 | -2)[], truth: TagPattern): number {
  let err = 0
  for (let i = 0; i < 36; i++) {
    const d = decoded[i]
    if (d === -1 || d === -2) continue
    if (d !== truth[i]) err++
  }
  return err
}

export interface DecodeAccuracyRow {
  wh: number
  id: number
  dist: number
  rotation: number
  cellErr: number
  unknowns: number
}

export function perspectiveDecodeAccuracyRow(
  wh: number,
  tagId: number,
  supersample: number,
  truth: TagPattern,
): DecodeAccuracyRow {
  const scene: SceneSpec = {
    kind: 'perspective',
    width: wh,
    height: wh,
    tagId,
    supersample,
  }
  const pack = buildRasterPack({
    scene,
    noise: [
      {
        type: 'speckle',
        amplitude: DECODE_STRESS_SPECKLE_AMP,
        seed: decodeStressSpeckleSeed(wh, wh, tagId),
      },
    ],
  })
  const grid = buildTagGrid(decodeStressCornersGridOrder(pack.groundTruthStrip), 6)
  const decodedPattern = decodeTagPattern(grid, pack.sobel, wh, undefined, wh)
  if (!decodedPattern) {
    return { wh, id: -1, dist: -1, rotation: -1, cellErr: -1, unknowns: -1 }
  }
  const unknowns = decodedPattern.filter((v) => v === -1 || v === -2).length
  const best = decodeTag36h11Best(decodedPattern, 8)
  const rot = decodeTag36h11AnyRotation(decodedPattern, 8)
  return {
    wh,
    id: rot?.id ?? -1,
    dist: best.dist,
    rotation: rot?.rotation ?? -1,
    cellErr: cellErrorsVsTruth(decodedPattern, truth),
    unknowns,
  }
}

export function buildPerspectiveAccuracyTable(
  sizes: readonly number[],
  tagId: number,
  supersample: number,
): DecodeAccuracyRow[] {
  const truth = codeToPattern(TAG36H11_CODES[tagId]!)
  return sizes.map((wh) => perspectiveDecodeAccuracyRow(wh, tagId, supersample, truth))
}
