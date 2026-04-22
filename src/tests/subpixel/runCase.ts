import { computeHomography } from '@/lib/geometry'
import type { SubpixelRefineInput } from '@/lib/subpixelRefinement'
import { buildRasterPack } from '@/tests/shared/pipeline'
import type { BuildImageOptions, StripCorners } from '@/tests/shared/types'
import { applyInitialRoughStrip, type InitialSpec } from '@/tests/subpixel/initialEstimate'
import { roundCornerStats, cornerErrorStats } from '@/tests/subpixel/metrics/corners'
import { homographyTransferStats, roundTransferStats } from '@/tests/subpixel/metrics/homographyTransfer'
import { stripCornersFromHomography8 } from '@/tests/subpixel/quadCpu'
import type { SubpixelRefineFn } from '@/tests/subpixel/types'

export interface RunSubpixelCaseArgs extends BuildImageOptions {
  initial: InitialSpec
  refiner: SubpixelRefineFn
  /** Passed to refiner when synthetic decode is known. */
  decodedTagId?: number
}

export interface SubpixelCaseResult {
  width: number
  height: number
  grayscale: Float32Array
  groundTruthStrip: StripCorners
  roughStrip: StripCorners
  H_gt: Float32Array
  H_init: Float32Array
  H_refined: Float32Array
  metrics: {
    cornersRounded: ReturnType<typeof roundCornerStats>
    transferRounded: ReturnType<typeof roundTransferStats>
  }
}

export function runSubpixelAlignmentCase(args: RunSubpixelCaseArgs): SubpixelCaseResult {
  const pack = buildRasterPack({
    scene: args.scene,
    noise: args.noise,
    radialDistortion: args.radialDistortion,
  })
  const gtStrip = pack.groundTruthStrip
  const H_gt = computeHomography(gtStrip)
  const roughStrip = applyInitialRoughStrip(gtStrip, args.initial)
  const H_init = computeHomography(roughStrip)

  const refineIn: SubpixelRefineInput = {
    width: pack.width,
    height: pack.height,
    grayscale: pack.grayscale,
    homography8: H_init,
    decodedTagId: args.decodedTagId,
  }
  const H_refined = args.refiner(refineIn)
  const refinedStrip = stripCornersFromHomography8(H_refined)

  const cornerGt = cornerErrorStats(gtStrip, refinedStrip)
  const transfer = homographyTransferStats(H_gt, H_refined)

  return {
    width: pack.width,
    height: pack.height,
    grayscale: pack.grayscale,
    groundTruthStrip: gtStrip,
    roughStrip,
    H_gt,
    H_init,
    H_refined,
    metrics: {
      cornersRounded: roundCornerStats(cornerGt),
      transferRounded: roundTransferStats(transfer),
    },
  }
}
