import type { TgpuRoot } from 'typegpu'

import type { GradientProfileDisplayMode, GradientProfilePipeline } from './gradientProfilePipeline'

export type GradientProfileComputeOptions = {
  /** Active quads from the previous frame's `quadCount` readback (same frame's homography). */
  quadCount?: number
  displayMode?: GradientProfileDisplayMode
}

function shouldRunProfile(displayMode: GradientProfileDisplayMode | undefined): boolean {
  return (
    displayMode === 'quadGrid' ||
    displayMode === 'fittedLines' ||
    displayMode === 'lineFitDebug'
  )
}

/**
 * Full gradient-profile compute chain for one frame.
 * `enc` must be submitted before the external texture expires.
 */
export function encodeGradientProfileCompute(
  enc: GPUCommandEncoder,
  root: TgpuRoot,
  pipeline: GradientProfilePipeline,
  video: HTMLVideoElement,
  threshold: number,
  options: GradientProfileComputeOptions = {},
): void {
  const quadCount = options.quadCount ?? 0
  const displayMode = options.displayMode

  pipeline.ingest.encodeIngest(enc, root, video)

  pipeline.nms.thresholdBuffer.write(threshold)
  pipeline.histogram.thresholdBinBuffer.write(Math.round(threshold * 255))

  const computePass = enc.beginComputePass({ label: 'gradient profiles compute' })
  pipeline.gray.encodeCompute(computePass)
  pipeline.sobel.encodeCompute(computePass)
  pipeline.histogram.encodeAccumulateCompute(computePass)
  pipeline.nms.encodeCompute(computePass)
  pipeline.pointerJump.encodeCompute(computePass)
  pipeline.compact.encodeCompute(computePass)
  pipeline.edgeHistogram.encodeCompute(computePass)
  if (displayMode === 'lineFitDebug') {
    pipeline.lineFitDebug.encodeCompute(computePass)
  }
  pipeline.orientHistViz?.encodePackCompute(computePass)
  pipeline.lineFit.encodeCompute(computePass)
  pipeline.quadHomography.encodeCompute(computePass)
  if (shouldRunProfile(displayMode)) {
    pipeline.profile.encodeCompute(computePass)
  }
  computePass.end()

  pipeline.tagDecode.encodeVotePasses(enc, quadCount)

  const decodePass = enc.beginComputePass({ label: 'tag decode' })
  pipeline.tagDecode.encodeDecode(decodePass, quadCount)
  decodePass.end()
}
