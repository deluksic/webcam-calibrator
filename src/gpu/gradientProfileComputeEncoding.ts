import type { TgpuRoot } from 'typegpu'

import { MAX_QUADS } from '@/gpu/pipelines/edgeHistogramClusterPipeline'

import type { GradientProfileDisplayMode, GradientProfilePipeline } from './gradientProfilePipeline'

export type GradientProfileComputeOptions = {
  displayMode?: GradientProfileDisplayMode
  /** When true (e.g. Pause), re-run vision on the last ingested frame without copying video again. */
  skipIngest?: boolean
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
  if (!options.skipIngest) {
    pipeline.ingest.encodeIngest(enc, root, video)
  }

  pipeline.nms.thresholdBuffer.write(threshold)
  pipeline.histogram.thresholdBinBuffer.write(Math.round(threshold * 255))

  const computePass = enc.beginComputePass({ label: 'gradient profiles compute' })
  pipeline.gray.encodeCompute(computePass)
  pipeline.sobel.encodeCompute(computePass)
  pipeline.histogram.encodeAccumulateCompute(computePass)
  pipeline.nms.encodeCompute(computePass)
  pipeline.pointerJump.encodeCompute(computePass)
  pipeline.compact.encodeCompute(computePass)
  pipeline.boundaryFilter.encodeCompute(computePass)
  pipeline.edgeHistogram.encodeCompute(computePass)
  pipeline.lineRejects.encodeCompute(computePass)
  pipeline.orientHistViz?.encodePackCompute(computePass)
  pipeline.lineFit.encodeCompute(computePass)
  pipeline.quadHomography.encodeCompute(computePass)
  // Profile plot canvas is always presented; compute is not tied to camera display mode.
  pipeline.profile.encodeCompute(computePass)
  computePass.end()

  pipeline.publishQuadCount.encodePublish(enc)
  pipeline.tagDecode.encodeVotePasses(enc, MAX_QUADS)

  const decodePass = enc.beginComputePass({ label: 'tag decode' })
  pipeline.tagDecode.encodeDecode(decodePass, MAX_QUADS)
  decodePass.end()
}
