import type { TgpuRoot } from 'typegpu'

import { MAX_INSTANCES } from '@/gpu/pipelines/gridVizPipeline'
import type { GradientProfilePipeline } from './gradientProfilePipeline'

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
): void {
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
  pipeline.lineFitDebug.encodeCompute(computePass)
  pipeline.orientHistViz?.encodePackCompute(computePass)
  pipeline.lineFit.encodeCompute(computePass)
  pipeline.quadHomography.encodeCompute(computePass)
  pipeline.profile.encodeCompute(computePass)
  computePass.end()

  // Tag decode: render pass fills quads → fragment shader accumulates votes → compute pass decodes
  pipeline.tagDecode.encodeVotes(enc, MAX_INSTANCES)

  const decodePass = enc.beginComputePass({ label: 'tag decode' })
  pipeline.tagDecode.encodeDecode(decodePass)
  decodePass.end()
}
