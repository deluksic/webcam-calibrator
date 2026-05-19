import type { TgpuRoot } from 'typegpu'

import type { FrameSlot } from '@/gpu/frameSlotPool'

import type { CameraPipeline } from './cameraPipeline'

/**
 * Append external-texture ingest and the full camera compute chain to `enc`.
 * Does not submit. After this, compact labels and NMS filtered buffer hold this frame.
 *
 * IMPORTANT: `enc` must be submitted in the same task (before returning to the
 * event loop) so the GPUExternalTexture does not expire.
 */
export function encodeCameraCompute(
  enc: GPUCommandEncoder,
  root: TgpuRoot,
  pipeline: CameraPipeline,
  video: HTMLVideoElement,
  threshold: number,
  slot?: FrameSlot,
  voteInstanceCount: number = 0,
): void {
  pipeline.ingest.encodeIngest(enc, root, video)

  pipeline.nms.thresholdBuffer.write(threshold)
  pipeline.histogram.thresholdBinBuffer.write(Math.round(threshold * 255))

  const computePass = enc.beginComputePass({ label: 'gray + sobel + histogram + filter' })
  pipeline.gray.encodeCompute(computePass)
  pipeline.sobel.encodeCompute(computePass)
  pipeline.histogram.encodeAccumulateCompute(computePass)
  pipeline.nms.encodeCompute(computePass)
  pipeline.pointerJump.encodeCompute(computePass)
  pipeline.compact.encodeCompute(computePass)
  pipeline.boundaryFilter.encodeCompute(computePass)
  if (slot !== undefined) {
    pipeline.edgeHistogram.encodeCompute(computePass)
    pipeline.lineFit.encodeCompute(computePass)
    pipeline.quadHomography.encodeCompute(computePass)
  }
  computePass.end()

  if (slot !== undefined) {
    if (voteInstanceCount > 0) {
      pipeline.tagDecode.encodeVotePasses(enc, voteInstanceCount)
    }
    const decodePass = enc.beginComputePass({ label: 'camera tag decode' })
    pipeline.tagDecode.encodeDecode(decodePass, voteInstanceCount)
    pipeline.hostQuadReadback.encodePack(decodePass, voteInstanceCount)
    decodePass.end()
  }
}
