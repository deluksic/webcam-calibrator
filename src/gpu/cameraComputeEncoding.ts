import type { TgpuRoot } from 'typegpu'

import type { FrameSlot } from '@/gpu/frameSlotPool'
import { MAX_INSTANCES } from '@/gpu/pipelines/gridVizPipeline'

import type { CameraPipeline } from './cameraPipeline'

let nextFrameId = 0

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
  if (slot !== undefined) {
    pipeline.edgeHistogram.encodeCompute(computePass)
    pipeline.lineFit.encodeCompute(computePass)
    pipeline.quadHomography.encodeCompute(computePass)
  }
  computePass.end()

  if (slot !== undefined) {
    pipeline.tagDecode.encodeVotes(enc, MAX_INSTANCES)
    const decodePass = enc.beginComputePass({ label: 'camera tag decode' })
    pipeline.tagDecode.encodeDecode(decodePass)
    decodePass.end()

    pipeline.frameSlotPool.enqueueCopiesForSlot(enc, pipeline, slot)
    slot.frameId = nextFrameId++
    slot.state = 'inflight'
  }
}
