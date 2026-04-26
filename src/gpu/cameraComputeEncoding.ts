import type { TgpuRoot } from 'typegpu'

import type { FrameSlot } from '@/gpu/frameSlotPool'

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
  pipeline.extent.encodeCompute(computePass)
  computePass.end()

  if (slot !== undefined) {
    pipeline.frameSlotPool.enqueueCopiesForSlot(enc, pipeline, slot)
    slot.frameId = nextFrameId++
    slot.state = 'inflight'
  }
}
