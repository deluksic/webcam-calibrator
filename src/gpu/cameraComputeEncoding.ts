import type { TgpuRoot } from 'typegpu'

import { prepareGpuProfiling, runComputeStage } from '@/gpu/gpuProfiling'
import { MAX_QUADS } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
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
): void {
  prepareGpuProfiling(root)

  pipeline.ingest.encodeIngest(enc, root, video)

  pipeline.nms.thresholdBuffer.write(threshold)
  pipeline.histogram.thresholdBinBuffer.write(Math.round(threshold * 255))

  runComputeStage(enc, 'gray', (p) => pipeline.gray.encodeCompute(p))
  runComputeStage(enc, 'sobel', (p) => pipeline.sobel.encodeCompute(p))
  if (pipeline.histogram.tickAccumFrame()) {
    runComputeStage(enc, 'histogram', (p) => pipeline.histogram.encodeAccumulateCompute(p))
  }
  runComputeStage(enc, 'nms', (p) => pipeline.nms.encodeCompute(p))
  runComputeStage(enc, 'pointer-jump', (p) => pipeline.pointerJump.encodeCompute(p))
  runComputeStage(enc, 'compact', (p) => pipeline.compact.encodeCompute(p))
  runComputeStage(enc, 'boundary-filter', (p) => pipeline.boundaryFilter.encodeCompute(p))

  if (slot !== undefined) {
    runComputeStage(enc, 'edge-histogram', (p) => pipeline.edgeHistogram.encodeCompute(p))
    runComputeStage(enc, 'line-fit', (p) => pipeline.lineFit.encodeCompute(p))
    runComputeStage(enc, 'quad-homography', (p) => pipeline.quadHomography.encodeCompute(p))

    pipeline.profile.encodeCompute(enc)
    runComputeStage(enc, 'profile-minmax', (p) => pipeline.profile.encodeMinMaxPass(p))

    pipeline.publishQuadCount.encodePublish(enc)
    pipeline.tagDecode.encodeVotePasses(enc, MAX_QUADS)
    runComputeStage(enc, 'tag-decode', (p) => {
      pipeline.tagDecode.encodeDecode(p, MAX_QUADS)
    })
    runComputeStage(enc, 'host-quad-pack', (p) => {
      pipeline.hostQuadReadback.encodePack(p, MAX_QUADS)
    })
  }
}
