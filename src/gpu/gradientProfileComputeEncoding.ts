import type { TgpuRoot } from 'typegpu'

import { prepareGpuProfiling, runComputeStage } from '@/gpu/gpuProfiling'
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
  prepareGpuProfiling(root)

  if (!options.skipIngest) {
    pipeline.ingest.encodeIngest(enc, root, video)
  }

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
  runComputeStage(enc, 'edge-histogram', (p) => pipeline.edgeHistogram.encodeCompute(p))
  runComputeStage(enc, 'line-rejects', (p) => pipeline.lineRejects.encodeCompute(p))
  if (pipeline.orientHistViz) {
    runComputeStage(enc, 'orient-hist-pack', (p) => pipeline.orientHistViz!.encodePackCompute(p))
  }
  runComputeStage(enc, 'line-fit', (p) => pipeline.lineFit.encodeCompute(p))
  runComputeStage(enc, 'quad-homography', (p) => pipeline.quadHomography.encodeCompute(p))
  runComputeStage(enc, 'profile', (p) => pipeline.profile.encodeCompute(p))

  pipeline.publishQuadCount.encodePublish(enc)
  pipeline.tagDecode.encodeVotePasses(enc, MAX_QUADS)

  runComputeStage(enc, 'tag-decode', (p) => pipeline.tagDecode.encodeDecode(p, MAX_QUADS))
}
