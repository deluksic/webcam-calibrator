import type { d, TgpuRoot } from 'typegpu'

import { type DetectedQuad, extractRegions, validateAndFilterQuads } from '@/gpu/contour'
import type { FrameSlot } from '@/gpu/frameSlotPool'
import type { ExtentEntry } from '@/gpu/pipelines/extentTrackingPipeline'
import { MAX_U32 } from '@/gpu/pipelines/extentTrackingPipeline'

import type { CameraPipeline } from './cameraPipeline'

export type ExtentRow = d.Infer<typeof ExtentEntry>

/**
 * Read extent buffer and filter to valid quad candidates.
 * Uses the same bboxes that appear in the debug view.
 */
export async function readExtentDataForQuads(pipeline: CameraPipeline): Promise<ExtentRow[]> {
  const all = await pipeline.extent.extentBuffer.read()
  return all.filter((e) => e.minX !== MAX_U32)
}

/**
 * Append copies of `compactLabelBuffer` and `filteredBuffer` into the given
 * staging buffers. The encoder must be submitted by the caller before calling
 * `readDetection`.
 */
export function enqueueReadbackCopies(
  enc: GPUCommandEncoder,
  pipeline: CameraPipeline,
  labelStaging: GPUBuffer,
  filteredStaging: GPUBuffer,
) {
  const labelStorage = pipeline.compact.compactLabelBuffer.buffer
  enc.copyBufferToBuffer(labelStorage, 0, labelStaging, 0, labelStorage.size)
  const edgeStorage = pipeline.nms.filteredBuffer.buffer
  enc.copyBufferToBuffer(edgeStorage, 0, filteredStaging, 0, edgeStorage.size)
}

/**
 * After the encoder containing the readback copies has been submitted, await
 * GPU completion, map both staging buffers, run CPU region extraction + quad
 * fitting, and read the extent buffer. Returns the same shape as the old
 * `detectContours`.
 */
export async function readDetection(
  root: TgpuRoot,
  pipeline: CameraPipeline,
  labelStaging: GPUBuffer,
  filteredStaging: GPUBuffer,
): Promise<{
  quads: DetectedQuad[]
  extentData: ExtentRow[]
  dilatedGradients: Float32Array
  labelData: Uint32Array
}> {
  await root.device.queue.onSubmittedWorkDone()

  await labelStaging.mapAsync(GPUMapMode.READ)
  const labelDataCopy = new Uint32Array(new Uint32Array(labelStaging.getMappedRange()))
  labelStaging.unmap()

  await filteredStaging.mapAsync(GPUMapMode.READ)
  const dilatedCopy = new Float32Array(new Float32Array(filteredStaging.getMappedRange()))
  filteredStaging.unmap()

  const regions = extractRegions(labelDataCopy, pipeline.width, pipeline.height, dilatedCopy)
  const maxArea = pipeline.width * pipeline.height * 0.5
  const quads = validateAndFilterQuads(regions, dilatedCopy, labelDataCopy, pipeline.width, 400, maxArea).filter(
    (q) => q.area < pipeline.width * pipeline.height * 0.25,
  )

  const extentData: ExtentRow[] = await pipeline.extent.extentBuffer.read()

  return { quads, extentData, dilatedGradients: dilatedCopy, labelData: labelDataCopy }
}

/**
 * Read detection results from a slot whose copies were already enqueued by
 * `runCompute`. Returns the quads, extents, and the slot itself so the caller
 * can call `swapDisplaySlot` and `encodeAndSubmitGridPresent` after.
 */
export async function detectForSlot(
  root: TgpuRoot,
  pipeline: CameraPipeline,
  slot: FrameSlot,
): Promise<{
  quads: ReturnType<typeof validateAndFilterQuads>
  extentData: ExtentRow[]
  dilatedGradients: Float32Array
  labelData: Uint32Array
  frameId: number
  slot: FrameSlot
}> {
  const { quads, extentData, dilatedGradients, labelData } = await readDetection(
    root,
    pipeline,
    slot.labelStaging,
    slot.filteredStaging,
  )
  return { quads, extentData, dilatedGradients, labelData, frameId: slot.frameId, slot }
}
