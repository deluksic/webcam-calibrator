import type { TgpuRoot } from 'typegpu'
import { d } from 'typegpu'

import type { DetectedQuad } from '@/gpu/contour'
import type { FrameSlot } from '@/gpu/frameSlotPool'
import {
  HISTOGRAM_BINS,
  COMPUTE_WORKGROUP_SIZE,
  POINTER_JUMP_ITERATIONS,
  computeDispatch2d,
} from '@/gpu/pipelines/constants'
import { createCopyBindGroup } from '@/gpu/pipelines/copyPipeline'
import {
  DECODED_TAG_ID_UNKNOWN,
  type QuadData,
  MAX_INSTANCES,
  type GridVizFailInterrogateMode,
} from '@/gpu/pipelines/gridVizPipeline'
import type { ReprojPairGpu } from '@/gpu/pipelines/reprojectionOverlayPipeline'
import { tryComputeHomography } from '@/lib/geometry'
import type { ReprojectionOverlayPair } from '@/lib/reprojectionLive'

import { MAX_DETECTED_TAGS, MAX_EXTENT_COMPONENTS, type CameraPipeline, type DisplayMode } from './cameraPipeline'

const { min, ceil, round } = Math

/**
 * Non-grid display modes that paint synchronously to the canvas.
 * `'grid'` is intentionally excluded — use `presentGridFrame` for that path,
 * which pairs the gray snapshot with its matching detection.
 */
export type NonGridDisplayMode = Exclude<DisplayMode, 'grid'>

let nextFrameId = 0

/**
 * Append the external-texture copy and the full compute chain to `enc`.
 * Does not submit, does not draw to the canvas.
 * After this call `pipeline.compact.compactLabelBuffer` and `pipeline.nms.filteredBuffer`
 * contain the results for this frame.
 *
 * When `slot` is provided the matching copies are also appended so that slot's
 * pinned buffers capture this frame's data. The slot is transitioned to
 * `'inflight'` and assigned a monotonically-increasing `frameId`.
 *
 * IMPORTANT: `enc` must be submitted in the same task (before returning to the
 * event loop) to avoid the GPUExternalTexture expiring.
 */
export function runCompute(
  enc: GPUCommandEncoder,
  root: TgpuRoot,
  pipeline: CameraPipeline,
  video: HTMLVideoElement,
  threshold: number,
  slot?: FrameSlot,
) {
  const copyBindGroup = createCopyBindGroup(root, video)

  pipeline.nms.thresholdBuffer.write(threshold)
  pipeline.histogram.thresholdBinBuffer.write(round(threshold * 255))

  // RENDER: Copy external → grayTex (MUST happen before compute)
  pipeline.ingest.copyPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.ingest.grayTex.createView() })
    .with(copyBindGroup)
    .draw(3)

  // COMPUTE: Gray → Sobel → Histogram + NMS filter + pointer-jump labeling → compact remap → extent tracking
  const computePass = enc.beginComputePass({ label: 'gray + sobel + histogram + filter' })
  const [wgX, wgY] = computeDispatch2d(pipeline.width, pipeline.height)
  const area = pipeline.width * pipeline.height

  pipeline.gray.pipeline.with(computePass).with(pipeline.gray.bindGroup).dispatchWorkgroups(wgX, wgY)
  pipeline.sobel.pipeline.with(computePass).with(pipeline.sobel.bindGroup).dispatchWorkgroups(wgX, wgY)
  pipeline.histogram.resetPipeline
    .with(computePass)
    .with(pipeline.histogram.resetBindGroup)
    .dispatchWorkgroups(HISTOGRAM_BINS)
  pipeline.histogram.computePipeline
    .with(computePass)
    .with(pipeline.histogram.computeBindGroup)
    .dispatchWorkgroups(wgX, wgY)
  pipeline.nms.pipeline.with(computePass).with(pipeline.nms.bindGroup).dispatchWorkgroups(wgX, wgY)

  // Pointer-jump connected component labeling
  pipeline.pointerJump.pointerJumpInitPipeline
    .with(computePass)
    .with(pipeline.pointerJump.pointerJumpInitBindGroup)
    .dispatchWorkgroups(wgX, wgY)
  let pj = 0
  for (let s = 0; s < POINTER_JUMP_ITERATIONS; s++) {
    pipeline.pointerJump.pointerJumpStepPipeline
      .with(computePass)
      .with(pipeline.pointerJump.pointerJumpPingPongBindGroups[pj]!)
      .dispatchWorkgroups(wgX, wgY)
    pj ^= 1
    pipeline.pointerJump.pointerJumpLabelsToAtomicPipeline
      .with(computePass)
      .with(pipeline.pointerJump.pointerJumpLabelsToAtomicBindGroups[pj]!)
      .dispatchWorkgroups(wgX, wgY)
    pipeline.pointerJump.pointerJumpParentTightenPipeline
      .with(computePass)
      .with(pipeline.pointerJump.pointerJumpParentTightenBindGroups[pj]!)
      .dispatchWorkgroups(wgX, wgY)
    pipeline.pointerJump.pointerJumpAtomicToLabelsPipeline
      .with(computePass)
      .with(pipeline.pointerJump.pointerJumpAtomicToLabelsBindGroups[pj]!)
      .dispatchWorkgroups(wgX, wgY)
  }

  // Compact labeling: 3-pass remap of pixel-index labels to compact IDs (0..N-1).
  // Always runs — needed for extent tracking (labels must fit in extent buffer).
  // Reads pointerJumpBuffer0, which holds the converged result because
  // POINTER_JUMP_ITERATIONS is even (pj === 0 on exit).
  pipeline.compact.compactResetPipeline
    .with(computePass)
    .with(pipeline.compact.compactResetBindGroup)
    .dispatchWorkgroups(ceil(area / COMPUTE_WORKGROUP_SIZE))
  pipeline.compact.compactClaimPipeline
    .with(computePass)
    .with(pipeline.compact.compactClaimBindGroup)
    .dispatchWorkgroups(wgX, wgY)
  pipeline.compact.compactRemapPipeline
    .with(computePass)
    .with(pipeline.compact.compactRemapBindGroup)
    .dispatchWorkgroups(wgX, wgY)

  // Extent tracking on compact labels
  pipeline.extent.extentResetPipeline
    .with(computePass)
    .with(pipeline.extent.extentResetBindGroup)
    .dispatchWorkgroups(ceil(MAX_EXTENT_COMPONENTS / COMPUTE_WORKGROUP_SIZE))
  pipeline.extent.extentTrackPipeline
    .with(computePass)
    .with(pipeline.extent.extentTrackBindGroup)
    .dispatchWorkgroups(wgX, wgY)

  computePass.end()

  // If a slot was provided, pin this frame's buffers into it for async readback.
  if (slot !== undefined) {
    pipeline.frameSlotPool.enqueueCopiesForSlot(enc, pipeline, slot)
    slot.frameId = nextFrameId++
    slot.state = 'inflight'
  }
}

/**
 * Append display-mode render passes and the histogram draw to `enc`.
 * After this the caller should submit `enc`.
 * `pipeline.compact.compactLabelBuffer` is used for label/debug modes (always the
 * compact output after `runCompute`).
 *
 * Only non-grid modes are accepted. Grid mode is handled by `presentGridFrame`
 * so that the gray snapshot and the overlay always originate from the same frame.
 */
export function presentFrame(
  enc: GPUCommandEncoder,
  root: TgpuRoot,
  pipeline: CameraPipeline,
  displayMode: NonGridDisplayMode,
  onError?: (msg: string) => void,
) {
  if (displayMode === 'edges') {
    try {
      pipeline.render.sobel.pipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(pipeline.render.sobel.bindGroup)
        .draw(3)
    } catch (e) {
      const msg = `[camera] sobelRender failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  } else if (displayMode === 'nms') {
    try {
      pipeline.render.edges.pipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(pipeline.render.edges.bindGroup)
        .draw(3)
    } catch (e) {
      const msg = `[camera] edgesPipeline (nms) failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  } else if (displayMode === 'labels' || displayMode === 'debug') {
    const labelVizBindGroup = root.createBindGroup(pipeline.render.labelViz.layout, {
      labelBuffer: pipeline.compact.compactLabelBuffer,
    })
    try {
      pipeline.render.labelViz.pipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(labelVizBindGroup)
        .draw(3)
    } catch (e) {
      const msg = `[camera] labelVizPipeline (${displayMode}) failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  } else {
    try {
      pipeline.render.grayscale.pipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(pipeline.render.grayscale.bindGroup)
        .draw(3)
    } catch (e) {
      const msg = `[camera] grayRender fallback failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  }

  if (pipeline.histContext) {
    pipeline.histogram.displayPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.histContext })
      .with(pipeline.histogram.displayBindGroup)
      .draw(6, HISTOGRAM_BINS)
  }
}

/** Write homography matrix (mat3x3) + debug data per quad to the GPU buffer.
 * H maps unit square → detected quad. Vertex shader applies: (x,y,z) = H * (u,v,1).
 * H is column-major mat3x3f: [c0.x, c0.y, c0.z, 0, c1..., c2...]
 */
export function updateQuadCornersBuffer(
  pipeline: CameraPipeline,
  quads: DetectedQuad[],
  showFallbacks: boolean = true,
): void {
  const filtered = showFallbacks ? quads : quads.filter((q) => q.hasCorners && typeof q.decodedTagId === 'number')
  const count = min(filtered.length, MAX_INSTANCES)

  const data: QuadData[] = []
  for (let i = 0; i < count; i++) {
    const quad = filtered[i]!
    const H = tryComputeHomography(quad.corners)
    const debug = quad.cornerDebug
    const tagId = quad.vizTagId !== undefined ? quad.vizTagId >>> 0 : DECODED_TAG_ID_UNKNOWN

    data.push({
      homography: H
        ? d.mat3x3f(
            // transpose the matrix (row-major Mat3 → column-major GPU)
            H[0],
            H[3],
            H[6],
            H[1],
            H[4],
            H[7],
            H[2],
            H[5],
            H[8],
          )
        : d.mat3x3f(0, 0, 0, 0, 0, 0, 0, 0, 1),
      debug: {
        failureCode: debug ? debug.failureCode : 0,
        edgePixelCount: debug ? debug.edgePixelCount / 100 : 0,
        minR2: debug ? debug.minR2 : 0,
        intersectionCount: debug ? debug.intersectionCount : 0,
      },
      decodedTagId: d.u32(tagId),
    })
  }

  // Pad remaining slots with zeros
  for (let i = count; i < MAX_INSTANCES; i++) {
    data.push({
      homography: d.mat3x3f(0, 0, 0, 0, 0, 0, 0, 0, 1),
      debug: {
        failureCode: 0,
        edgePixelCount: 0,
        minR2: 0,
        intersectionCount: 0,
      },
      decodedTagId: d.u32(DECODED_TAG_ID_UNKNOWN),
    })
  }

  pipeline.grid.quadCornersBuffer.write(data)
}

/** Upload `{ original, reprojected }` pairs for the GPU reprojection overlay; pads to `MAX_INSTANCES`. */
export function updateReprojectionOverlayBuffer(
  pipeline: CameraPipeline,
  pairs: ReprojectionOverlayPair[],
  count: number,
): void {
  const capped = min(count, MAX_INSTANCES)
  const data: ReprojPairGpu[] = []
  for (let i = 0; i < capped; i++) {
    const p = pairs[i]!
    data.push({
      original: d.vec2f(p.original.x, p.original.y),
      reprojected: d.vec2f(p.reprojected.x, p.reprojected.y),
    })
  }
  const dead = d.vec2f(-1, -1)
  for (let i = capped; i < MAX_INSTANCES; i++) {
    data.push({ original: dead, reprojected: dead })
  }
  pipeline.reproj.reprojOverlayBuffer.write(data)
  pipeline.reproj.reprojOverlayDrawState.instanceCount = capped
}

/** Grid overlay: 0 = legacy fail colors, 1 = red highlights insufficient-edge failures, 2 = blue highlights line-fit failures. */
export function setGridVizFailInterrogate(pipeline: CameraPipeline, mode: GridVizFailInterrogateMode): void {
  pipeline.grid.gridVizDebugModeBuffer.write(mode)
}

/**
 * Encode and submit a command buffer that renders the pinned gray snapshot
 * from `slot` plus the current grid viz overlay and histogram.
 * Call this right after `updateQuadCornersBuffer` so the write and draw are
 * in the same synchronous block.
 */
export function presentGridFrame(root: TgpuRoot, pipeline: CameraPipeline, slot: FrameSlot): void {
  const enc = root.device.createCommandEncoder({ label: 'grid frame present' })

  pipeline.render.grayscale.pipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.context, loadOp: 'load', storeOp: 'store' })
    .with(slot.grayRenderBindGroup)
    .draw(3)

  try {
    pipeline.grid.gridVizPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context, loadOp: 'load', storeOp: 'store' })
      .with(pipeline.grid.gridVizBindGroup)
      .draw(4, MAX_DETECTED_TAGS)
  } catch (e) {
    console.error('[presentGridFrame] gridViz failed:', e)
  }

  const reprojN = pipeline.reproj.reprojOverlayDrawState.instanceCount
  if (reprojN > 0) {
    const load = { view: pipeline.context, loadOp: 'load' as const, storeOp: 'store' as const }
    try {
      pipeline.reproj.reprojOriginalPipeline
        .with(enc)
        .withColorAttachment(load)
        .with(pipeline.reproj.reprojOverlayBindGroup)
        .draw(4, reprojN)
      pipeline.reproj.reprojTargetPipeline
        .with(enc)
        .withColorAttachment(load)
        .with(pipeline.reproj.reprojOverlayBindGroup)
        .draw(4, reprojN)
      pipeline.reproj.reprojLinesPipeline
        .with(enc)
        .withColorAttachment(load)
        .with(pipeline.reproj.reprojOverlayBindGroup)
        .draw(2, reprojN)
    } catch (e) {
      console.error('[presentGridFrame] reprojection overlay failed:', e)
    }
  }

  if (pipeline.histContext) {
    pipeline.histogram.displayPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.histContext })
      .with(pipeline.histogram.displayBindGroup)
      .draw(6, HISTOGRAM_BINS)
  }

  root.device.queue.submit([enc.finish()])
}
