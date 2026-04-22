import { tgpu, d } from 'typegpu'

import type { DetectedQuad } from '@/gpu/contour'
import type { FrameSlot } from '@/gpu/frameSlotPool'
import {
  HISTOGRAM_BINS,
  COMPUTE_WORKGROUP_SIZE,
  POINTER_JUMP_ITERATIONS,
  computeDispatch2d,
} from '@/gpu/pipelines/constants'
import {
  DECODED_TAG_ID_UNKNOWN,
  type QuadData,
  MAX_INSTANCES,
  type GridVizFailInterrogateMode,
} from '@/gpu/pipelines/gridVizPipeline'
import { tryComputeHomography } from '@/lib/geometry'

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
 * After this call `pipeline.compactLabelBuffer` and `pipeline.filteredBuffer`
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
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  video: HTMLVideoElement,
  threshold: number,
  slot?: FrameSlot,
) {
  const copyBindGroup = root.createBindGroup(pipeline.copyLayoutTemplate, {
    cameraTex: root.device.importExternalTexture({ source: video }),
    sampler: pipeline.sampler,
  })

  pipeline.thresholdBuffer.write(threshold)
  pipeline.thresholdBinBuffer.write(round(threshold * 255))

  // RENDER: Copy external → grayTex (MUST happen before compute)
  pipeline.copyPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.grayTex.createView() })
    .with(copyBindGroup)
    .draw(3)

  // COMPUTE: Gray → Sobel → Histogram + NMS filter + pointer-jump labeling → compact remap → extent tracking
  const computePass = enc.beginComputePass({ label: 'gray + sobel + histogram + filter' })
  const [wgX, wgY] = computeDispatch2d(pipeline.width, pipeline.height)
  const area = pipeline.width * pipeline.height

  pipeline.grayPipeline.with(computePass).with(pipeline.grayTexToBufferBindGroup).dispatchWorkgroups(wgX, wgY)
  pipeline.sobelPipeline.with(computePass).with(pipeline.sobelBindGroup).dispatchWorkgroups(wgX, wgY)
  pipeline.histogramResetPipeline
    .with(computePass)
    .with(pipeline.histogramResetBindGroup)
    .dispatchWorkgroups(HISTOGRAM_BINS)
  pipeline.histogramPipeline.with(computePass).with(pipeline.histogramComputeBindGroup).dispatchWorkgroups(wgX, wgY)
  pipeline.edgeFilterPipeline.with(computePass).with(pipeline.edgeFilterBindGroup).dispatchWorkgroups(wgX, wgY)

  // Pointer-jump connected component labeling
  pipeline.pointerJumpInitPipeline
    .with(computePass)
    .with(pipeline.pointerJumpInitBindGroup)
    .dispatchWorkgroups(wgX, wgY)
  let pj = 0
  for (let s = 0; s < POINTER_JUMP_ITERATIONS; s++) {
    pipeline.pointerJumpStepPipeline
      .with(computePass)
      .with(pipeline.pointerJumpPingPongBindGroups[pj]!)
      .dispatchWorkgroups(wgX, wgY)
    pj ^= 1
    pipeline.pointerJumpLabelsToAtomicPipeline
      .with(computePass)
      .with(pipeline.pointerJumpLabelsToAtomicBindGroups[pj]!)
      .dispatchWorkgroups(wgX, wgY)
    pipeline.pointerJumpParentTightenPipeline
      .with(computePass)
      .with(pipeline.pointerJumpParentTightenBindGroups[pj]!)
      .dispatchWorkgroups(wgX, wgY)
    pipeline.pointerJumpAtomicToLabelsPipeline
      .with(computePass)
      .with(pipeline.pointerJumpAtomicToLabelsBindGroups[pj]!)
      .dispatchWorkgroups(wgX, wgY)
  }

  // Compact labeling: 3-pass remap of pixel-index labels to compact IDs (0..N-1).
  // Always runs — needed for extent tracking (labels must fit in extent buffer).
  // Reads pointerJumpBuffer0, which holds the converged result because
  // POINTER_JUMP_ITERATIONS is even (pj === 0 on exit).
  pipeline.compactResetPipeline
    .with(computePass)
    .with(pipeline.compactResetBindGroup)
    .dispatchWorkgroups(ceil(area / COMPUTE_WORKGROUP_SIZE))
  pipeline.compactClaimPipeline.with(computePass).with(pipeline.compactClaimBindGroup).dispatchWorkgroups(wgX, wgY)
  pipeline.compactRemapPipeline.with(computePass).with(pipeline.compactRemapBindGroup).dispatchWorkgroups(wgX, wgY)

  // Extent tracking on compact labels
  pipeline.extentResetPipeline
    .with(computePass)
    .with(pipeline.extentResetBindGroup)
    .dispatchWorkgroups(ceil(MAX_EXTENT_COMPONENTS / COMPUTE_WORKGROUP_SIZE))
  pipeline.extentTrackPipeline.with(computePass).with(pipeline.extentTrackBindGroup).dispatchWorkgroups(wgX, wgY)

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
 * `pipeline.compactLabelBuffer` is used for label/debug modes (always the
 * compact output after `runCompute`).
 *
 * Only non-grid modes are accepted. Grid mode is handled by `presentGridFrame`
 * so that the gray snapshot and the overlay always originate from the same frame.
 */
export function presentFrame(
  enc: GPUCommandEncoder,
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  displayMode: NonGridDisplayMode,
  onError?: (msg: string) => void,
) {
  if (displayMode === 'edges') {
    try {
      pipeline.sobelRenderPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(pipeline.sobelRenderBindGroup)
        .draw(3)
    } catch (e) {
      const msg = `[camera] sobelRender failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  } else if (displayMode === 'nms') {
    try {
      pipeline.edgesPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(pipeline.edgesBindGroup)
        .draw(3)
    } catch (e) {
      const msg = `[camera] edgesPipeline (nms) failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  } else if (displayMode === 'labels' || displayMode === 'debug') {
    const labelVizBindGroup = root.createBindGroup(pipeline.labelVizLayout, {
      labelBuffer: pipeline.compactLabelBuffer,
    })
    try {
      pipeline.labelVizPipeline
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
      pipeline.grayRenderPipeline
        .with(enc)
        .withColorAttachment({ view: pipeline.context })
        .with(pipeline.grayRenderBindGroup)
        .draw(3)
    } catch (e) {
      const msg = `[camera] grayRender fallback failed: ${e}`
      console.error(msg)
      onError?.(msg)
    }
  }

  pipeline.histogramDisplayPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.histContext })
    .with(pipeline.histogramDisplayBindGroup)
    .draw(6, HISTOGRAM_BINS)
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
    const quad = filtered[i]
    const H = tryComputeHomography(quad.corners)
    const debug = quad.cornerDebug
    const tagId = quad.vizTagId !== undefined ? quad.vizTagId >>> 0 : DECODED_TAG_ID_UNKNOWN

    data.push({
      homography: H
        ? d.mat3x3f(
            // transpose the matrix
            H[0],
            H[3],
            H[6],
            H[1],
            H[4],
            H[7],
            H[2],
            H[5],
            1,
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

  pipeline.quadCornersBuffer.write(data)
}

/** Grid overlay: 0 = legacy fail colors, 1 = red highlights insufficient-edge failures, 2 = blue highlights line-fit failures. */
export function setGridVizFailInterrogate(pipeline: CameraPipeline, mode: GridVizFailInterrogateMode): void {
  pipeline.gridVizDebugModeBuffer.write(mode)
}

/**
 * Encode and submit a command buffer that renders the pinned gray snapshot
 * from `slot` plus the current grid viz overlay and histogram.
 * Call this right after `updateQuadCornersBuffer` so the write and draw are
 * in the same synchronous block.
 */
export function presentGridFrame(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  pipeline: CameraPipeline,
  slot: FrameSlot,
): void {
  const enc = root.device.createCommandEncoder({ label: 'grid frame present' })

  pipeline.grayRenderPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.context, loadOp: 'load', storeOp: 'store' })
    .with(slot.grayRenderBindGroup)
    .draw(3)

  try {
    pipeline.gridVizPipeline
      .with(enc)
      .withColorAttachment({ view: pipeline.context, loadOp: 'load', storeOp: 'store' })
      .with(pipeline.gridVizBindGroup)
      .draw(4, MAX_DETECTED_TAGS)
  } catch (e) {
    console.error('[presentGridFrame] gridViz failed:', e)
  }

  pipeline.histogramDisplayPipeline
    .with(enc)
    .withColorAttachment({ view: pipeline.histContext })
    .with(pipeline.histogramDisplayBindGroup)
    .draw(6, HISTOGRAM_BINS)

  root.device.queue.submit([enc.finish()])
}

