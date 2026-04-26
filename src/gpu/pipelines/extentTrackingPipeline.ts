// Extent tracking: atomic min/max per component across a single pass over the label buffer.
// GPU tracks (minX, minY, maxX, maxY) for each component ID.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { atomicMin, atomicMax, atomicStore } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/contour'
import type { CompactLabelMapBuffer } from '@/gpu/pipelines/compactLabelPipeline'

/** Match compact-label and cameraFrame extent dispatches / full-frame [16,16] grid. */
const WORKGROUP_SIZE = 16

export const MAX_U32 = 0xffffffff
export const EXTENT_FIELDS = 4 as const

/**
 * Max labeled components with extent slots on GPU. Compact-label discards roots with
 * index ≥ this; extent buffer has this many entries. Keep in sync with compact pass.
 */
export const MAX_EXTENT_COMPONENTS = 16384

/** Extent entry stored in the extent buffer. */
export const ExtentEntry = d.struct({
  minX: d.atomic(d.u32), // or MAX_U32 if uninitialized
  minY: d.atomic(d.u32), // or MAX_U32 if uninitialized
  maxX: d.atomic(d.u32), // or 0 if uninitialized
  maxY: d.atomic(d.u32), // or 0 if uninitialized
})

// ════════════════════════════════════════════════════════════════════════════
// Layouts
// ════════════════════════════════════════════════════════════════════════════

export const extentResetLayout = tgpu.bindGroupLayout({
  extentBuffer: { storage: d.arrayOf(ExtentEntry), access: 'mutable' },
})

export function createExtentTrackingLayouts() {
  const trackLayout = tgpu.bindGroupLayout({
    labelBuffer: { storage: d.arrayOf(d.u32), access: 'readonly' },
    extentBuffer: { storage: d.arrayOf(ExtentEntry), access: 'mutable' },
  })
  return { trackLayout }
}

/** Allocates `extentBuffer`; reads compact `labelBuffer` (upstream). */
export function createExtentTrackingStage(
  root: TgpuRoot,
  width: number,
  height: number,
  maxComponents: number,
  compactLabelBuffer: CompactLabelMapBuffer,
) {
  const { trackLayout: extentTrackLayout } = createExtentTrackingLayouts()
  const extentBuffer = root.createBuffer(d.arrayOf(ExtentEntry, maxComponents)).$usage('storage')
  const extentResetPipeline = createExtentResetPipeline(root, extentResetLayout, maxComponents)
  const extentTrackPipeline = createExtentTrackPipeline(root, extentTrackLayout, width, height, maxComponents)
  const extentResetBindGroup = root.createBindGroup(extentResetLayout, { extentBuffer })
  const extentTrackBindGroup = root.createBindGroup(extentTrackLayout, {
    labelBuffer: compactLabelBuffer,
    extentBuffer,
  })
  return {
    extentBuffer,
    extentTrackLayout,
    extentResetPipeline,
    extentResetBindGroup,
    extentTrackPipeline,
    extentTrackBindGroup,
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Reset: fill extentBuffer with sentinel values so atomics work correctly
// ════════════════════════════════════════════════════════════════════════════

export function createExtentResetPipeline(
  root: TgpuRoot,
  resetLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  numComponents: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const slot = d.u32(input.gid.x)
    const numSlots = d.u32(numComponents)
    if (slot >= numSlots) {
      return
    }
    const entry = resetLayout.$.extentBuffer[slot]
    atomicStore(entry.minX, d.u32(MAX_U32))
    atomicStore(entry.minY, d.u32(MAX_U32))
    atomicStore(entry.maxX, d.u32(0))
    atomicStore(entry.maxY, d.u32(0))
  })
  return root.createComputePipeline({ compute: kernel })
}

// ════════════════════════════════════════════════════════════════════════════
// Track: one pass — every pixel with a valid label updates its extent atomically
// ════════════════════════════════════════════════════════════════════════════

export function createExtentTrackPipeline(
  root: TgpuRoot,
  trackLayout: ReturnType<typeof createExtentTrackingLayouts>['trackLayout'],
  width: number,
  height: number,
  maxComponents: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, WORKGROUP_SIZE, 1],
  })((input) => {
    'use gpu'
    const x = d.i32(input.gid.x)
    const y = d.i32(input.gid.y)
    const w = d.i32(width)
    const h = d.i32(height)
    if (x >= w || y >= h) {
      return
    }

    const idx = d.u32(y * w + x)
    const label = trackLayout.$.labelBuffer[idx]!
    if (label === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    // Key by originalLabel (root pixel index), skip if out of bounds
    if (label >= d.u32(maxComponents)) {
      return
    }

    const entry = trackLayout.$.extentBuffer[label]!
    atomicMin(entry.minX, d.u32(x))
    atomicMin(entry.minY, d.u32(y))
    atomicMax(entry.maxX, d.u32(x))
    atomicMax(entry.maxY, d.u32(y))
  })
  return root.createComputePipeline({ compute: kernel })
}
