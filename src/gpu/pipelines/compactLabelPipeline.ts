// Compact labeling: remap pointer-jump roots to compact IDs (0..N-1).
//
// Problem: pointer-jump labels are raw pixel indices (up to area-1). The extent
// buffer is sized for `MAX_EXTENT_COMPONENTS` (see extentTrackingPipeline). Roots with
// index >= that limit are discarded — acceptable for our use case.
//
// Pipeline (3 passes after pointer-jump):
//   1. Reset canonicalRoot to INVALID (needed because atomicMin is used)
//   2. Claim: each root pixel (label == pixel idx) stores its own index
//             at canonicalRoot[root_idx] as the compact ID.
//             Root pixels with index >= maxComponents skip claiming.
//   3. Remap: L[i] = canonicalRoot[label] (compact ID), or INVALID if > maxComponents
//
// Root index IS the compact ID — this is deterministic, no counter race.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { atomicLoad, atomicStore, atomicAdd } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/contour'
import type { PointerJumpConvergedLabels } from '@/gpu/pipelines/pointerJumpPipeline'

/**
 * Thread count per workgroup for compact passes (1D rows and 2D remap).
 * Match `FULL_FRAME_WG` / `computeDispatch2d` in cameraFrame and pointer-jump.
 */
const WORKGROUP_SIZE = 16

/** Allocates remap buffers; reads converged `pointerJumpBuffer0` (upstream). */
export function createCompactLabelStage(
  root: TgpuRoot,
  width: number,
  height: number,
  maxComponents: number,
  pointerJumpBuffer0: PointerJumpConvergedLabels,
) {
  const area = width * height
  const canonicalRootBuffer = root.createBuffer(d.arrayOf(d.atomic(d.u32), area)).$usage('storage')
  const compactLabelBuffer = root.createBuffer(d.arrayOf(d.u32, area)).$usage('storage')
  const compactCounterBuffer = root.createBuffer(d.atomic(d.u32)).$usage('storage')
  const { resetLayout, claimLayout, remapLayout } = createCompactLabelLayouts()
  const compactResetPipeline = createCanonicalResetPipeline(root, resetLayout, area)
  const compactClaimPipeline = createCanonicalClaimPipeline(root, claimLayout, width, height, maxComponents)
  const compactRemapPipeline = createRemapLabelPipeline(root, remapLayout, width, height)
  const compactResetBindGroup = root.createBindGroup(resetLayout, {
    compactCounter: compactCounterBuffer,
    canonicalRoot: canonicalRootBuffer,
  })
  const compactClaimBindGroup = root.createBindGroup(claimLayout, {
    labelBuffer: pointerJumpBuffer0,
    compactCounter: compactCounterBuffer,
    canonicalRoot: canonicalRootBuffer,
  })
  const compactRemapBindGroup = root.createBindGroup(remapLayout, {
    labelBuffer: pointerJumpBuffer0,
    compactLabelBuffer,
    canonicalRoot: canonicalRootBuffer,
  })
  const wgX = Math.ceil(width / WORKGROUP_SIZE)
  const wgY = Math.ceil(height / WORKGROUP_SIZE)
  const resetWg = Math.ceil(area / WORKGROUP_SIZE)
  const encodeCompute = (pass: GPUComputePassEncoder) => {
    compactResetPipeline.with(pass).with(compactResetBindGroup).dispatchWorkgroups(resetWg)
    compactClaimPipeline.with(pass).with(compactClaimBindGroup).dispatchWorkgroups(wgX, wgY)
    compactRemapPipeline.with(pass).with(compactRemapBindGroup).dispatchWorkgroups(wgX, wgY)
  }
  return {
    canonicalRootBuffer,
    compactLabelBuffer,
    compactCounterBuffer,
    encodeCompute,
  }
}

export function createCompactLabelLayouts() {
  const resetLayout = tgpu.bindGroupLayout({
    compactCounter: { storage: d.atomic(d.u32), access: 'mutable' },
    canonicalRoot: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
  })
  const claimLayout = tgpu.bindGroupLayout({
    labelBuffer: { storage: d.arrayOf(d.u32), access: 'readonly' },
    compactCounter: { storage: d.atomic(d.u32), access: 'mutable' },
    canonicalRoot: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
  })
  const remapLayout = tgpu.bindGroupLayout({
    labelBuffer: { storage: d.arrayOf(d.u32), access: 'readonly' },
    compactLabelBuffer: { storage: d.arrayOf(d.u32), access: 'mutable' },
    // Note: atomics in storage address space require 'mutable' (read_write) access.
    // The remap kernel uses atomicLoad which needs mutable, but this is safe since
    // we only read (atomicLoad doesn't modify the value, just reads it atomically).
    canonicalRoot: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
  })
  return { resetLayout, claimLayout, remapLayout }
}

export function createCanonicalResetPipeline(
  root: TgpuRoot,
  resetLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  area: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const slot = d.u32(input.gid.x)
    if (slot >= d.u32(area)) {
      return
    }
    atomicStore(resetLayout.$.canonicalRoot[slot], d.u32(COMPONENT_LABEL_INVALID))
    if (slot === d.u32(0)) {
      atomicStore(resetLayout.$.compactCounter, d.u32(0))
    }
  })
  return root.createComputePipeline({ compute: kernel })
}

export function createCanonicalClaimPipeline(
  root: TgpuRoot,
  claimLayout: ReturnType<typeof tgpu.bindGroupLayout>,
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
    const label = claimLayout.$.labelBuffer[idx]
    if (label === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    // Only roots (minimal pixel in component) claim a compact ID.
    // A pixel is a root iff label == idx (pointer-jump sets this).
    if (label !== idx) {
      return
    }
    // Only claim if root pixel index fits in canonicalRoot buffer.
    const area = d.u32(width * height)
    if (label >= area) {
      return
    }

    // Atomically claim next compact ID — sequential regardless of root pixel index.
    const compactId = atomicAdd(claimLayout.$.compactCounter, d.u32(1))
    if (compactId >= d.u32(maxComponents)) {
      return
    }
    atomicStore(claimLayout.$.canonicalRoot[label], compactId)
  })
  return root.createComputePipeline({ compute: kernel })
}

export function createRemapLabelPipeline(
  root: TgpuRoot,
  remapLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
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
    const label = remapLayout.$.labelBuffer[idx]
    if (label === d.u32(COMPONENT_LABEL_INVALID)) {
      remapLayout.$.compactLabelBuffer[idx] = d.u32(COMPONENT_LABEL_INVALID)
      return
    }

    const compactId = atomicLoad(remapLayout.$.canonicalRoot[label])
    remapLayout.$.compactLabelBuffer[idx] = compactId
  })
  return root.createComputePipeline({ compute: kernel })
}

export type CompactLabelMapBuffer = ReturnType<typeof createCompactLabelStage>['compactLabelBuffer']
