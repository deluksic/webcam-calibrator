// Compact labeling: remap pointer-jump roots to compact IDs (0..N-1).
//
// Problem: pointer-jump labels are raw pixel indices (up to area-1). The extent
// buffer is sized for MAX_EXTENT_COMPONENTS slots. Components with root pixel
// index >= MAX_EXTENT_COMPONENTS are discarded — acceptable for our use case.
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
import { COMPUTE_WORKGROUP_SIZE } from '@/gpu/pipelines/constants'

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
    workgroupSize: [COMPUTE_WORKGROUP_SIZE, 1, 1],
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
    workgroupSize: [COMPUTE_WORKGROUP_SIZE, COMPUTE_WORKGROUP_SIZE, 1],
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
    workgroupSize: [COMPUTE_WORKGROUP_SIZE, COMPUTE_WORKGROUP_SIZE, 1],
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
