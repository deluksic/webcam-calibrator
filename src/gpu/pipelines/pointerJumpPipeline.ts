// Pointer doubling on edge graph: init L[i] → min linear index among 3×3 edge neighbors; then L'[i] = L[L[i]].
// After each doubling step, parent tightening: for edge i with p=L[i], atomicMin(L[p], min_neighbor L[j]) so shared parents update laterally.
//
// Gradient compatibility: neighbors with strongly opposing gradient directions (dot(g_i, g_j) < cosThreshold)
// are NOT connected — this prevents corners from spanning across edge discontinuities.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d, std } from 'typegpu'
import { atomicLoad, atomicMin, atomicStore, length } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/contour'
import type { EdgeFilterBindResources } from '@/gpu/pipelines/edgeFilterPipeline'

/**
 * Pointer-doubling passes: L'[i]=L[L[i]] after init (see shaders below).
 *
 * MUST be even. After each full iteration the ping-pong index flips once, so an even
 * count guarantees `pj === 0` on exit and compact-label (fixed to pointerJumpBuffer0)
 * reads the converged buffer.
 */
export const POINTER_JUMP_ITERATIONS = 10

/** Full-frame compute tile; match [16,16,1] kernels here and `computeDispatch2d` in cameraFrame. */
const FULL_FRAME_WG = 16

export function createPointerJumpLayouts() {
  const initLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    labelBuffer: { storage: d.arrayOf(d.u32), access: 'mutable' },
  })
  const stepLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    readBuffer: { storage: d.arrayOf(d.u32), access: 'readonly' },
    writeBuffer: { storage: d.arrayOf(d.u32), access: 'mutable' },
  })
  const labelsToAtomicLayout = tgpu.bindGroupLayout({
    source: { storage: d.arrayOf(d.u32), access: 'readonly' },
    atomicLabels: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
  })
  const parentTightenLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    labelRead: { storage: d.arrayOf(d.u32), access: 'readonly' },
    atomicLabels: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
  })
  const atomicToLabelsLayout = tgpu.bindGroupLayout({
    atomicLabels: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
    dest: { storage: d.arrayOf(d.u32), access: 'mutable' },
  })
  return {
    initLayout,
    stepLayout,
    labelsToAtomicLayout,
    parentTightenLayout,
    atomicToLabelsLayout,
  }
}

export function createPointerJumpInitPipeline(
  root: TgpuRoot,
  initLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [FULL_FRAME_WG, FULL_FRAME_WG, 1],
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
    const ei = initLayout.$.edgeBuffer[idx]
    const em = length(ei)
    // Only initialize labels for EDGE pixels (non-zero edges)
    if (em <= d.f32(0)) {
      initLayout.$.labelBuffer[idx] = d.u32(COMPONENT_LABEL_INVALID)
      return
    }
    let best = idx
    for (const dy of tgpu.unroll(std.range(-1, 2))) {
      for (const dx of tgpu.unroll(std.range(-1, 2))) {
        const nx = x + dx
        const ny = y + dy
        if (nx >= d.i32(0) && nx < w && ny >= d.i32(0) && ny < h) {
          const nIdx = d.u32(ny * w + nx)
          const ej = initLayout.$.edgeBuffer[nIdx]
          const ejm = length(ej)
          if (ejm > d.f32(0) && nIdx < best) {
            best = nIdx
          }
        }
      }
    }
    initLayout.$.labelBuffer[idx] = best
  })

  return root.createComputePipeline({ compute: kernel })
}

export function createPointerJumpStepPipeline(
  root: TgpuRoot,
  stepLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [FULL_FRAME_WG, FULL_FRAME_WG, 1],
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
    const wh = d.u32(w) * d.u32(h)
    const p = stepLayout.$.readBuffer[idx]
    if (p === d.u32(COMPONENT_LABEL_INVALID) || p >= wh) {
      stepLayout.$.writeBuffer[idx] = d.u32(COMPONENT_LABEL_INVALID)
      return
    }
    stepLayout.$.writeBuffer[idx] = stepLayout.$.readBuffer[p]
  })

  return root.createComputePipeline({ compute: kernel })
}

export function createPointerJumpLabelsToAtomicPipeline(
  root: TgpuRoot,
  labelsToAtomicLayout: ReturnType<typeof createPointerJumpLayouts>['labelsToAtomicLayout'],
  width: number,
  height: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [FULL_FRAME_WG, FULL_FRAME_WG, 1],
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
    const v = labelsToAtomicLayout.$.source[idx]!
    atomicStore(labelsToAtomicLayout.$.atomicLabels[idx]!, v)
  })

  return root.createComputePipeline({ compute: kernel })
}

export function createPointerJumpParentTightenPipeline(
  root: TgpuRoot,
  parentTightenLayout: ReturnType<typeof createPointerJumpLayouts>['parentTightenLayout'],
  width: number,
  height: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [FULL_FRAME_WG, FULL_FRAME_WG, 1],
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
    const ei = parentTightenLayout.$.edgeBuffer[idx]!
    if (length(ei) <= d.f32(0)) {
      return
    }

    const wh = d.u32(w) * d.u32(h)
    const p = parentTightenLayout.$.labelRead[idx]!
    if (p === d.u32(COMPONENT_LABEL_INVALID) || p >= wh) {
      return
    }

    let cand = d.u32(COMPONENT_LABEL_INVALID)
    for (const dy of tgpu.unroll(std.range(-1, 2))) {
      for (const dx of tgpu.unroll(std.range(-1, 2))) {
        const nx = x + dx
        const ny = y + dy
        if (nx >= d.i32(0) && nx < w && ny >= d.i32(0) && ny < h) {
          const nIdx = d.u32(ny * w + nx)
          const ej = parentTightenLayout.$.edgeBuffer[nIdx]!
          if (length(ej) > d.f32(0)) {
            const lj = parentTightenLayout.$.labelRead[nIdx]!
            if (lj !== d.u32(COMPONENT_LABEL_INVALID)) {
              cand = std.min(cand, lj)
            }
          }
        }
      }
    }

    if (cand !== d.u32(COMPONENT_LABEL_INVALID)) {
      atomicMin(parentTightenLayout.$.atomicLabels[p]!, cand)
    }
  })

  return root.createComputePipeline({ compute: kernel })
}

export function createPointerJumpAtomicToLabelsPipeline(
  root: TgpuRoot,
  atomicToLabelsLayout: ReturnType<typeof createPointerJumpLayouts>['atomicToLabelsLayout'],
  width: number,
  height: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [FULL_FRAME_WG, FULL_FRAME_WG, 1],
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
    const v = atomicLoad(atomicToLabelsLayout.$.atomicLabels[idx]!)
    atomicToLabelsLayout.$.dest[idx] = v
  })

  return root.createComputePipeline({ compute: kernel })
}

/** Allocates label + atomic ping-pong buffers; reads NMS `filteredBuffer` (upstream). */
export function createPointerJumpLabeling(
  root: TgpuRoot,
  width: number,
  height: number,
  filteredBuffer: EdgeFilterBindResources['filteredBuffer'],
) {
  const area = width * height
  const pointerJumpBuffer0 = root.createBuffer(d.arrayOf(d.u32, area)).$usage('storage')
  const pointerJumpBuffer1 = root.createBuffer(d.arrayOf(d.u32, area)).$usage('storage')
  const pointerJumpAtomicBuffer = root.createBuffer(d.arrayOf(d.atomic(d.u32), area)).$usage('storage')

  const {
    initLayout: pointerJumpInitLayout,
    stepLayout: pointerJumpStepLayout,
    labelsToAtomicLayout: pointerJumpLabelsToAtomicLayout,
    parentTightenLayout: pointerJumpParentTightenLayout,
    atomicToLabelsLayout: pointerJumpAtomicToLabelsLayout,
  } = createPointerJumpLayouts()
  const pointerJumpInitPipeline = createPointerJumpInitPipeline(root, pointerJumpInitLayout, width, height)
  const pointerJumpStepPipeline = createPointerJumpStepPipeline(root, pointerJumpStepLayout, width, height)
  const pointerJumpLabelsToAtomicPipeline = createPointerJumpLabelsToAtomicPipeline(
    root,
    pointerJumpLabelsToAtomicLayout,
    width,
    height,
  )
  const pointerJumpParentTightenPipeline = createPointerJumpParentTightenPipeline(
    root,
    pointerJumpParentTightenLayout,
    width,
    height,
  )
  const pointerJumpAtomicToLabelsPipeline = createPointerJumpAtomicToLabelsPipeline(
    root,
    pointerJumpAtomicToLabelsLayout,
    width,
    height,
  )

  const pointerJumpInitBindGroup = root.createBindGroup(pointerJumpInitLayout, {
    edgeBuffer: filteredBuffer,
    labelBuffer: pointerJumpBuffer0,
  })
  const pointerJumpPingPongBindGroups = [
    root.createBindGroup(pointerJumpStepLayout, {
      edgeBuffer: filteredBuffer,
      readBuffer: pointerJumpBuffer0,
      writeBuffer: pointerJumpBuffer1,
    }),
    root.createBindGroup(pointerJumpStepLayout, {
      edgeBuffer: filteredBuffer,
      readBuffer: pointerJumpBuffer1,
      writeBuffer: pointerJumpBuffer0,
    }),
  ]
  const pointerJumpLabelsToAtomicBindGroups = [
    root.createBindGroup(pointerJumpLabelsToAtomicLayout, {
      source: pointerJumpBuffer0,
      atomicLabels: pointerJumpAtomicBuffer,
    }),
    root.createBindGroup(pointerJumpLabelsToAtomicLayout, {
      source: pointerJumpBuffer1,
      atomicLabels: pointerJumpAtomicBuffer,
    }),
  ]
  const pointerJumpParentTightenBindGroups = [
    root.createBindGroup(pointerJumpParentTightenLayout, {
      edgeBuffer: filteredBuffer,
      labelRead: pointerJumpBuffer0,
      atomicLabels: pointerJumpAtomicBuffer,
    }),
    root.createBindGroup(pointerJumpParentTightenLayout, {
      edgeBuffer: filteredBuffer,
      labelRead: pointerJumpBuffer1,
      atomicLabels: pointerJumpAtomicBuffer,
    }),
  ]
  const pointerJumpAtomicToLabelsBindGroups = [
    root.createBindGroup(pointerJumpAtomicToLabelsLayout, {
      atomicLabels: pointerJumpAtomicBuffer,
      dest: pointerJumpBuffer0,
    }),
    root.createBindGroup(pointerJumpAtomicToLabelsLayout, {
      atomicLabels: pointerJumpAtomicBuffer,
      dest: pointerJumpBuffer1,
    }),
  ]

  const wgX = Math.ceil(width / FULL_FRAME_WG)
  const wgY = Math.ceil(height / FULL_FRAME_WG)
  const encodeCompute = (pass: GPUComputePassEncoder) => {
    pointerJumpInitPipeline.with(pass).with(pointerJumpInitBindGroup).dispatchWorkgroups(wgX, wgY)
    let pj = 0
    for (let s = 0; s < POINTER_JUMP_ITERATIONS; s++) {
      pointerJumpStepPipeline.with(pass).with(pointerJumpPingPongBindGroups[pj]!).dispatchWorkgroups(wgX, wgY)
      pj ^= 1
      pointerJumpLabelsToAtomicPipeline
        .with(pass)
        .with(pointerJumpLabelsToAtomicBindGroups[pj]!)
        .dispatchWorkgroups(wgX, wgY)
      pointerJumpParentTightenPipeline
        .with(pass)
        .with(pointerJumpParentTightenBindGroups[pj]!)
        .dispatchWorkgroups(wgX, wgY)
      pointerJumpAtomicToLabelsPipeline
        .with(pass)
        .with(pointerJumpAtomicToLabelsBindGroups[pj]!)
        .dispatchWorkgroups(wgX, wgY)
    }
  }

  return {
    pointerJumpBuffer0,
    pointerJumpBuffer1,
    pointerJumpAtomicBuffer,
    encodeCompute,
  }
}

export type PointerJumpConvergedLabels = ReturnType<typeof createPointerJumpLabeling>['pointerJumpBuffer0']
