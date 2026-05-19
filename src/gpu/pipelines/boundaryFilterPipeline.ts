// Drop interior pixels from connected components — keep only boundary outline.
//
// Two per-label buffers track row and column extents independently:
//   rowBounds[label * height + row] → { minCol, maxCol }  (col extent on that row)
//   colBounds[label * width  + col] → { minRow, maxRow }  (row extent on that column)
//
// A pixel is interior (dropped) iff it is further than BOUNDARY_MARGIN_PX from
// the nearest extent in BOTH its row dimension AND its column dimension.
//
// Modifies compactLabelBuffer in-place so downstream stages see only boundary pixels.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d, std } from 'typegpu'
import { atomicStore, atomicMin, atomicMax } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/detectedQuad'
import type { CompactLabelMapBuffer } from '@/gpu/pipelines/compactLabelPipeline'

export const BOUNDARY_MARGIN_PX = 1
const WORKGROUP_SIZE = 16
const WORKGROUP_SIZE_RESET = 256

// ── Per (label, row): column extent on this specific row ──
const LabelRowBoundsAtomic = d.struct({
  minCol: d.atomic(d.u32),
  maxCol: d.atomic(d.u32),
})
const LabelRowBoundsReadonly = d.struct({
  minCol: d.u32,
  maxCol: d.u32,
})

// ── Per (label, col): row extent on this specific column ──
const LabelColBoundsAtomic = d.struct({
  minRow: d.atomic(d.u32),
  maxRow: d.atomic(d.u32),
})
const LabelColBoundsReadonly = d.struct({
  minRow: d.u32,
  maxRow: d.u32,
})

function createBoundaryFilterLayouts() {
  const resetRowLayout = tgpu.bindGroupLayout({
    rowBounds: { storage: d.arrayOf(LabelRowBoundsAtomic), access: 'mutable' },
  })
  const resetColLayout = tgpu.bindGroupLayout({
    colBounds: { storage: d.arrayOf(LabelColBoundsAtomic), access: 'mutable' },
  })
  const reduceLayout = tgpu.bindGroupLayout({
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    rowBounds: { storage: d.arrayOf(LabelRowBoundsAtomic), access: 'mutable' },
    colBounds: { storage: d.arrayOf(LabelColBoundsAtomic), access: 'mutable' },
  })
  const filterLayout = tgpu.bindGroupLayout({
    compactLabels: { storage: d.arrayOf(d.u32), access: 'mutable' },
    rowBounds: { storage: d.arrayOf(LabelRowBoundsReadonly), access: 'readonly' },
    colBounds: { storage: d.arrayOf(LabelColBoundsReadonly), access: 'readonly' },
  })
  return { resetRowLayout, resetColLayout, reduceLayout, filterLayout }
}

// ── Reset: per (label, row) slot ──
function createRowBoundsResetPipeline(
  root: TgpuRoot,
  resetRowLayout: ReturnType<typeof createBoundaryFilterLayouts>['resetRowLayout'],
  rowSlots: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE_RESET, 1, 1],
  })((input) => {
    'use gpu'
    const slot = d.u32(input.gid.x)
    if (slot >= d.u32(rowSlots)) {
      return
    }
    const b = resetRowLayout.$.rowBounds[slot]!
    atomicStore(b.minCol, d.u32(0xffff_ffff))
    atomicStore(b.maxCol, d.u32(0))
  })
  return root.createComputePipeline({ compute: kernel })
}

// ── Reset: per (label, col) slot ──
function createColBoundsResetPipeline(
  root: TgpuRoot,
  resetColLayout: ReturnType<typeof createBoundaryFilterLayouts>['resetColLayout'],
  colSlots: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE_RESET, 1, 1],
  })((input) => {
    'use gpu'
    const slot = d.u32(input.gid.x)
    if (slot >= d.u32(colSlots)) {
      return
    }
    const b = resetColLayout.$.colBounds[slot]!
    atomicStore(b.minRow, d.u32(0xffff_ffff))
    atomicStore(b.maxRow, d.u32(0))
  })
  return root.createComputePipeline({ compute: kernel })
}

// ── Reduce: full-frame → atomicMin/atomicMax into per-(label,row) and per-(label,col) ──
function createBoundsReducePipeline(
  root: TgpuRoot,
  reduceLayout: ReturnType<typeof createBoundaryFilterLayouts>['reduceLayout'],
  width: number,
  height: number,
) {
  const h = d.u32(height)
  const w = d.u32(width)

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, WORKGROUP_SIZE, 1],
  })((input) => {
    'use gpu'
    const x = d.i32(input.gid.x)
    const y = d.i32(input.gid.y)
    if (x >= d.i32(w) || y >= d.i32(h)) {
      return
    }

    const idx = d.u32(y * d.i32(w) + x)
    const label = reduceLayout.$.compactLabels[idx]!
    if (label === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const row = d.u32(y)
    const col = d.u32(x)

    const rowSlot = label * h + row
    const rb = reduceLayout.$.rowBounds[rowSlot]!
    atomicMin(rb.minCol, col)
    atomicMax(rb.maxCol, col)

    const colSlot = label * w + col
    const cb = reduceLayout.$.colBounds[colSlot]!
    atomicMin(cb.minRow, row)
    atomicMax(cb.maxRow, row)
  })
  return root.createComputePipeline({ compute: kernel })
}

// ── Filter: drop pixels too far inside in both dimensions ──
function createBoundaryFilterPipeline(
  root: TgpuRoot,
  filterLayout: ReturnType<typeof createBoundaryFilterLayouts>['filterLayout'],
  width: number,
  height: number,
) {
  const margin = d.u32(BOUNDARY_MARGIN_PX)
  const h = d.u32(height)
  const w = d.u32(width)

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, WORKGROUP_SIZE, 1],
  })((input) => {
    'use gpu'
    const x = d.i32(input.gid.x)
    const y = d.i32(input.gid.y)
    if (x >= d.i32(w) || y >= d.i32(h)) {
      return
    }

    const idx = d.u32(y * d.i32(w) + x)
    const label = filterLayout.$.compactLabels[idx]!
    if (label === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const row = d.u32(y)
    const col = d.u32(x)

    // Distance to column extent on this row.
    const rb = filterLayout.$.rowBounds[label * h + row]!
    const distC = d.u32(std.min(col - rb.minCol, rb.maxCol - col))

    // Distance to row extent on this column.
    const cb = filterLayout.$.colBounds[label * w + col]!
    const distR = d.u32(std.min(row - cb.minRow, cb.maxRow - row))

    if (distR > margin && distC > margin) {
      filterLayout.$.compactLabels[idx] = d.u32(COMPONENT_LABEL_INVALID)
    }
  })
  return root.createComputePipeline({ compute: kernel })
}

// ── Stage factory ──
export function createBoundaryFilterStage(
  root: TgpuRoot,
  width: number,
  height: number,
  maxComponents: number,
  compactLabelBuffer: CompactLabelMapBuffer,
) {
  const rowSlots = maxComponents * height
  const colSlots = maxComponents * width
  const rowBounds = root.createBuffer(d.arrayOf(LabelRowBoundsAtomic, rowSlots)).$usage('storage')
  const colBounds = root.createBuffer(d.arrayOf(LabelColBoundsAtomic, colSlots)).$usage('storage')

  const layouts = createBoundaryFilterLayouts()
  const resetRowPipeline = createRowBoundsResetPipeline(root, layouts.resetRowLayout, rowSlots)
  const resetColPipeline = createColBoundsResetPipeline(root, layouts.resetColLayout, colSlots)
  const reducePipeline = createBoundsReducePipeline(root, layouts.reduceLayout, width, height)
  const filterPipeline = createBoundaryFilterPipeline(root, layouts.filterLayout, width, height)

  const resetRowBindGroup = root.createBindGroup(layouts.resetRowLayout, { rowBounds })
  const resetColBindGroup = root.createBindGroup(layouts.resetColLayout, { colBounds })
  const reduceBindGroup = root.createBindGroup(layouts.reduceLayout, {
    compactLabels: compactLabelBuffer,
    rowBounds,
    colBounds,
  })
  const filterBindGroup = root.createBindGroup(layouts.filterLayout, {
    compactLabels: compactLabelBuffer,
    rowBounds,
    colBounds,
  })

  const wgX = Math.ceil(width / WORKGROUP_SIZE)
  const wgY = Math.ceil(height / WORKGROUP_SIZE)
  const resetRowWg = Math.ceil(rowSlots / WORKGROUP_SIZE_RESET)
  const resetColWg = Math.ceil(colSlots / WORKGROUP_SIZE_RESET)

  const encodeCompute = (pass: GPUComputePassEncoder) => {
    resetRowPipeline.with(pass).with(resetRowBindGroup).dispatchWorkgroups(resetRowWg)
    resetColPipeline.with(pass).with(resetColBindGroup).dispatchWorkgroups(resetColWg)
    reducePipeline.with(pass).with(reduceBindGroup).dispatchWorkgroups(wgX, wgY)
    filterPipeline.with(pass).with(filterBindGroup).dispatchWorkgroups(wgX, wgY)
  }

  return { rowBounds, colBounds, encodeCompute }
}

export type BoundaryFilterStage = ReturnType<typeof createBoundaryFilterStage>
