import type { StorageFlag, TgpuBuffer, TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import { atomicLoad, atomicMin, atomicStore, length, max, min, select, workgroupBarrier } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/pipelines/pointerJumpPipeline'

/**
 * Tile-based connected-component labeling in three dispatches.
 *
 * 1. **tile roots** — one workgroup per 16×16 tile. Each tile labels itself in
 *    workgroup memory: every foreground pixel starts at the lowest index among
 *    itself and its in-tile 4-neighbours, then the local forest is pointer-
 *    jumped until it collapses. Because the array is `var<workgroup>`, those
 *    iterations are barrier syncs rather than dispatches.
 * 2. **border merge** — only a tile's right column and bottom row can have a
 *    neighbour in another tile, so this dispatches `tiles × 32` threads instead
 *    of the full frame (12.5% of it) and unions the two tile roots.
 * 3. **resolve** — every pixel walks its own chain to the root. Chains are
 *    short because pass 1 already collapsed each tile and the merge attaches
 *    roots directly to the lowest root it has seen, so a bounded walk inside a
 *    single dispatch replaces a series of pointer-jump dispatches.
 *
 * Init and union only ever *lower* a parent, so `parent[x] <= x` holds
 * throughout, every walk strictly descends, and the surviving root is the
 * lowest pixel index in the component — the same "lowest index is the label"
 * convention the pointer-jump path uses, so downstream stages are unchanged.
 */

const TILE = 16
const TILE_AREA = TILE * TILE
/**
 * Relaxation rounds inside a tile: a lateral hook plus one pointer jump each.
 * The hook is what merges sibling branches — a min-neighbour init can leave two
 * roots inside one component whenever the path doubles back in index order
 * (e.g. right then up), and doubling alone can never join them. Sixteen rounds
 * propagate a hook across the full tile width.
 */
const TILE_ITERATIONS = 16
/**
 * Bound on a root walk, not a limit on the component's size. A union has to be
 * applied to the *current* root, so both sides of a border are re-walked before
 * every attempt; unions only lower a parent, so the tree flattens as it grows
 * and real chains measured 1–2 hops even for a thin shape spanning the frame.
 */
const MAX_ROOT_HOPS = 64
/**
 * A union is dropped when `atomicMin` targets a slot that another border has
 * already lowered below the value being written. Re-walking and retrying turns
 * that into a repair instead of a split; a second merge *dispatch* would not,
 * because it re-reads the same tile-root slots and repeats the same no-op.
 */
const UNION_ATTEMPTS = 4
const MERGE_WORKGROUP = 64
const RESOLVE_WORKGROUP = 16

const tileLayout = tgpu.bindGroupLayout({
  edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
  parent: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
}).$name('tile-label-bgl')

const mergeLayout = tgpu.bindGroupLayout({
  edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
  parent: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
}).$name('tile-merge-bgl')

const resolveLayout = tgpu.bindGroupLayout({
  parent: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
  labels: { storage: d.arrayOf(d.u32), access: 'mutable' },
}).$name('tile-resolve-bgl')

const tileLocal = tgpu.workgroupVar(d.arrayOf(d.u32, TILE_AREA))

export function createTileLabeling(
  root: TgpuRoot,
  width: number,
  height: number,
  edgeBuffer: TgpuBuffer<d.WgslArray<d.Vec2f>> & StorageFlag,
) {
  const area = width * height
  const tilesX = Math.ceil(width / TILE)
  const tilesY = Math.ceil(height / TILE)
  const mergeThreads = tilesX * tilesY * TILE * 2

  const labels = root.createBuffer(d.arrayOf(d.u32, area)).$usage('storage')
  const parent = root.createBuffer(d.arrayOf(d.atomic(d.u32), area)).$usage('storage')

  const tileKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [TILE, TILE, 1],
  })((input) => {
    'use gpu'
    const w = d.i32(width)
    const h = d.i32(height)
    const tile = d.i32(TILE)
    const lastInTile = d.i32(TILE - 1)
    const gx = d.i32(input.gid.x)
    const gy = d.i32(input.gid.y)
    const lx = gx % tile
    const ly = gy % tile
    const local = d.u32(ly * tile + lx)
    const shared = tileLocal.$

    // Every thread of the workgroup must reach the barriers below, including
    // the ones whose pixel lies outside the frame on trailing tiles.
    shared[local] = d.u32(COMPONENT_LABEL_INVALID)
    workgroupBarrier()

    const inFrame = gx < w && gy < h
    let foreground = false
    if (inFrame) {
      const idx = d.u32(gy * w + gx)
      foreground = length(tileLayout.$.edgeBuffer[idx]!) > d.f32(0)
      if (foreground) {
        let best = local
        // In-tile 4-neighbourhood only; anything across a tile border is
        // joined by the merge pass.
        if (lx > d.i32(0)) {
          const nIdx = d.u32(gy * w + (gx - d.i32(1)))
          if (length(tileLayout.$.edgeBuffer[nIdx]!) > d.f32(0)) {
            best = min(best, local - d.u32(1))
          }
        }
        if (lx < lastInTile && gx + d.i32(1) < w) {
          const nIdx = d.u32(gy * w + (gx + d.i32(1)))
          if (length(tileLayout.$.edgeBuffer[nIdx]!) > d.f32(0)) {
            best = min(best, local + d.u32(1))
          }
        }
        if (ly > d.i32(0)) {
          const nIdx = d.u32((gy - d.i32(1)) * w + gx)
          if (length(tileLayout.$.edgeBuffer[nIdx]!) > d.f32(0)) {
            best = min(best, local - d.u32(TILE))
          }
        }
        if (ly < lastInTile && gy + d.i32(1) < h) {
          const nIdx = d.u32((gy + d.i32(1)) * w + gx)
          if (length(tileLayout.$.edgeBuffer[nIdx]!) > d.f32(0)) {
            best = min(best, local + d.u32(TILE))
          }
        }
        shared[local] = best
      }
    }
    workgroupBarrier()

    for (let step = 0; step < TILE_ITERATIONS; step += 1) {
      // Lateral hook: pull the smallest label from any in-tile neighbour.
      const current: number = shared[local] as number
      if (current !== d.u32(COMPONENT_LABEL_INVALID)) {
        let best = current
        if (lx > d.i32(0)) {
          const candidate: number = shared[local - d.u32(1)] as number
          if (candidate !== d.u32(COMPONENT_LABEL_INVALID)) {
            best = min(best, candidate)
          }
        }
        if (lx < lastInTile) {
          const candidate: number = shared[local + d.u32(1)] as number
          if (candidate !== d.u32(COMPONENT_LABEL_INVALID)) {
            best = min(best, candidate)
          }
        }
        if (ly > d.i32(0)) {
          const candidate: number = shared[local - d.u32(TILE)] as number
          if (candidate !== d.u32(COMPONENT_LABEL_INVALID)) {
            best = min(best, candidate)
          }
        }
        if (ly < lastInTile) {
          const candidate: number = shared[local + d.u32(TILE)] as number
          if (candidate !== d.u32(COMPONENT_LABEL_INVALID)) {
            best = min(best, candidate)
          }
        }
        shared[local] = best
      }
      workgroupBarrier()

      // Pointer jump: follow the chain, so long shapes converge quickly.
      const value: number = shared[local] as number
      if (value !== d.u32(COMPONENT_LABEL_INVALID)) {
        const climbed: number = shared[value] as number
        if (climbed !== d.u32(COMPONENT_LABEL_INVALID) && climbed < value) {
          shared[local] = climbed
        }
      }
      workgroupBarrier()
    }

    if (inFrame) {
      const idx = d.u32(gy * w + gx)
      if (foreground) {
        // Local index -> global pixel index for the tile's root.
        const rootLocal = d.i32(shared[local])
        const rootX = gx - lx + rootLocal % tile
        const rootY = gy - ly + d.i32(rootLocal / tile)
        atomicStore(tileLayout.$.parent[idx]!, d.u32(rootY * w + rootX))
      } else {
        atomicStore(tileLayout.$.parent[idx]!, d.u32(COMPONENT_LABEL_INVALID))
      }
    }
  })

  const mergeKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [MERGE_WORKGROUP, 1, 1],
  })((input) => {
    'use gpu'
    const w = d.i32(width)
    const h = d.i32(height)
    const tile = d.i32(TILE)
    const perTile = d.i32(TILE * 2)
    const lastInTile = d.i32(TILE - 1)
    const columns = d.i32(tilesX)
    const slot = d.i32(input.gid.x)
    if (slot >= d.i32(mergeThreads)) {
      return
    }
    const tileIndex = d.i32(slot / perTile)
    const slotInTile = slot % perTile
    const tileX = tileIndex % columns
    const tileY = d.i32(tileIndex / columns)

    const originX = tileX * tile
    const originY = tileY * tile
    // Slots 0..15 walk the right column; 16..31 walk the bottom row, where the
    // slot index has to be rebased onto the tile.
    const rightColumn = slotInTile < tile
    const column = select(slotInTile - tile, slotInTile, rightColumn)
    const x = select(originX + column, originX + lastInTile, rightColumn)
    const y = select(originY + lastInTile, originY + column, rightColumn)
    const nx = select(x, x + d.i32(1), rightColumn)
    const ny = select(y + d.i32(1), y, rightColumn)
    if (x >= w || y >= h || nx >= w || ny >= h) {
      return
    }
    const a = d.u32(y * w + x)
    const b = d.u32(ny * w + nx)
    if (length(mergeLayout.$.edgeBuffer[a]!) <= d.f32(0)) {
      return
    }
    if (length(mergeLayout.$.edgeBuffer[b]!) <= d.f32(0)) {
      return
    }
    if (
      atomicLoad(mergeLayout.$.parent[a]!) === d.u32(COMPONENT_LABEL_INVALID) ||
      atomicLoad(mergeLayout.$.parent[b]!) === d.u32(COMPONENT_LABEL_INVALID)
    ) {
      return
    }
    for (let attempt = 0; attempt < UNION_ATTEMPTS; attempt += 1) {
      // Both sides are walked to their *current* root first. `atomicMin` is a
      // no-op when the target already holds something smaller, so a union
      // applied to a root captured earlier is silently dropped — and a dropped
      // union is exactly what splits one blob into pieces at a tile seam.
      let walkA = atomicLoad(mergeLayout.$.parent[a]!)
      let walkB = atomicLoad(mergeLayout.$.parent[b]!)
      for (let hop = 0; hop < MAX_ROOT_HOPS; hop += 1) {
        const next = atomicLoad(mergeLayout.$.parent[walkA]!)
        if (next >= walkA) {
          break
        }
        walkA = next
      }
      for (let hop = 0; hop < MAX_ROOT_HOPS; hop += 1) {
        const next = atomicLoad(mergeLayout.$.parent[walkB]!)
        if (next >= walkB) {
          break
        }
        walkB = next
      }
      if (walkA === walkB) {
        return
      }
      const low = min(walkA, walkB)
      const high = max(walkA, walkB)
      atomicMin(mergeLayout.$.parent[high]!, low)
      // Only a concurrent lowering between the walk and the write can lose the
      // union; re-check and retry rather than leaving the components apart.
      if (atomicLoad(mergeLayout.$.parent[high]!) === low) {
        return
      }
    }
  })

  const resolveKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [RESOLVE_WORKGROUP, RESOLVE_WORKGROUP, 1],
  })((input) => {
    'use gpu'
    const w = d.i32(width)
    const h = d.i32(height)
    const gx = d.i32(input.gid.x)
    const gy = d.i32(input.gid.y)
    if (gx >= w || gy >= h) {
      return
    }
    const idx = d.u32(gy * w + gx)
    const start = atomicLoad(resolveLayout.$.parent[idx]!)
    if (start === d.u32(COMPONENT_LABEL_INVALID)) {
      resolveLayout.$.labels[idx] = d.u32(COMPONENT_LABEL_INVALID)
      return
    }
    let root = start
    for (let hop = 0; hop < MAX_ROOT_HOPS; hop += 1) {
      const next = atomicLoad(resolveLayout.$.parent[root]!)
      if (next === d.u32(COMPONENT_LABEL_INVALID) || next >= root) {
        break
      }
      root = next
    }
    resolveLayout.$.labels[idx] = root
  })

  const tilePipeline = root.createComputePipeline({ compute: tileKernel })
  const mergePipeline = root.createComputePipeline({ compute: mergeKernel })
  const resolvePipeline = root.createComputePipeline({ compute: resolveKernel })

  const tileBindGroup = root.createBindGroup(tileLayout, { edgeBuffer, parent })
  const mergeBindGroup = root.createBindGroup(mergeLayout, { edgeBuffer, parent })
  const resolveBindGroup = root.createBindGroup(resolveLayout, { parent, labels })

  return {
    labels,
    parent,
    encodeCompute(pass: GPUComputePassEncoder) {
      tilePipeline.with(pass).with(tileBindGroup).dispatchWorkgroups(tilesX, tilesY)
      mergePipeline
        .with(pass)
        .with(mergeBindGroup)
        .dispatchWorkgroups(Math.ceil(mergeThreads / MERGE_WORKGROUP))
      resolvePipeline
        .with(pass)
        .with(resolveBindGroup)
        .dispatchWorkgroups(
          Math.ceil(width / RESOLVE_WORKGROUP),
          Math.ceil(height / RESOLVE_WORKGROUP),
        )
    },
    destroy() {
      labels.destroy()
      parent.destroy()
    },
  }
}

export type TileLabeling = ReturnType<typeof createTileLabeling>
