// Extent tracking: atomic min/max per component across a single pass over the label buffer.
// GPU tracks (minX, minY, maxX, maxY) for each component ID.
// CPU reads the tiny extent array, sorts, and picks the top N candidates.
import { tgpu, d } from 'typegpu';
import { atomicMin, atomicMax, atomicStore } from 'typegpu/std';
import { COMPUTE_WORKGROUP_SIZE } from './constants';
import { COMPONENT_LABEL_INVALID } from '../contour';

/**
 * Extent entry: 5 × u32 packed as [minX, minY, maxX, maxY, originalLabel].
 * Slot layout:
 *   - 5*i+0: minX         (or MAX_U32 if uninitialized)
 *   - 5*i+1: minY         (or MAX_U32 if uninitialized)
 *   - 5*i+2: maxX         (or 0 if uninitialized)
 *   - 5*i+3: maxY         (or 0 if uninitialized)
 *   - 5*i+4: originalLabel (the root pixel index)
 */
export const EXTENT_FIELDS = 5 as const;
export const MAX_U32 = 0xFFFFFFFF;

// ════════════════════════════════════════════════════════════════════════════
// Layouts
// ════════════════════════════════════════════════════════════════════════════

export function createExtentTrackingLayouts(
  root: Awaited<ReturnType<typeof tgpu.init>>,
) {
  // One pass: read labels, write extents
  const trackLayout = tgpu.bindGroupLayout({
    labelBuffer: { storage: d.arrayOf(d.u32), access: 'readonly' },
    // extentBuffer[i*4..i*4+3] = [minX, minY, maxX, maxY] for component i
    // All fields are atomic so atomicMin/atomicMax work
    extentBuffer: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
  });
  return { trackLayout };
}

// ════════════════════════════════════════════════════════════════════════════
// Reset: fill extentBuffer with sentinel values so atomics work correctly
// ════════════════════════════════════════════════════════════════════════════

export function createExtentResetPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  resetLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  numComponents: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [COMPUTE_WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu';
    const slot = d.u32(input.gid.x);
    const numSlots = d.u32(numComponents);
    if (slot >= numSlots) { return; }
    const base = slot * d.u32(EXTENT_FIELDS);
    // minX = MAX, minY = MAX, maxX = 0, maxY = 0, originalLabel = INVALID
    atomicStore(resetLayout.$.extentBuffer[base + d.u32(0)], d.u32(MAX_U32));
    atomicStore(resetLayout.$.extentBuffer[base + d.u32(1)], d.u32(MAX_U32));
    atomicStore(resetLayout.$.extentBuffer[base + d.u32(2)], d.u32(0));
    atomicStore(resetLayout.$.extentBuffer[base + d.u32(3)], d.u32(0));
    atomicStore(resetLayout.$.extentBuffer[base + d.u32(4)], d.u32(COMPONENT_LABEL_INVALID));
  });
  return root.createComputePipeline({ compute: kernel });
}

// ════════════════════════════════════════════════════════════════════════════
// Track: one pass — every pixel with a valid label updates its extent atomically
// ════════════════════════════════════════════════════════════════════════════

export function createExtentTrackPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  trackLayout: ReturnType<typeof createExtentTrackingLayouts>['trackLayout'],
  width: number,
  height: number,
  maxComponents: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [COMPUTE_WORKGROUP_SIZE, COMPUTE_WORKGROUP_SIZE, 1],
  })((input) => {
    'use gpu';
    const x = d.i32(input.gid.x);
    const y = d.i32(input.gid.y);
    const w = d.i32(width);
    const h = d.i32(height);
    if (x >= w || y >= h) { return; }

    const idx = d.u32(y * w + x);
    const label = trackLayout.$.labelBuffer[idx];
    if (label === d.u32(COMPONENT_LABEL_INVALID)) { return; }

    // Key by originalLabel (root pixel index), skip if out of bounds
    if (label >= d.u32(maxComponents)) { return; }

    const base = label * d.u32(EXTENT_FIELDS);
    atomicMin(trackLayout.$.extentBuffer[base + d.u32(0)], d.u32(x));
    atomicMin(trackLayout.$.extentBuffer[base + d.u32(1)], d.u32(y));
    atomicMax(trackLayout.$.extentBuffer[base + d.u32(2)], d.u32(x));
    atomicMax(trackLayout.$.extentBuffer[base + d.u32(3)], d.u32(y));
    // originalLabel as last field (redundant but useful for validation)
    atomicStore(trackLayout.$.extentBuffer[base + d.u32(4)], label);
  });
  return root.createComputePipeline({ compute: kernel });
}
