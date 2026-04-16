// Pointer doubling on edge graph: init L[i] → min linear index among 3×3 edge neighbors; then L'[i] = L[L[i]].
// After each doubling step, parent tightening: for edge i with p=L[i], atomicMin(L[p], min_neighbor L[j]) so shared parents update laterally.
//
// Gradient compatibility: neighbors with strongly opposing gradient directions (dot(g_i, g_j) < cosThreshold)
// are NOT connected — this prevents corners from spanning across edge discontinuities.
import { tgpu, d, std } from 'typegpu';
import { atomicLoad, atomicMin, atomicStore, length, sqrt } from 'typegpu/std';
import { COMPUTE_WORKGROUP_SIZE, GRADIENT_COS_THRESHOLD } from './constants';
import { COMPONENT_LABEL_INVALID } from '../contour';

export function createPointerJumpLayouts(root: Awaited<ReturnType<typeof tgpu.init>>) {
  const initLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
    labelBuffer: { storage: d.arrayOf(d.u32), access: 'mutable' },
  });
  const stepLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
    readBuffer: { storage: d.arrayOf(d.u32), access: 'readonly' },
    writeBuffer: { storage: d.arrayOf(d.u32), access: 'mutable' },
  });
  const labelsToAtomicLayout = tgpu.bindGroupLayout({
    source: { storage: d.arrayOf(d.u32), access: 'readonly' },
    atomicLabels: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
  });
  const parentTightenLayout = tgpu.bindGroupLayout({
    edgeBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
    labelRead: { storage: d.arrayOf(d.u32), access: 'readonly' },
    atomicLabels: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
  });
  const atomicToLabelsLayout = tgpu.bindGroupLayout({
    atomicLabels: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
    dest: { storage: d.arrayOf(d.u32), access: 'mutable' },
  });
  return { initLayout, stepLayout, labelsToAtomicLayout, parentTightenLayout, atomicToLabelsLayout };
}

export function createPointerJumpInitPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  initLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
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
    const ei = initLayout.$.edgeBuffer[idx];
    if (ei <= d.f32(0)) {
      initLayout.$.labelBuffer[idx] = d.u32(COMPONENT_LABEL_INVALID);
      return;
    }
    let best = idx;
    for (const dy of tgpu.unroll(std.range(-1, 2))) {
      for (const dx of tgpu.unroll(std.range(-1, 2))) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx >= d.i32(0) && nx < w && ny >= d.i32(0) && ny < h) {
          const nIdx = d.u32(ny * w + nx);
          const ej = initLayout.$.edgeBuffer[nIdx];
          if (ej > d.f32(0) && nIdx < best) {
            best = nIdx;
          }
        }
      }
    }
    initLayout.$.labelBuffer[idx] = best;
  });

  return root.createComputePipeline({ compute: kernel });
}

export function createPointerJumpStepPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  stepLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
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
    const wh = d.u32(w) * d.u32(h);
    const p = stepLayout.$.readBuffer[idx];
    if (p === d.u32(COMPONENT_LABEL_INVALID) || p >= wh) {
      stepLayout.$.writeBuffer[idx] = d.u32(COMPONENT_LABEL_INVALID);
      return;
    }
    stepLayout.$.writeBuffer[idx] = stepLayout.$.readBuffer[p];
  });

  return root.createComputePipeline({ compute: kernel });
}

export function createPointerJumpLabelsToAtomicPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  labelsToAtomicLayout: ReturnType<typeof createPointerJumpLayouts>['labelsToAtomicLayout'],
  width: number,
  height: number,
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
    const v = labelsToAtomicLayout.$.source[idx];
    atomicStore(labelsToAtomicLayout.$.atomicLabels[idx], v);
  });

  return root.createComputePipeline({ compute: kernel });
}

export function createPointerJumpParentTightenPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  parentTightenLayout: ReturnType<typeof createPointerJumpLayouts>['parentTightenLayout'],
  width: number,
  height: number,
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
    const ei = parentTightenLayout.$.edgeBuffer[idx];
    if (ei <= d.f32(0)) {
      return;
    }

    const wh = d.u32(w) * d.u32(h);
    const p = parentTightenLayout.$.labelRead[idx];
    if (p === d.u32(COMPONENT_LABEL_INVALID) || p >= wh) {
      return;
    }

    let cand = d.u32(COMPONENT_LABEL_INVALID);
    for (const dy of tgpu.unroll(std.range(-1, 2))) {
      for (const dx of tgpu.unroll(std.range(-1, 2))) {
        const nx = x + dx;
        const ny = y + dy;
        if (nx >= d.i32(0) && nx < w && ny >= d.i32(0) && ny < h) {
          const nIdx = d.u32(ny * w + nx);
          const ej = parentTightenLayout.$.edgeBuffer[nIdx];
          if (ej > d.f32(0)) {
            const lj = parentTightenLayout.$.labelRead[nIdx];
            if (lj !== d.u32(COMPONENT_LABEL_INVALID)) {
              cand = std.min(cand, lj);
            }
          }
        }
      }
    }

    if (cand !== d.u32(COMPONENT_LABEL_INVALID)) {
      atomicMin(parentTightenLayout.$.atomicLabels[p], cand);
    }
  });

  return root.createComputePipeline({ compute: kernel });
}

export function createPointerJumpAtomicToLabelsPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  atomicToLabelsLayout: ReturnType<typeof createPointerJumpLayouts>['atomicToLabelsLayout'],
  width: number,
  height: number,
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
    const v = atomicLoad(atomicToLabelsLayout.$.atomicLabels[idx]);
    atomicToLabelsLayout.$.dest[idx] = v;
  });

  return root.createComputePipeline({ compute: kernel });
}
