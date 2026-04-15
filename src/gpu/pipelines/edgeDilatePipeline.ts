// Merge edge responses only along the local edge tangent (perpendicular to Sobel gradient).
// Skips neighbors that lie mostly along the gradient normal so the mask does not thicken.
import { tgpu, d, std } from 'typegpu';
import { abs, length, max, sqrt } from 'typegpu/std';
import { EDGE_DILATE_THRESHOLD } from './constants';

export function createEdgeDilatePipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  edgeDilateLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  const dilateKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (d.i32(input.gid.x) >= d.i32(width) || d.i32(input.gid.y) >= d.i32(height)) { return; }

    const x = d.i32(input.gid.x);
    const y = d.i32(input.gid.y);
    const w = d.i32(width);
    const h = d.i32(height);
    const wU32 = d.u32(w);
    const idx = d.u32(y) * wU32 + d.u32(x);

    const g = edgeDilateLayout.$.grad[idx];
    const gm = length(g);
    const eps = d.f32(1e-6);
    let nx = d.f32(1);
    let ny = d.f32(0);
    if (gm > eps) {
      nx = g.x / gm;
      ny = g.y / gm;
    }

    let m = edgeDilateLayout.$.src[idx];
    if (gm <= eps) {
      edgeDilateLayout.$.dst[idx] = m;
      return;
    }

    for (const iy of tgpu.unroll(std.range(3))) {
      for (const ix of tgpu.unroll(std.range(3))) {
        const dx = d.i32(ix) - d.i32(1);
        const dy = d.i32(iy) - d.i32(1);
        if (dx !== d.i32(0) || dy !== d.i32(0)) {
          const nx2 = x + dx;
          const ny2 = y + dy;
          if (nx2 >= d.i32(0) && nx2 < w && ny2 >= d.i32(0) && ny2 < h) {
            const fx = d.f32(dx);
            const fy = d.f32(dy);
            const ulen = sqrt(fx * fx + fy * fy);
            const ux = fx / ulen;
            const uy = fy / ulen;
            const align = abs(ux * nx + uy * ny);
            if (align <= d.f32(EDGE_DILATE_THRESHOLD)) {
              const nIdx = d.u32(ny2) * wU32 + d.u32(nx2);
              m = max(m, edgeDilateLayout.$.src[nIdx]);
            }
          }
        }
      }
    }
    edgeDilateLayout.$.dst[idx] = m;
  });

  return root.createComputePipeline({ compute: dilateKernel });
}
