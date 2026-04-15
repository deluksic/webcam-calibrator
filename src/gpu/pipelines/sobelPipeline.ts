// Sobel pipeline: grayBuffer → sobelBuffer
import { tgpu, d } from 'typegpu';
import { sqrt, select } from 'typegpu/std';

export function createSobelPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  sobelLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  function sobelLoad(px: d.i32, py: d.i32, w: d.u32, h: d.u32) {
    'use gpu';
    // Clamp to valid [0, w-1] x [0, h-1] for same-padding
    const clampedX = select(d.u32(px), d.u32(0), px < d.i32(0));
    const cx2 = select(w - d.u32(1), clampedX, px >= d.i32(w));
    const clampedY = select(d.u32(py), d.u32(0), py < d.i32(0));
    const cy2 = select(h - d.u32(1), clampedY, py >= d.i32(h));
    return sobelLayout.$.grayBuffer[cy2 * w + cx2];
  }

  const sobelKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (input.gid.x >= d.u32(width) || input.gid.y >= d.u32(height)) { return; }

    const x = input.gid.x;
    const y = input.gid.y;
    const w = d.u32(width);
    const h = d.u32(height);

    const ix = d.i32(x);
    const iy = d.i32(y);

    const tl = sobelLoad(ix - d.i32(1), iy - d.i32(1), w, h);
    const t  = sobelLoad(ix, iy - d.i32(1), w, h);
    const tr = sobelLoad(ix + d.i32(1), iy - d.i32(1), w, h);
    const ml = sobelLoad(ix - d.i32(1), iy, w, h);
    const mr = sobelLoad(ix + d.i32(1), iy, w, h);
    const bl = sobelLoad(ix - d.i32(1), iy + d.i32(1), w, h);
    const b  = sobelLoad(ix, iy + d.i32(1), w, h);
    const br = sobelLoad(ix + d.i32(1), iy + d.i32(1), w, h);

    const gx = (tr + d.f32(2.0) * mr + br) - (tl + d.f32(2.0) * ml + bl);
    const gy = (bl + d.f32(2.0) * b  + br) - (tl + d.f32(2.0) * t  + tr);
    const magnitude = sqrt(gx * gx + gy * gy);
    sobelLayout.$.sobelBuffer[y * w + x] = magnitude;
  });

  return root.createComputePipeline({ compute: sobelKernel });
}
