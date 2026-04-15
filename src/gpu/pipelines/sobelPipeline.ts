// Sobel pipeline: grayBuffer → sobelBuffer
import { tgpu, d } from 'typegpu';
import { sqrt, clamp } from 'typegpu/std';

export function createSobelPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  sobelLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  function sobelLoad(px: number, py: number, w: number, h: number) {
    'use gpu';
    // Clamp to valid [0, w-1] x [0, h-1] for same-padding
    const cx2 = clamp(d.i32(px), d.i32(0), d.i32(w) - d.i32(1));
    const cy2 = clamp(d.i32(py), d.i32(0), d.i32(h) - d.i32(1));
    return sobelLayout.$.grayBuffer[d.u32(cy2) * d.u32(w) + d.u32(cx2)];
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
    const iw = d.i32(w);
    const ih = d.i32(h);

    const tl = sobelLoad(ix - 1, iy - 1, iw, ih);
    const t  = sobelLoad(ix, iy - 1, iw, ih);
    const tr = sobelLoad(ix + 1, iy - 1, iw, ih);
    const ml = sobelLoad(ix - 1, iy, iw, ih);
    const mr = sobelLoad(ix + 1, iy, iw, ih);
    const bl = sobelLoad(ix - 1, iy + 1, iw, ih);
    const b  = sobelLoad(ix, iy + 1, iw, ih);
    const br = sobelLoad(ix + 1, iy + 1, iw, ih);

    const gx = (tr + d.f32(2.0) * mr + br) - (tl + d.f32(2.0) * ml + bl);
    const gy = (bl + d.f32(2.0) * b  + br) - (tl + d.f32(2.0) * t  + tr);
    const magnitude = sqrt(gx * gx + gy * gy);
    sobelLayout.$.sobelBuffer[y * w + x] = magnitude;
  });

  return root.createComputePipeline({ compute: sobelKernel });
}
