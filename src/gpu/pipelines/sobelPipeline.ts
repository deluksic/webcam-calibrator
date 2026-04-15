// Sobel pipeline: grayBuffer → sobelBuffer
import { tgpu, d } from 'typegpu';
import { sqrt } from 'typegpu/std';

export function createSobelPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  sobelLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  function sobelLoad(px: number, py: number, w: number) {
    'use gpu';
    if (px >= w || py >= w) { return d.f32(0); }
    return sobelLayout.$.grayBuffer[py * w + px];
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

    const tl = sobelLoad(x - d.u32(1), y - d.u32(1), w);
    const t  = sobelLoad(x, y - d.u32(1), w);
    const tr = sobelLoad(x + d.u32(1), y - d.u32(1), w);
    const ml = sobelLoad(x - d.u32(1), y, w);
    const mr = sobelLoad(x + d.u32(1), y, w);
    const bl = sobelLoad(x - d.u32(1), y + d.u32(1), w);
    const b  = sobelLoad(x, y + d.u32(1), w);
    const br = sobelLoad(x + d.u32(1), y + d.u32(1), w);

    const gx = (tr + d.f32(2.0) * mr + br) - (tl + d.f32(2.0) * ml + bl);
    const gy = (bl + d.f32(2.0) * b  + br) - (tl + d.f32(2.0) * t  + tr);
    const magnitude = sqrt(gx * gx + gy * gy);
    sobelLayout.$.sobelBuffer[y * w + x] = magnitude;
  });

  return root.createComputePipeline({ compute: sobelKernel });
}
