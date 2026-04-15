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
    const cx2 = clamp(px, d.u32(0), w - d.u32(1));
    const cy2 = clamp(py, d.u32(0), h - d.u32(1));
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

    const tl = sobelLoad(x - d.u32(1), y - d.u32(1), w, h);
    const t  = sobelLoad(x, y - d.u32(1), w, h);
    const tr = sobelLoad(x + d.u32(1), y - d.u32(1), w, h);
    const ml = sobelLoad(x - d.u32(1), y, w, h);
    const mr = sobelLoad(x + d.u32(1), y, w, h);
    const bl = sobelLoad(x - d.u32(1), y + d.u32(1), w, h);
    const b  = sobelLoad(x, y + d.u32(1), w, h);
    const br = sobelLoad(x + d.u32(1), y + d.u32(1), w, h);

    const gx = (tr + d.f32(2.0) * mr + br) - (tl + d.f32(2.0) * ml + bl);
    const gy = (bl + d.f32(2.0) * b  + br) - (tl + d.f32(2.0) * t  + tr);
    const magnitude = sqrt(gx * gx + gy * gy);
    sobelLayout.$.sobelBuffer[y * w + x] = magnitude;
  });

  return root.createComputePipeline({ compute: sobelKernel });
}
