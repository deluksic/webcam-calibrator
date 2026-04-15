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
    const cx2 = clamp(px, 0, w - 1);
    const cy2 = clamp(py, 0, h - 1);
    return sobelLayout.$.grayBuffer[cy2 * w + cx2];
  }

  const sobelKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (input.gid.x >= d.u32(width) || input.gid.y >= d.u32(height)) { return; }

    const x = d.i32(input.gid.x);
    const y = d.i32(input.gid.y);
    const w = d.i32(width);
    const h = d.i32(height);

    const tl = sobelLoad(x - 1, y - 1, w, h);
    const t = sobelLoad(x, y - 1, w, h);
    const tr = sobelLoad(x + 1, y - 1, w, h);
    const ml = sobelLoad(x - 1, y, w, h);
    const mr = sobelLoad(x + 1, y, w, h);
    const bl = sobelLoad(x - 1, y + 1, w, h);
    const b = sobelLoad(x, y + 1, w, h);
    const br = sobelLoad(x + 1, y + 1, w, h);

    const gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
    const gy = (bl + 2 * b + br) - (tl + 2 * t + tr);
    const magnitude = sqrt(gx * gx + gy * gy);
    sobelLayout.$.sobelBuffer[y * width + x] = magnitude;
  });

  return root.createComputePipeline({ compute: sobelKernel });
}
