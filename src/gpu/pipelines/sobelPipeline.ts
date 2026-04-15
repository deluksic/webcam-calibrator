// Sobel pipeline: grayBuffer → sobelBuffer
import { tgpu, d } from 'typegpu';
import { clamp } from 'typegpu/std';

export function createSobelPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  sobelLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  function sobelLoad(px: number, py: number, w: number, h: number) {
    'use gpu';
    const wi = d.i32(w);
    const hi = d.i32(h);
    const pxi = d.i32(px);
    const pyi = d.i32(py);
    const cx2 = clamp(pxi, d.i32(0), wi - d.i32(1));
    const cy2 = clamp(pyi, d.i32(0), hi - d.i32(1));
    return sobelLayout.$.grayBuffer[d.u32(cy2 * wi + cx2)];
  }

  const sobelKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (d.i32(input.gid.x) >= d.i32(width) || d.i32(input.gid.y) >= d.i32(height)) { return; }

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
    sobelLayout.$.sobelBuffer[d.u32(y * w + x)] = d.vec2f(gx, gy);
  });

  return root.createComputePipeline({ compute: sobelKernel });
}
