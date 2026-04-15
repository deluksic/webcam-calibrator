// Edge threshold filter pipeline: sobelBuffer → filtered edges
import { tgpu, d } from 'typegpu';
import { length } from 'typegpu/std';

export function createEdgeFilterPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  edgeFilterLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  const edgeFilterKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (d.i32(input.gid.x) >= d.i32(width) || d.i32(input.gid.y) >= d.i32(height)) { return; }

    const idx = d.u32(d.i32(input.gid.y) * d.i32(width) + d.i32(input.gid.x));
    const mag = length(edgeFilterLayout.$.sobelBuffer[idx]);
    const threshold = edgeFilterLayout.$.threshold;

    // Keep only edges above threshold, zero out the rest
    if (mag < threshold) {
      edgeFilterLayout.$.filteredBuffer[idx] = d.f32(0);
    } else {
      edgeFilterLayout.$.filteredBuffer[idx] = mag;
    }
  });

  return root.createComputePipeline({ compute: edgeFilterKernel });
}
