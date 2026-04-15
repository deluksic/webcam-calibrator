// Edge threshold filter pipeline: sobelBuffer → filtered edges
import { tgpu, d } from 'typegpu';

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
    if (input.gid.x >= d.u32(width) || input.gid.y >= d.u32(height)) { return; }

    const idx = input.gid.y * d.u32(width) + input.gid.x;
    const mag = edgeFilterLayout.$.sobelBuffer[idx];
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
