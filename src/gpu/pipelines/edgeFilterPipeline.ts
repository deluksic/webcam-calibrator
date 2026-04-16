// Non-Maximum Suppression (NMS) pipeline: sobelBuffer → filteredBuffer
//
// Keeps only pixels that are local maxima along the gradient tangent direction.
// Produces thin, single-pixel edges instead of thick noisy blobs.
//
// Output: filteredBuffer[i] = suppressed magnitude (0 if not a local max or below threshold).
import { tgpu, d } from 'typegpu';
import { abs, length, select } from 'typegpu/std';

export function createEdgeFilterPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  edgeFilterLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu';
    if (d.i32(input.gid.x) >= d.i32(width) || d.i32(input.gid.y) >= d.i32(height)) { return; }

    const x = d.i32(input.gid.x);
    const y = d.i32(input.gid.y);
    const w = d.i32(width);
    const h = d.i32(height);
    const idx = d.u32(y) * d.u32(w) + d.u32(x);

    const g = edgeFilterLayout.$.sobelBuffer[idx];
    const gm = length(g);
    const threshold = edgeFilterLayout.$.threshold;

    if (gm < threshold) {
      edgeFilterLayout.$.filteredBuffer[idx] = d.f32(0);
      return;
    }

    // NMS: compare against neighbors along the edge tangent (relaxed for edge continuity).
    const abx = abs(g.x);
    const aby = abs(g.y);

    let tanDx = d.i32(0);
    let tanDy = d.i32(0);
    if (abx >= aby) { tanDx = d.i32(0); tanDy = d.i32(1); }
    else { tanDx = d.i32(1); tanDy = d.i32(0); }

    const relaxFactor = d.f32(0.85);
    let suppressed = false;

    // Tangent neighbor at dist=1, sign=+1
    {
      const nx = x + tanDx;
      const ny = y + tanDy;
      if (nx >= d.i32(0) && nx < w && ny >= d.i32(0) && ny < h) {
        const nIdx = d.u32(ny) * d.u32(w) + d.u32(nx);
        const gmN = length(edgeFilterLayout.$.sobelBuffer[nIdx]);
        if (gm <= gmN * relaxFactor) { suppressed = true; }
      }
    }
    // Tangent neighbor at dist=1, sign=-1
    {
      const nx = x - tanDx;
      const ny = y - tanDy;
      if (nx >= d.i32(0) && nx < w && ny >= d.i32(0) && ny < h) {
        const nIdx = d.u32(ny) * d.u32(w) + d.u32(nx);
        const gmN = length(edgeFilterLayout.$.sobelBuffer[nIdx]);
        if (gm <= gmN * relaxFactor) { suppressed = true; }
      }
    }
    // Tangent neighbor at dist=2, sign=+1
    {
      const nx = x + tanDx * d.i32(2);
      const ny = y + tanDy * d.i32(2);
      if (nx >= d.i32(0) && nx < w && ny >= d.i32(0) && ny < h) {
        const nIdx = d.u32(ny) * d.u32(w) + d.u32(nx);
        const gmN = length(edgeFilterLayout.$.sobelBuffer[nIdx]);
        if (gm <= gmN * relaxFactor) { suppressed = true; }
      }
    }
    // Tangent neighbor at dist=2, sign=-1
    {
      const nx = x - tanDx * d.i32(2);
      const ny = y - tanDy * d.i32(2);
      if (nx >= d.i32(0) && nx < w && ny >= d.i32(0) && ny < h) {
        const nIdx = d.u32(ny) * d.u32(w) + d.u32(nx);
        const gmN = length(edgeFilterLayout.$.sobelBuffer[nIdx]);
        if (gm <= gmN * relaxFactor) { suppressed = true; }
      }
    }

    edgeFilterLayout.$.filteredBuffer[idx] = select(gm, d.f32(0), suppressed);
  });

  return root.createComputePipeline({ compute: kernel });
}
