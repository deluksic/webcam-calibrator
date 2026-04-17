// Non-Maximum Suppression (NMS) pipeline: sobelBuffer → filteredBuffer
//
// Keeps only pixels that are local maxima along the edge tangent direction.
// Edge tangent = gradient direction rotated 90°: tangent = (-gy/gm, gx/gm).
// Uses bilinear interpolation for sub-pixel neighbor sampling.
//
// Output: filteredBuffer[i] = suppressed gradient (vec2f; zero if not a local max or below threshold).
// The edges display pipeline computes magnitude from length() on-the-fly.
import { tgpu, d, std } from 'typegpu';
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
      edgeFilterLayout.$.filteredBuffer[idx] = d.vec2f(d.f32(0), d.f32(0));
      return;
    }

    // Gradient direction (perpendicular to edge) for NMS — round to nearest int pixel
    const gxn = g.x / gm;
    const gyn = g.y / gm;
    const stepX = d.i32(std.select(gxn - d.f32(0.5), gxn + d.f32(0.5), gxn >= d.f32(0)));
    const stepY = d.i32(std.select(gyn - d.f32(0.5), gyn + d.f32(0.5), gyn >= d.f32(0)));
    const nxP = x + stepX;
    const nyP = y + stepY;
    const nxN = x - stepX;
    const nyN = y - stepY;

    // Collect neighbor magnitudes (0 if out of bounds)
    let gmPos = d.f32(0);
    if (nxP >= d.i32(0) && nxP < w && nyP >= d.i32(0) && nyP < h) {
      const idxP = d.u32(nyP) * d.u32(w) + d.u32(nxP);
      gmPos = length(edgeFilterLayout.$.sobelBuffer[idxP]);
    }
    let gmNeg = d.f32(0);
    if (nxN >= d.i32(0) && nxN < w && nyN >= d.i32(0) && nyN < h) {
      const idxN = d.u32(nyN) * d.u32(w) + d.u32(nxN);
      gmNeg = length(edgeFilterLayout.$.sobelBuffer[idxN]);
    }

    // Only suppress if neighbor is meaningfully stronger (not just slightly)
    const strongFactor = d.f32(1.05); // neighbor must be >5% stronger to suppress
    const suppressed = gmPos > gm * strongFactor || gmNeg > gm * strongFactor;

    // Store suppressed gradient (edges pipeline derives magnitude via length())
    edgeFilterLayout.$.filteredBuffer[idx] = select(g, d.vec2f(d.f32(0), d.f32(0)), suppressed);
  });

  return root.createComputePipeline({ compute: kernel });
}
