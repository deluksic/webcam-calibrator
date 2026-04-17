// v2 - w=1 hardcoded for corner position test
// Grid visualization pipeline: instanced quad rendering via projective weights
import { tgpu, d, std } from 'typegpu';
import { abs, floor, fract, min, max, dpdx, dpdy } from 'typegpu/std';

export const GRID_DIVISIONS = 8;
export const GRID_LINE_WIDTH = 0.06;
export const MAX_INSTANCES = 64;

// Per-quad data: 4 corner positions (x,y) + 4 weights packed as 4 vec4f
// vec4f(px0, py0, px1, py1) + vec4f(px2, py2, px3, py3) + vec4f(w0, w1, w2, w3) = 3 vec4f per quad
export const gridCornersSchema = d.arrayOf(d.vec4f, MAX_INSTANCES * 3);

export function createGridVizLayouts(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  _quadCornersBuffer: unknown,
) {
  const gridVizLayout = tgpu.bindGroupLayout({
    quadCorners: { storage: gridCornersSchema, access: 'readonly' },
  });
  return { gridVizLayout, gridCornersSchema };
}

export function createGridVizPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  gridVizLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const halfW = d.f32(width) * d.f32(0.5);
  const halfH = d.f32(height) * d.f32(0.5);
  const gridVizVert = tgpu.vertexFn({
    in: {
      vertexIndex: d.builtin.vertexIndex,
      instanceIndex: d.builtin.instanceIndex,
    },
    out: {
      outPos: d.builtin.position,
      uv: d.vec2f,
    },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu';
    // Buffer layout per instance (3 vec4f = 12 f32):
    // entry[0] = {x0, y0, x1, y1} — TL and TR
    // entry[1] = {x2, y2, x3, y3} — BL and BR
    // entry[2] = {w0, w1, w2, w3} — weights for each corner
    const entry0 = gridVizLayout.$.quadCorners[instanceIndex];
    const entry1 = gridVizLayout.$.quadCorners[instanceIndex + d.u32(1)];
    const entry2 = gridVizLayout.$.quadCorners[instanceIndex + d.u32(2)];

    // Triangle strip: TL(0), TR(1), BL(2), BR(3)
    const isRight = std.select(false, true, vertexIndex === d.u32(1));
    const isBottom = std.select(false, true, vertexIndex >= d.u32(2));

    // Select corner position (vec2) based on vertex position
    // Triangle strip: TL(0), TR(1), BL(2), BR(3)
    // entry0.xy = TL, entry0.zw = TR, entry1.xy = BL, entry1.zw = BR
    // isBottom selects row (entry0 for top, entry1 for bottom)
    // isRight selects column (xy for left, zw for right)
    const rowEntry = std.select(entry1, entry0, isBottom);
    const corner = std.select(rowEntry.xy, rowEntry.zw, isRight);
    const w = std.select(
      std.select(entry2.x, entry2.z, isBottom),
      std.select(entry2.y, entry2.w, isBottom),
      isRight
    );

    // Hardcode w=1 for now to test corner positions
    const testW = d.f32(1.0);

    // Apply w scaling to corner position, then convert to NDC
    const scaledX = corner.x * testW;
    const scaledY = corner.y * testW;
    const ndcX = (scaledX * d.f32(2.0) / halfW) - d.f32(1.0);
    const ndcY = d.f32(1.0) - (scaledY * d.f32(2.0) / halfH);

    return {
      outPos: d.vec4f(ndcX, ndcY, 0, d.f32(1)),
      uv: d.vec2f(d.f32(isRight), d.f32(isBottom)),
    };
  });

  // Simple grid - no AA
  const gridTexture = (p: d.v2f) => {
    'use gpu';
    const N = GRID_DIVISIONS;
    const lw = GRID_LINE_WIDTH;
    const f = fract(p * N + lw * 0.5);
    const t = d.vec2f(1.0) * lw;
    const i = std.select(d.vec2f(0), d.vec2f(1), std.lt(f, t));
    return (1 - i.x) * (1 - i.y);
  };

  // Filtered grid - Shadertoy gridTextureGradBox, vectorized
  const gridTextureGradBox = (p: d.v2f, ddx: d.v2f, ddy: d.v2f) => {
    'use gpu';
    const N = GRID_DIVISIONS;
    const half = 0.5;
    const epsilon = 0.01;
    const lw = GRID_LINE_WIDTH;

    const scaledP = p * N + lw * 0.5;
    const scaledDdx = ddx * N;
    const scaledDdy = ddy * N;

    // w = max(|ddx|, |ddy|) + epsilon
    const w = max(abs(scaledDdx), abs(scaledDdy)) + epsilon;

    // a = p + w * 0.5, b = p - w * 0.5
    const a = scaledP + w * half;
    const b = scaledP - w * half;

    // i = (floor(a) + min(fract(a)*N, 1) - floor(b) - min(fract(b)*N, 1)) / (N*w)
    const i = (floor(a) + min(fract(a) * N, d.vec2f(1)) - floor(b) - min(fract(b) * N, d.vec2f(1))) / (N * w);

    // Extract scalars: (1-i.x) * (1-i.y)
    return (1 - i.x) * (1 - i.y);
  };

  const gridVizFrag = tgpu.fragmentFn({
    in: { uv: d.vec2f },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    'use derivatives';
    const ddx = dpdx(i.uv);
    const ddy = dpdy(i.uv);
    const mask = gridTextureGradBox(i.uv, ddx, ddy);

    return d.vec4f(1, 0, 0, 1 - mask);
  });

  return root.createRenderPipeline({
    vertex: gridVizVert,
    fragment: gridVizFrag,
    targets: {
      format: presentationFormat,
      blend: {
        color: { operation: 'add', srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha' },
        alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one-minus-src-alpha' },
      },
    },
    primitive: { topology: 'triangle-strip' },
  });
}