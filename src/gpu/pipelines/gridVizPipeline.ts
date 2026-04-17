// Grid visualization pipeline: instanced quad rendering via projective weights
import { tgpu, d, std } from 'typegpu';
import { abs, floor, fract, min, max, dpdx, dpdy } from 'typegpu/std';

export const GRID_DIVISIONS = 8;
export const GRID_LINE_WIDTH = 0.06;
export const MAX_INSTANCES = 64;

// CornerInfo: 4 vec3f fields = 4×16 bytes = 64 bytes in storage buffers
// Each vec3f is 12 bytes but aligned to 16 in storage. The pad field aligns c1→index 4.
const CornerInfo = d.struct({
  c0: d.vec3f, // TL: x, y, w (index 0-2)
  p0: d.f32,   // pad to align c1→index 4
  c1: d.vec3f, // TR: x, y, w (index 4-6)
  p1: d.f32,   // pad to align c2→index 8
  c2: d.vec3f, // BL: x, y, w (index 8-10)
  p2: d.f32,   // pad to align c3→index 12
  c3: d.vec3f, // BR: x, y, w (index 12-14)
  p3: d.f32,   // pad (index 15, unused)
});

export const gridCornersSchema = d.arrayOf(CornerInfo, MAX_INSTANCES);

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
    const quad = gridVizLayout.$.quadCorners[instanceIndex];
    const corners = [d.vec3f(quad.c0), d.vec3f(quad.c1), d.vec3f(quad.c2), d.vec3f(quad.c3)];
    const uvs = [d.vec2f(d.f32(0), d.f32(0)), d.vec2f(d.f32(1), d.f32(0)), d.vec2f(d.f32(0), d.f32(1)), d.vec2f(d.f32(1), d.f32(1))];
    const corner = corners[vertexIndex];
    const uv = uvs[vertexIndex];

    // Convert to NDC
    const ndcX = corner.x / halfW - d.f32(1.0);
    const ndcY = d.f32(1.0) - corner.y / halfH;

    return {
      outPos: d.vec4f(ndcX, ndcY, 0, d.f32(1)),
      uv,
    };
  });

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

    return d.vec4f(0, 0, 1, 1 - mask);
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
