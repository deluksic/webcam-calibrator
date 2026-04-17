// Grid visualization pipeline: instanced quad rendering from quadCornersBuffer
import { tgpu, d } from 'typegpu';
import { abs, floor, fract, min, max, dpdx, dpdy } from 'typegpu/std';

export const GRID_DIVISIONS = 6;
export const GRID_LINE_WIDTH = 0.06;
export const MAX_INSTANCES = 64;

// Storage buffer array type — must match what camera.ts binds
export const gridCornersSchema = d.arrayOf(d.vec4f, MAX_INSTANCES);

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
    // Read this instance's bounding box from the bind group
    const corners = gridVizLayout.$.quadCorners[instanceIndex];
    const minX = corners.x;
    const minY = corners.y;
    const maxX = corners.z;
    const maxY = corners.w;

    // Triangle strip corners: TL (0), TR (1), BL (2), BR (3)
    const localX = [minX, maxX, minX, maxX][vertexIndex];
    const localY = [minY, minY, maxY, maxY][vertexIndex];

    const ndcX = (localX / d.f32(width)) * d.f32(2) - d.f32(1);
    const ndcY = d.f32(1) - (localY / d.f32(height)) * d.f32(2);

    const uvX = (localX - minX) / (maxX - minX);
    const uvY = (localY - minY) / (maxY - minY);

    return {
      outPos: d.vec4f(ndcX, ndcY, 0, 1),
      uv: d.vec2f(uvX, uvY),
    };
  });

  // Filtered grid - Shadertoy gridTextureGradBox, vectorized
  const gridTextureGradBox = (p: d.v2f, ddx: d.v2f, ddy: d.v2f) => {
    'use gpu';
    const N = d.f32(GRID_DIVISIONS);
    const half = d.f32(0.5);
    const epsilon = d.f32(0.01);
    const one2 = d.vec2f(1.0);

    // w = max(|ddx|, |ddy|) + epsilon
    const w = max(abs(ddx), abs(ddy)).add(epsilon);

    // a = p + w * 0.5, b = p - w * 0.5
    const a = p.add(w.mul(half));
    const b = p.sub(w.mul(half));

    // i = (floor(a) + min(fract(a)*N, 1) - floor(b) - min(fract(b)*N, 1)) / (N*w)
    const i = floor(a).add(min(fract(a).mul(N), one2)).sub(floor(b)).sub(min(fract(b).mul(N), one2)).div(w.mul(N));

    // Extract scalars: (1-i.x) * (1-i.y)
    return (one2.x - i.x) * (one2.x - i.y);
  };

  const gridVizFrag = tgpu.fragmentFn({
    in: { uv: d.vec2f },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    'use derivatives';
    const ddx = d.vec2f(dpdx(i.uv.x), dpdx(i.uv.y));
    const ddy = d.vec2f(dpdy(i.uv.x), dpdy(i.uv.y));
    const mask = gridTextureGradBox(i.uv, ddx, ddy);

    if (mask > d.f32(0.5)) return d.vec4f(0, 1, 1, 1); // opaque cyan
    return d.vec4f(0, 0, 0, 0); // transparent
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