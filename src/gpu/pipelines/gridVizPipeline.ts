// Grid visualization pipeline: instanced quad rendering via homography warping
// Sends hasCorners flag to fragment shader for color/style selection
import { tgpu, d, std } from 'typegpu';
import { abs, floor, fract, min, max, dpdx, dpdy, select } from 'typegpu/std';

export const GRID_DIVISIONS = 8;
export const GRID_LINE_WIDTH = 0.06;
export const MAX_INSTANCES = 64;

// 12 f32 per quad packed into 3 vec4f:
// vec4f(h1, h2, h3, h4) + vec4f(h5, h6, h7, h8) + vec4f(hasCorners, 0, 0, 0)
// hasCorners: 1.0 = real corners (blue grid), 0.0 = fallback bbox (red outline, 0.5 opacity)
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
      hasCorners: d.f32,
    },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu';
    // Read 12 params from 3 vec4f per instance
    const H0 = gridVizLayout.$.quadCorners[instanceIndex * 3 + 0];
    const H1 = gridVizLayout.$.quadCorners[instanceIndex * 3 + 1];
    const H2 = gridVizLayout.$.quadCorners[instanceIndex * 3 + 2];
    const h1 = H0.x;
    const h2 = H0.y;
    const h3 = H0.z;
    const h4 = H0.w;
    const h5 = H1.x;
    const h6 = H1.y;
    const h7 = H1.z;
    const h8 = H1.w;
    const hasCorners = H2.x;

    // UV at each triangle-strip vertex: [TL, TR, BL, BR]
    const uv = [d.vec2f(0, 0), d.vec2f(1, 0), d.vec2f(0, 1), d.vec2f(1, 1)][vertexIndex];
    const u = uv.x;
    const v = uv.y;

    // w = h7*u + h8*v + 1
    const w = h7 * u + h8 * v + d.f32(1.0);

    // Apply homography: (x,y) = (h1*u + h2*v + h3) / w, (h4*u + h5*v + h6) / w
    const imgX = (h1 * u + h2 * v + h3) / w;
    const imgY = (h4 * u + h5 * v + h6) / w;

    // Convert to NDC via w in outPos.w — rasterizer divides by w automatically
    const ndcX = imgX / halfW - d.f32(1.0);
    const ndcY = d.f32(1.0) - imgY / halfH;

    return {
      outPos: d.vec4f(ndcX * w, ndcY * w, 0, w),
      uv,
      hasCorners,
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
    const wv = max(abs(scaledDdx), abs(scaledDdy)) + epsilon;

    // a = p + w * 0.5, b = p - w * 0.5
    const a = scaledP + wv * half;
    const b = scaledP - wv * half;

    // i = (floor(a) + min(fract(a)*N, 1) - floor(b) - min(fract(b)*N, 1)) / (N*w)
    const iv = (floor(a) + min(fract(a) * N, d.vec2f(1)) - floor(b) - min(fract(b) * N, d.vec2f(1))) / (N * wv);

    // Extract scalars: (1-i.x) * (1-i.y)
    return (1 - iv.x) * (1 - iv.y);
  };

  // Edge outline mask: returns 1 near UV edges, 0 in interior
  const edgeMask = (uv: d.v2f, ddx: d.v2f, ddy: d.v2f) => {
    'use gpu';
    const edgeWidth = d.f32(0.02);
    // Distance to nearest edge in UV space (accounting for derivatives for AA)
    const distToEdgeX = min(uv.x, d.f32(1.0) - uv.x);
    const distToEdgeY = min(uv.y, d.f32(1.0) - uv.y);
    const minDist = min(distToEdgeX, distToEdgeY);
    // Approx pixel size in UV space
    const pixelUV = max(abs(ddx.x) + abs(ddx.y), abs(ddy.x) + abs(ddy.y));
    // Smooth step from edge
    const t = minDist / (edgeWidth + pixelUV + d.f32(0.001));
    // 1 on edge, 0 in interior
    return d.f32(1.0) - smoothstep(d.f32(0.0), d.f32(1.0), t);
  };

  // Smoothstep
  const smoothstep = (edge0: d.f32, edge1: d.f32, x: d.f32) => {
    'use gpu';
    const t = clamp((x - edge0) / (edge1 - edge0), d.f32(0.0), d.f32(1.0));
    return t * t * (d.f32(3.0) - d.f32(2.0) * t);
  };

  const clamp = (x: d.f32, minVal: d.f32, maxVal: d.f32) => {
    'use gpu';
    return min(max(x, minVal), maxVal);
  };

  const gridVizFrag = tgpu.fragmentFn({
    in: { uv: d.vec2f, outPos: d.builtin.position, hasCorners: d.f32 },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    'use derivatives';
    const uv = i.uv;
    const ddx = dpdx(uv);
    const ddy = dpdy(uv);

    const isFallback = i.hasCorners < d.f32(0.5);

    // Grid pattern for real corners, edge-only for fallback
    const grid = gridTextureGradBox(uv, ddx, ddy);
    const edge = edgeMask(uv, ddx, ddy);
    const mask = select(grid, edge, isFallback);

    // Blue for real corners, red for fallback
    const r = select(d.f32(0.0), d.f32(1.0), isFallback);
    const b = select(d.f32(1.0), d.f32(0.0), isFallback);
    const alpha = select(d.f32(1.0), d.f32(0.5), isFallback);

    return d.vec4f(r, 0, b, alpha * (d.f32(1.0) - mask));
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
