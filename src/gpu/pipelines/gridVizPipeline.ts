// Grid visualization pipeline: instanced quad rendering via homography warping
// Sends hasCorners flag to fragment shader for color/style selection
import { tgpu, d, std } from 'typegpu';
import { abs, floor, fract, min, max, dpdx, dpdy, select, smoothstep } from 'typegpu/std';

export const GRID_DIVISIONS = 8;
export const GRID_LINE_WIDTH = 0.06;
export const MAX_INSTANCES = 64;

// 12 f32 per quad packed into 3 vec4f:
// vec4f(h1, h2, h3, h4) + vec4f(h5, h6, h7, h8) + vec4f(failureMask, debug, debug, debug)
// failureMask: 0.0 = success (hasCorners), > 0 = failure code (for fallback quads)
//   bit 0: insufficient edge pixels
//   bit 1: aspect ratio out of bounds
//   bit 2: line fit failed (null)
//   bit 3: plausibility check failed (convex/ratio/R²)
//   bit 4: intersection out of bounds
// debug floats: extra info (edgePixelCount, minR2, etc.) for shader visualization
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
      failureCode: d.f32,
      edgeCount: d.f32,
      minR2: d.f32,
      intersectionCount: d.f32,
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
    const failureCode = H2.x;
    const edgeCount = H2.y;
    const minR2 = H2.z;
    const intersectionCount = H2.w;

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
      failureCode,
      edgeCount,
      minR2,
      intersectionCount,
    };
  });

  // Filtered grid - Shadertoy gridTextureGradBox, vectorized
  const gridTextureGradBox = (p: d.v2f, ddx: d.v2f, ddy: d.v2f, N: number) => {
    'use gpu';
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

  const gridVizFrag = tgpu.fragmentFn({
    in: { uv: d.vec2f, outPos: d.builtin.position, failureCode: d.f32, edgeCount: d.f32, minR2: d.f32, intersectionCount: d.f32 },
    out: d.vec4f,
  })((i) => {
    'use gpu';
    'use derivatives';
    const uv = i.uv;
    const ddx = dpdx(uv);
    const ddy = dpdy(uv);

    // failureCode == 0 means success (real corners)
    const isSuccess = i.failureCode < d.f32(0.5);

    // Success: N-division grid, full opacity, blue
    // Failure: single-cell outline (N=1), 0.3 opacity, gray
    const mask = select(
      gridTextureGradBox(uv, ddx, ddy, GRID_DIVISIONS),
      gridTextureGradBox(uv, ddx, ddy, 1),
      isSuccess,
    );

    const r = select(d.f32(0.5), d.f32(0.0), isSuccess);
    const g = select(d.f32(0.5), d.f32(0.0), isSuccess);
    const b = select(d.f32(1.0), d.f32(0.5), isSuccess);
    const alpha = select(d.f32(0.3), d.f32(1.0), isSuccess);

    return d.vec4f(r, g, b, alpha * mask);
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
