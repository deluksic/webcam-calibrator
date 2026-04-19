// Grid visualization pipeline: instanced quad rendering via homography warping
import { tgpu, d, std } from 'typegpu';
import { abs, floor, fract, min, max, dpdx, dpdy, select, mul } from 'typegpu/std';

export const GRID_DIVISIONS = 8;
export const GRID_LINE_WIDTH = 0.06;
export const MAX_INSTANCES = 64;

const QuadDebug = d.struct({
  failureCode: d.u32,
  edgePixelCount: d.f32,
  minR2: d.f32,
  intersectionCount: d.f32,
});

const QuadData = d.struct({
  homography: d.mat3x3f,
  debug: QuadDebug,
});

export type QuadData = d.Infer<typeof QuadData>;

export const GridDataSchema = d.arrayOf(QuadData, MAX_INSTANCES);

export function createGridVizLayouts(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  _quadCornersBuffer: unknown,
) {
  const gridVizLayout = tgpu.bindGroupLayout({
    quads: { storage: GridDataSchema, access: 'readonly' },
  });
  return { gridVizLayout };
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
      failureCode: d.interpolate('flat', d.u32),
      edgeCount: d.f32,
      minR2: d.f32,
      intersectionCount: d.f32,
    },
  })(({ vertexIndex, instanceIndex }) => {
    const quad = gridVizLayout.$.quads[instanceIndex];
    const H = quad.homography;
    const debug = quad.debug;

    const uv = [d.vec2f(0, 0), d.vec2f(1, 0), d.vec2f(0, 1), d.vec2f(1, 1)][vertexIndex];
    const imgPos = mul(d.vec3f(uv, 1), H);
    const imgX = imgPos.x;
    const imgY = imgPos.y;
    const w = imgPos.z;

    const ndcX = imgX / halfW - d.f32(1.0);
    const ndcY = d.f32(1.0) - imgY / halfH;

    return {
      outPos: d.vec4f(ndcX * w, ndcY * w, 0, w),
      uv,
      failureCode: debug.failureCode,
      edgeCount: debug.edgePixelCount,
      minR2: debug.minR2,
      intersectionCount: debug.intersectionCount,
    };
  });

  const gridTextureGradBox = (p: d.v2f, ddx: d.v2f, ddy: d.v2f, N: number) => {
    'use gpu';
    const half = 0.5;
    const epsilon = 0.01;
    const lw = GRID_LINE_WIDTH;

    const scaledP = p * d.f32(N) + lw * 0.5;
    const scaledDdx = ddx * d.f32(N);
    const scaledDdy = ddy * d.f32(N);

    const wv = max(abs(scaledDdx), abs(scaledDdy)) + epsilon;

    const a = scaledP + wv * half;
    const b = scaledP - wv * half;

    const iv = (floor(a) + min(fract(a) * d.f32(N), d.vec2f(1)) - floor(b) - min(fract(b) * d.f32(N), d.vec2f(1))) / (d.f32(N) * wv);

    const inside = (1 - iv.x) * (1 - iv.y);
    return 1 - inside;
  };

  const gridVizFrag = tgpu.fragmentFn({
    in: { uv: d.vec2f, outPos: d.builtin.position, failureCode: d.interpolate('flat', d.u32) },
    out: d.vec4f,
  })(({ uv, failureCode }) => {
    const ddx = dpdx(uv);
    const ddy = dpdy(uv);

    if (failureCode === 0) {
      const grid = gridTextureGradBox(uv, ddx, ddy, GRID_DIVISIONS);
      return d.vec4f(d.f32(0.0), d.f32(1.0), d.f32(0.0), grid);
    }

    const code = failureCode;
    const fc = d.f32((code & 1) > 0);
    const ec = d.f32((code & 2) > 0);
    const lc = d.f32((code & 4) > 0);
    const pc = d.f32((code & 8) > 0);
    const nc = d.f32((code & 16) > 0);

    const failR = fc + lc + nc;
    const failG = ec + pc + lc;
    const failB = fc + ec + pc;

    const cell = gridTextureGradBox(uv, ddx, ddy, 1);
    return d.vec4f(failR, failG, failB, d.f32(0.2) + d.f32(0.8) * cell);
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