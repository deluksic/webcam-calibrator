// Grid visualization pipeline: instanced quad rendering via homography warping
import { tgpu, d, std } from 'typegpu';
import { abs, floor, fract, min, max, dpdx, dpdy, select, mul } from 'typegpu/std';

export const GRID_DIVISIONS = 8;
export const GRID_LINE_WIDTH = 0.06;
export const MAX_INSTANCES = 1024;

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

/** 0 = legacy RGB fail tint; 1 = interrogate FAIL_INSUFFICIENT_EDGES (red hit / black miss); 2 = interrogate FAIL_LINE_FIT_FAILED (blue). */
export type GridVizFailInterrogateMode = 0 | 1 | 2;

export function createGridVizLayouts(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  _quadCornersBuffer: unknown,
) {
  const gridVizLayout = tgpu.bindGroupLayout({
    quads: { storage: GridDataSchema, access: 'readonly' },
    failInterrogate: { uniform: d.u32 },
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

    const uvs = [d.vec2f(0, 0), d.vec2f(1, 0), d.vec2f(0, 1), d.vec2f(1, 1)];
    const uv = uvs[vertexIndex];
    const imgPos = mul(H, d.vec3f(uv, 1));
    const imgX = imgPos.x;
    const imgY = imgPos.y;
    const w = imgPos.z;

    // imgPos is homogeneous (x', y', w'); Cartesian image coords are x'/w', y'/w'.
    // Emit clip so that clip.xy / w = NDC with origin at image center, y flipped.
    const clipX = 2 * imgX / width - w;
    const clipY = w - 2 * imgY / height;

    return {
      outPos: d.vec4f(clipX, clipY, 0, w),
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
    const mode = gridVizLayout.$.failInterrogate;

    if (failureCode === d.u32(0)) {
      const grid = gridTextureGradBox(uv, ddx, ddy, GRID_DIVISIONS);
      return d.vec4f(d.f32(0.0), d.f32(1.0), d.f32(0.0), grid);
    }

    const grid = gridTextureGradBox(uv, ddx, ddy, GRID_DIVISIONS);
    const a = d.f32(0.2) + d.f32(0.75) * grid;
    const mask = 0b1000;
    if ((failureCode & mask) !== d.u32(0)) {
      if ((failureCode ^ mask) === d.u32(0)) {
        return d.vec4f(d.f32(1), d.f32(0), d.f32(0), a);
      }
      return d.vec4f(d.f32(0), d.f32(0.25), d.f32(1), a);
    }

    return d.vec4f(d.f32(0), d.f32(0), d.f32(0), a);
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