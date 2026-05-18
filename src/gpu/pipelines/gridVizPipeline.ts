// Grid viz: homography clip + w for perspective-correct UV; degenerate H uses screenCorners (affine fallback).
import type { ColorAttachment, TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { abs, floor, fract, length, min, max, dpdx, dpdy, mul } from 'typegpu/std'

import { MAX_EDGES_PER_LABEL } from '@/gpu/lineFitThresholds'

import { stableHashToRgb01 } from '@/lib/hashStableColor'

export const GRID_DIVISIONS = 8
export const GRID_LINE_WIDTH = 0.06
export const MAX_INSTANCES = 1024

/** App/UI cap for detected quads per frame; same as instance buffer length (`MAX_INSTANCES`). */
export const MAX_DETECTED_TAGS = MAX_INSTANCES

export const QuadDebug = d.struct({
  failureCode: d.u32,
  edgePixelCount: d.f32,
  minR2: d.f32,
  intersectionCount: d.f32,
})

/** Triangle-strip order: [0]=TL, [1]=TR, [2]=BL, [3]=BR — matches `Corners` / unit-square UVs. */
export const QuadScreenCorners = d.arrayOf(d.vec2f, MAX_EDGES_PER_LABEL)

export const QuadDataGpu = d.struct({
  /** Unit square → image; non-degenerate H gives perspective-correct clip + UV. */
  homography: d.mat3x3f,
  /** Degenerate H only: affine quad from line intersections (may show a strip diagonal kink). */
  screenCorners: QuadScreenCorners,
  debug: QuadDebug,
  /** `0xFFFFFFFF` = unknown — solid black (no hash). Same convention as CPU. */
  decodedTagId: d.u32,
  /** Clockwise quarter-turns from canonical orientation (0-3). */
  decodedRotation: d.u32,
})

export type QuadData = d.Infer<typeof QuadDataGpu>

/** Sentinel: no decoded id (GPU draws black, no hash). */
export const DECODED_TAG_ID_UNKNOWN = 0xffff_ffff

/** Vote pattern passed quality gate but dictionary decode failed — distinct tint in grid viz + “?” label. */
export const DECODED_TAG_ID_DICT_MISS = 0xffff_fffe

export const GridDataSchema = d.arrayOf(QuadDataGpu, MAX_INSTANCES)

/** 0 = legacy RGB fail tint; 1 = interrogate FAIL_INSUFFICIENT_EDGES (red hit / black miss); 2 = interrogate FAIL_LINE_FIT_FAILED (blue). */
export type GridVizFailInterrogateMode = 0 | 1 | 2

export function createGridVizLayouts() {
  const gridVizLayout = tgpu.bindGroupLayout({
    quads: { storage: GridDataSchema, access: 'readonly' },
    failInterrogate: { uniform: d.u32 },
  })
  return { gridVizLayout }
}

/** Matches GPU `quadCornerOrder` / `FAIL_*` bitmask order; first matching bit wins (high → low). */
const gridVizFailureTintRgb = tgpu.fn([d.u32], d.vec3f)((failureCode) => {
  'use gpu'
  const insufficient = d.u32(1 << 0)
  const aspect = d.u32(1 << 1)
  const lineFit = d.u32(1 << 2)
  const plausibility = d.u32(1 << 3)
  const noIntersections = d.u32(1 << 4)
  if ((failureCode & noIntersections) !== d.u32(0)) {
    return d.vec3f(0.95, 0.38, 0.24)
  }
  if ((failureCode & plausibility) !== d.u32(0)) {
    return d.vec3f(0.78, 0.28, 0.95)
  }
  if ((failureCode & lineFit) !== d.u32(0)) {
    return d.vec3f(0.22, 0.48, 0.98)
  }
  if ((failureCode & aspect) !== d.u32(0)) {
    return d.vec3f(0.32, 0.82, 0.44)
  }
  if ((failureCode & insufficient) !== d.u32(0)) {
    return d.vec3f(0.98, 0.74, 0.16)
  }
  return d.vec3f(0.55, 0.55, 0.6)
})

export function createGridVizPipeline(
  root: TgpuRoot,
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
      decodedTagId: d.interpolate('flat', d.u32),
    },
  })(({ vertexIndex, instanceIndex }) => {
    const quad = gridVizLayout.$.quads[instanceIndex]!
    const H = quad.homography
    const debug = quad.debug

    const uvs = [d.vec2f(0, 0), d.vec2f(1, 0), d.vec2f(0, 1), d.vec2f(1, 1)]
    const uv = uvs[vertexIndex]!

    const e0 = mul(H, d.vec3f(1, 0, 0))
    const e1 = mul(H, d.vec3f(0, 1, 0))
    const hDegenerate = length(e0) + length(e1) < d.f32(1e-6)

    let clipX = d.f32(0)
    let clipY = d.f32(0)
    let clipW = d.f32(1)
    if (hDegenerate) {
      const p = quad.screenCorners[vertexIndex]!
      clipX = (2 * p.x) / width - 1
      clipY = 1 - (2 * p.y) / height
      clipW = 1
    } else {
      const imgPos = mul(H, d.vec3f(uv, 1))
      const imgX = imgPos.x
      const imgY = imgPos.y
      const w = imgPos.z
      clipX = (2 * imgX) / width - w
      clipY = w - (2 * imgY) / height
      clipW = w
    }

    return {
      outPos: d.vec4f(clipX, clipY, 0, clipW),
      uv,
      failureCode: debug.failureCode,
      edgeCount: debug.edgePixelCount,
      minR2: debug.minR2,
      intersectionCount: debug.intersectionCount,
      decodedTagId: quad.decodedTagId,
    }
  })

  const gridTextureGradBox = (p: d.v2f, ddx: d.v2f, ddy: d.v2f, N: number) => {
    'use gpu'
    const half = 0.5
    const epsilon = 0.01
    const lw = GRID_LINE_WIDTH

    const scaledP = p * d.f32(N) + lw * 0.5
    const scaledDdx = ddx * d.f32(N)
    const scaledDdy = ddy * d.f32(N)

    const wv = max(abs(scaledDdx), abs(scaledDdy)) + epsilon

    const a = scaledP + wv * half
    const b = scaledP - wv * half

    const iv =
      (floor(a) + min(fract(a) * d.f32(N), d.vec2f(1)) - floor(b) - min(fract(b) * d.f32(N), d.vec2f(1))) /
      (d.f32(N) * wv)

    const inside = (1 - iv.x) * (1 - iv.y)
    return 1 - inside
  }

  const gridVizFrag = tgpu.fragmentFn({
    in: {
      uv: d.vec2f,
      failureCode: d.interpolate('flat', d.u32),
      decodedTagId: d.interpolate('flat', d.u32),
    },
    out: d.vec4f,
  })(({ uv, failureCode, decodedTagId }) => {
    'use gpu'
    const ddx = dpdx(uv)
    const ddy = dpdy(uv)
    const grid = gridTextureGradBox(uv, ddx, ddy, GRID_DIVISIONS)
    const a = 0.2 + 0.75 * grid

    if (failureCode === d.u32(0) && decodedTagId === d.u32(DECODED_TAG_ID_DICT_MISS)) {
      const amber = d.vec3f(0.92, 0.62, 0.18)
      return d.vec4f(mul(amber, d.vec3f(0.5, 0.5, 0.5)), 0.32 + 0.68 * grid)
    }

    if (failureCode === d.u32(0) && decodedTagId !== d.u32(DECODED_TAG_ID_UNKNOWN)) {
      const rgb = stableHashToRgb01(decodedTagId)
      const fill = mul(rgb, d.vec3f(0.55, 0.55, 0.55))
      return d.vec4f(fill, 0.28 + 0.72 * grid)
    }

    if (failureCode === d.u32(0)) {
      return d.vec4f(0, 0, 0, grid)
    }

    const tint = gridVizFailureTintRgb(failureCode)
    return d.vec4f(mul(tint, d.vec3f(0.32 + 0.68 * grid)), a)
  })

  return root.createRenderPipeline({
    vertex: gridVizVert,
    fragment: gridVizFrag,
    targets: {
      format: presentationFormat,
      blend: {
        color: {
          operation: 'add',
          srcFactor: 'src-alpha',
          dstFactor: 'one-minus-src-alpha',
        },
        alpha: {
          operation: 'add',
          srcFactor: 'one',
          dstFactor: 'one-minus-src-alpha',
        },
      },
    },
    primitive: { topology: 'triangle-strip' },
  })
}

/** Allocates quad + uniform storage; render pipeline for AprilTag overlay. */
export function createGridVizStage(
  root: TgpuRoot,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const quadCornersBuffer = root.createBuffer(GridDataSchema).$usage('storage')
  const { gridVizLayout } = createGridVizLayouts()
  const gridVizDebugModeBuffer = root.createBuffer(d.u32).$usage('uniform')
  gridVizDebugModeBuffer.write(0)
  const gridVizPipeline = createGridVizPipeline(root, gridVizLayout, width, height, presentationFormat)
  const gridVizBindGroup = root.createBindGroup(gridVizLayout, {
    quads: quadCornersBuffer,
    failInterrogate: gridVizDebugModeBuffer,
  })
  const encodeToCanvas = (
    enc: GPUCommandEncoder,
    colorAttachment: ColorAttachment,
    instanceCount: number = MAX_INSTANCES,
  ) => {
    if (instanceCount <= 0) {
      return
    }
    gridVizPipeline
      .with(enc)
      .withColorAttachment(colorAttachment)
      .with(gridVizBindGroup)
      .draw(4, instanceCount)
  }
  return {
    quadCornersBuffer,
    gridVizLayout,
    gridVizDebugModeBuffer,
    encodeToCanvas,
  }
}

export type GridVizQuadBuffer = ReturnType<typeof createGridVizStage>['quadCornersBuffer']
