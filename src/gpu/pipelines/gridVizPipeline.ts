// Grid visualization pipeline: instanced quad rendering via homography warping
import type { ColorAttachment, TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { abs, floor, fract, min, max, dpdx, dpdy, mul } from 'typegpu/std'

import { stableHashToRgb01 } from '@/lib/hashStableColor'

export const GRID_DIVISIONS = 8
export const GRID_LINE_WIDTH = 0.06
export const MAX_INSTANCES = 1024

/** App/UI cap for detected quads per frame; same as instance buffer length (`MAX_INSTANCES`). */
export const MAX_DETECTED_TAGS = MAX_INSTANCES

const QuadDebug = d.struct({
  failureCode: d.u32,
  edgePixelCount: d.f32,
  minR2: d.f32,
  intersectionCount: d.f32,
})

const QuadData = d.struct({
  homography: d.mat3x3f,
  debug: QuadDebug,
  /** `0xFFFFFFFF` = unknown — solid black (no hash). Same convention as CPU. */
  decodedTagId: d.u32,
})

export type QuadData = d.Infer<typeof QuadData>

/** Sentinel: no decoded id (GPU draws black, no hash). */
export const DECODED_TAG_ID_UNKNOWN = 0xffff_ffff

export const GridDataSchema = d.arrayOf(QuadData, MAX_INSTANCES)

/** 0 = legacy RGB fail tint; 1 = interrogate FAIL_INSUFFICIENT_EDGES (red hit / black miss); 2 = interrogate FAIL_LINE_FIT_FAILED (blue). */
export type GridVizFailInterrogateMode = 0 | 1 | 2

export function createGridVizLayouts() {
  const gridVizLayout = tgpu.bindGroupLayout({
    quads: { storage: GridDataSchema, access: 'readonly' },
    failInterrogate: { uniform: d.u32 },
  })
  return { gridVizLayout }
}

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
    const imgPos = mul(H, d.vec3f(uv, 1))
    const imgX = imgPos.x
    const imgY = imgPos.y
    const w = imgPos.z

    // imgPos is homogeneous (x', y', w'); Cartesian image coords are x'/w', y'/w'.
    // Emit clip so that clip.xy / w = NDC with origin at image center, y flipped.
    const clipX = (2 * imgX) / width - w
    const clipY = w - (2 * imgY) / height

    return {
      outPos: d.vec4f(clipX, clipY, 0, w),
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
      outPos: d.builtin.position,
      failureCode: d.interpolate('flat', d.u32),
      decodedTagId: d.interpolate('flat', d.u32),
    },
    out: d.vec4f,
  })(({ uv, failureCode, decodedTagId }) => {
    const ddx = dpdx(uv)
    const ddy = dpdy(uv)
    const grid = gridTextureGradBox(uv, ddx, ddy, GRID_DIVISIONS)
    const a = 0.2 + 0.75 * grid

    if (failureCode === d.u32(0) && decodedTagId !== d.u32(0xffffffff)) {
      const rgb = stableHashToRgb01(decodedTagId)
      const fill = mul(rgb, d.vec3f(0.55, 0.55, 0.55))
      return d.vec4f(fill, 0.28 + 0.72 * grid)
    }

    if (failureCode === d.u32(0)) {
      return d.vec4f(0, 0, 0, grid)
    }

    const mask = d.u32(8)
    if ((failureCode & mask) !== d.u32(0)) {
      if ((failureCode ^ mask) === d.u32(0)) {
        return d.vec4f(d.f32(1), d.f32(0), d.f32(0), a)
      }
      return d.vec4f(0, 0.25, 1, a)
    }

    return d.vec4f(0, 0, 0, a)
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
  const encodeToCanvas = (enc: GPUCommandEncoder, colorAttachment: ColorAttachment) => {
    gridVizPipeline.with(enc).withColorAttachment(colorAttachment).with(gridVizBindGroup).draw(4, MAX_INSTANCES)
  }
  return {
    quadCornersBuffer,
    gridVizLayout,
    gridVizDebugModeBuffer,
    encodeToCanvas,
  }
}
