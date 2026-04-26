// Reprojection error overlay: instanced circles + instanced line-list, one shared storage buffer.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { clamp, fwidth, length, max, mul, select, sub } from 'typegpu/std'

import { MAX_INSTANCES } from '@/gpu/pipelines/gridVizPipeline'

const ReprojPairStruct = d.struct({
  original: d.vec2f,
  /** Reprojected point in image space (`target` is reserved in WGSL). */
  reprojected: d.vec2f,
})

export type ReprojPairGpu = d.Infer<typeof ReprojPairStruct>

export const ReprojOverlaySchema = d.arrayOf(ReprojPairStruct, MAX_INSTANCES)

/** Pixel half-extent of instanced quads (CPU numbers inlined into shaders). */
const ORIGINAL_QUAD_HALF_PX = 8
const ORIGINAL_RING_OUTER_PX = 7
const ORIGINAL_RING_INNER_PX = 5

const TARGET_QUAD_HALF_PX = 5
const TARGET_FILL_RADIUS_PX = 3

const ERR_GOOD_PX = 0.75

const premultipliedAlphaBlend = {
  color: {
    operation: 'add' as const,
    srcFactor: 'src-alpha' as const,
    dstFactor: 'one-minus-src-alpha' as const,
  },
  alpha: {
    operation: 'add' as const,
    srcFactor: 'one' as const,
    dstFactor: 'one-minus-src-alpha' as const,
  },
}

export function createReprojectionOverlayLayouts() {
  const reprojOverlayLayout = tgpu.bindGroupLayout({
    pairs: { storage: ReprojOverlaySchema, access: 'readonly' },
  })
  return { reprojOverlayLayout }
}

export function createReprojectionOverlayOriginalPipeline(
  root: TgpuRoot,
  reprojOverlayLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const vert = tgpu.vertexFn({
    in: {
      vertexIndex: d.builtin.vertexIndex,
      instanceIndex: d.builtin.instanceIndex,
    },
    out: {
      outPos: d.builtin.position,
      centerPx: d.vec2f,
    },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu'
    const pair = reprojOverlayLayout.$.pairs[instanceIndex]!
    const c = pair.original
    const uvs = [d.vec2f(0, 0), d.vec2f(1, 0), d.vec2f(0, 1), d.vec2f(1, 1)]
    const uv = uvs[vertexIndex]!
    const local = mul(sub(uv, d.vec2f(0.5, 0.5)), d.f32(2 * ORIGINAL_QUAD_HALF_PX))
    const px = c + local
    const clipX = (d.f32(2) * px.x) / d.f32(width) - d.f32(1)
    const clipY = d.f32(1) - (d.f32(2) * px.y) / d.f32(height)
    return {
      outPos: d.vec4f(clipX, clipY, d.f32(0), d.f32(1)),
      centerPx: c,
    }
  })

  const frag = tgpu.fragmentFn({
    in: {
      centerPx: d.vec2f,
      fragPos: d.builtin.position,
    },
    out: d.vec4f,
  })(({ centerPx, fragPos }) => {
    'use gpu'
    const pxy = fragPos.xy
    const dist = length(sub(pxy, centerPx))
    const fOuter = dist - d.f32(ORIGINAL_RING_OUTER_PX)
    const wOuter = max(fwidth(dist), d.f32(1e-3))
    const outer = clamp(d.f32(0.5) - fOuter / (d.f32(2) * wOuter), d.f32(0), d.f32(1))
    const fInner = dist - d.f32(ORIGINAL_RING_INNER_PX)
    const inner = clamp(d.f32(0.5) - fInner / (d.f32(2) * wOuter), d.f32(0), d.f32(1))
    const ring = outer * (d.f32(1) - inner)
    const rgb = d.vec3f(0, 0.78, 1)
    const a = d.f32(0.85)
    return d.vec4f(mul(rgb, ring * a), ring * a)
  })

  return root.createRenderPipeline({
    vertex: vert,
    fragment: frag,
    targets: { format: presentationFormat, blend: premultipliedAlphaBlend },
    primitive: { topology: 'triangle-strip' },
  })
}

export function createReprojectionOverlayTargetPipeline(
  root: TgpuRoot,
  reprojOverlayLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const vert = tgpu.vertexFn({
    in: {
      vertexIndex: d.builtin.vertexIndex,
      instanceIndex: d.builtin.instanceIndex,
    },
    out: {
      outPos: d.builtin.position,
      centerPx: d.vec2f,
      errorPx: d.f32,
    },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu'
    const pair = reprojOverlayLayout.$.pairs[instanceIndex]!
    const c = pair.reprojected
    const err = length(sub(pair.reprojected, pair.original))
    const uvs = [d.vec2f(0, 0), d.vec2f(1, 0), d.vec2f(0, 1), d.vec2f(1, 1)]
    const uv = uvs[vertexIndex]!
    const local = mul(sub(uv, d.vec2f(0.5, 0.5)), d.f32(2 * TARGET_QUAD_HALF_PX))
    const px = c + local
    const clipX = (d.f32(2) * px.x) / d.f32(width) - d.f32(1)
    const clipY = d.f32(1) - (d.f32(2) * px.y) / d.f32(height)
    return {
      outPos: d.vec4f(clipX, clipY, d.f32(0), d.f32(1)),
      centerPx: c,
      errorPx: err,
    }
  })

  const frag = tgpu.fragmentFn({
    in: {
      centerPx: d.vec2f,
      errorPx: d.f32,
      fragPos: d.builtin.position,
    },
    out: d.vec4f,
  })(({ centerPx, errorPx, fragPos }) => {
    'use gpu'
    const pxy = fragPos.xy
    const dist = length(sub(pxy, centerPx))
    const f = dist - d.f32(TARGET_FILL_RADIUS_PX)
    const w = max(fwidth(dist), d.f32(1e-3))
    const fill = clamp(d.f32(0.5) - f / (d.f32(2) * w), d.f32(0), d.f32(1))
    const good = errorPx < d.f32(ERR_GOOD_PX)
    const blue = d.vec3f(0.1, 0.35, 1)
    const red = d.vec3f(0.95, 0.15, 0.12)
    const rgb = select(red, blue, good)
    const a = d.f32(0.92)
    return d.vec4f(mul(rgb, fill * a), fill * a)
  })

  return root.createRenderPipeline({
    vertex: vert,
    fragment: frag,
    targets: { format: presentationFormat, blend: premultipliedAlphaBlend },
    primitive: { topology: 'triangle-strip' },
  })
}

export function createReprojectionOverlayLinesPipeline(
  root: TgpuRoot,
  reprojOverlayLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const vert = tgpu.vertexFn({
    in: {
      vertexIndex: d.builtin.vertexIndex,
      instanceIndex: d.builtin.instanceIndex,
    },
    out: {
      outPos: d.builtin.position,
    },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu'
    const pair = reprojOverlayLayout.$.pairs[instanceIndex]!
    const dist = length(sub(pair.reprojected, pair.original))
    const longEnough = dist > d.f32(2)
    const p0 = pair.original
    const p1 = select(p0, pair.reprojected, longEnough)
    const atEnd = vertexIndex === d.u32(1)
    const p = select(p0, p1, atEnd)
    const clipX = (d.f32(2) * p.x) / d.f32(width) - d.f32(1)
    const clipY = d.f32(1) - (d.f32(2) * p.y) / d.f32(height)
    return {
      outPos: d.vec4f(clipX, clipY, d.f32(0), d.f32(1)),
    }
  })

  const frag = tgpu.fragmentFn({
    in: { fragPos: d.builtin.position },
    out: d.vec4f,
  })(({ fragPos }) => {
    'use gpu'
    void fragPos
    const rgb = d.vec3f(1, 0.45, 0.15)
    const a = d.f32(0.88)
    return d.vec4f(mul(rgb, a), a)
  })

  return root.createRenderPipeline({
    vertex: vert,
    fragment: frag,
    targets: { format: presentationFormat, blend: premultipliedAlphaBlend },
    primitive: { topology: 'line-list' },
  })
}

/** Allocates pair buffer + three draw pipelines for reprojection overlay. */
export function createReprojectionOverlayStage(
  root: TgpuRoot,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const reprojOverlayBuffer = root.createBuffer(ReprojOverlaySchema).$usage('storage')
  const { reprojOverlayLayout } = createReprojectionOverlayLayouts()
  const reprojOverlayBindGroup = root.createBindGroup(reprojOverlayLayout, {
    pairs: reprojOverlayBuffer,
  })
  const reprojOriginalPipeline = createReprojectionOverlayOriginalPipeline(
    root,
    reprojOverlayLayout,
    width,
    height,
    presentationFormat,
  )
  const reprojTargetPipeline = createReprojectionOverlayTargetPipeline(
    root,
    reprojOverlayLayout,
    width,
    height,
    presentationFormat,
  )
  const reprojLinesPipeline = createReprojectionOverlayLinesPipeline(
    root,
    reprojOverlayLayout,
    width,
    height,
    presentationFormat,
  )
  const reprojOverlayDrawState = { instanceCount: 0 }
  return {
    reprojOverlayBuffer,
    reprojOverlayLayout,
    reprojOverlayBindGroup,
    reprojOriginalPipeline,
    reprojTargetPipeline,
    reprojLinesPipeline,
    reprojOverlayDrawState,
  }
}
