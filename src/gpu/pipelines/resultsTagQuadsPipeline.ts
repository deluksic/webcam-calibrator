import type { ExtractBindGroupInputFromLayout, TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import { arrayOf, u16 } from 'typegpu/data'
import { floor, mul, select } from 'typegpu/std'

import {
  resultsCameraBindLayout,
  type ResultsCameraBindGroup,
} from '@/gpu/pipelines/resultsCameraTransform'
import { RESULTS_MSAA_SAMPLE_COUNT } from '@/gpu/pipelines/resultsMsaa'
import { cornersToVec3fArray } from '@/gpu/pipelines/resultsSceneCpu'
import { tagIdPattern } from '@/lib/tag36h11'
import type { CalibrationOk } from '@/workers/calibration.worker'

export const MAX_RESULTS_TAG_QUADS = 2048 as const

/**
 * One calibrated tag quad on the GPU: four board-space corners (TL/TR/BL/BR) and the row-major
 * 6×6 interior bit pattern packed into 36 bits across two `u32`s (`.x` = bits 0..31, `.y` = 32..35).
 */
export const TagQuad = d.struct({
  corners: d.arrayOf(d.vec3f, 4),
  packedPattern: d.vec2u,
})

export type TagQuadRow = d.Infer<typeof TagQuad>

const TagQuadsSchema = d.arrayOf(TagQuad, MAX_RESULTS_TAG_QUADS)

export const tagQuadsBindLayout = tgpu
  .bindGroupLayout({
    tags: {
      storage: TagQuadsSchema,
      access: 'readonly',
    },
  })
  .$idx(1)

export type TagQuadsGpuBindValues = ExtractBindGroupInputFromLayout<typeof tagQuadsBindLayout.entries>

function allocTagQuads(root: TgpuRoot) {
  return root.createBuffer(TagQuadsSchema).$usage('storage')
}

export type TagQuadsResultsBuffer = ReturnType<typeof allocTagQuads>

/** Pack the row-major 6×6 0/1 interior pattern (36 bits) into `vec2u` (lo: bits 0..31, hi: 32..35). */
function packTagPattern(tagId: number): d.v2u {
  const pattern = tagIdPattern(tagId)
  let lo = 0
  let hi = 0
  for (let i = 0; i < 36; i++) {
    if (pattern[i] !== 1) {
      continue
    }
    if (i < 32) {
      lo = (lo | (1 << i)) >>> 0
    } else {
      hi = (hi | (1 << (i - 32))) >>> 0
    }
  }
  return d.vec2u(lo, hi)
}

/** Instance count for `encodeScene` / indexed draw (valid tags only; cap matches buffer prefix). */
export function tagQuadDrawCountForGpu(ok: CalibrationOk): number {
  let n = 0
  for (const t of ok.updatedTargets) {
    if (!t?.corners) {
      continue
    }
    n++
    if (n >= MAX_RESULTS_TAG_QUADS) {
      return MAX_RESULTS_TAG_QUADS
    }
  }
  return n
}

/**
 * Full storage buffer upload: one row per calibrated tag, then padded to `MAX_RESULTS_TAG_QUADS`
 * with degenerate off-board quads. TypeGPU’s buffer writer indexes every array element; a short
 * array left `undefined` slots and triggered the fallback writer (`reading 'corners'`).
 */
export function tagQuadWritesForGpu(ok: CalibrationOk): TagQuadRow[] {
  const rows: TagQuadRow[] = []
  for (const t of ok.updatedTargets) {
    if (!t?.corners) {
      continue
    }
    if (rows.length >= MAX_RESULTS_TAG_QUADS) {
      break
    }
    rows.push({
      corners: cornersToVec3fArray(t.corners),
      packedPattern: packTagPattern(t.tagId),
    })
  }
  const deadCorner = d.vec3f(0, 0, -1e9)
  const dead = TagQuad({
    corners: [deadCorner, deadCorner, deadCorner, deadCorner],
    packedPattern: d.vec2u(0, 0),
  })
  for (let i = rows.length; i < MAX_RESULTS_TAG_QUADS; i++) {
    rows.push(dead)
  }
  return rows
}

/** Two triangles for a TL/TR/BL/BR quad: (TL, TR, BL), (BL, TR, BR). */
const tagQuadIndexU16 = new Uint16Array([0, 1, 2, 2, 1, 3])

/** Pipeline + pass bind group + encode in one closure (see {@link createGridVizStage}). */
export function createTagQuadsResultsStage(root: TgpuRoot, presentationFormat: GPUTextureFormat) {
  const tagQuadsBuf = allocTagQuads(root)
  const tagQuadsBg = root.createBindGroup(tagQuadsBindLayout, { tags: tagQuadsBuf })

  const vert = tgpu
    .vertexFn({
      in: {
        vertexIndex: d.builtin.vertexIndex,
        instanceIndex: d.builtin.instanceIndex,
      },
      out: {
        clipPos: d.builtin.position,
        uv: d.vec2f,
        instanceIndexFlat: d.interpolate('flat', d.u32),
      },
    })(({ vertexIndex, instanceIndex }) => {
      'use gpu'
      const cam = resultsCameraBindLayout.$.transform
      const tag = tagQuadsBindLayout.$.tags[instanceIndex]!
      const p = tag.corners[vertexIndex]!
      const uvU = d.f32(vertexIndex & d.u32(1))
      const uvV = d.f32(vertexIndex >> d.u32(1))
      return {
        clipPos: mul(cam.clipHomogeneousMatrixFromBoard, d.vec4f(p.x, -p.y, p.z, d.f32(1))),
        uv: d.vec2f(uvU, uvV),
        instanceIndexFlat: instanceIndex,
      }
    })
    .$uses({ camera: resultsCameraBindLayout, tags: tagQuadsBindLayout })

  const frag = tgpu
    .fragmentFn({
      in: {
        uv: d.vec2f,
        instanceIndexFlat: d.interpolate('flat', d.u32),
        frontFacing: d.builtin.frontFacing,
      },
      out: d.vec4f,
    })(({ uv, instanceIndexFlat, frontFacing }) => {
      'use gpu'
      if (frontFacing) {
        return d.vec4f(0, 0, 0, 1)
      }
      const cell = d.vec2i(floor(uv * d.f32(8)))
      const onBorder = cell.x <= d.i32(0) || cell.x >= d.i32(7) || cell.y <= d.i32(0) || cell.y >= d.i32(7)
      if (onBorder) {
        return d.vec4f(0, 0, 0, 1)
      }
      const row = d.u32(cell.y - d.i32(1))
      const col = d.u32(cell.x - d.i32(1))
      const bitIndex = row * d.u32(6) + col
      const tag = tagQuadsBindLayout.$.tags[instanceIndexFlat]!
      const word = select(tag.packedPattern.x, tag.packedPattern.y, bitIndex >= d.u32(32))
      const shift = select(bitIndex, bitIndex - d.u32(32), bitIndex >= d.u32(32))
      const bit = (word >> shift) & d.u32(1)
      const v = d.f32(bit)
      return d.vec4f(v, v, v, 1)
    })
    .$uses({ camera: resultsCameraBindLayout, tags: tagQuadsBindLayout })

  const pipeline = root
    .createRenderPipeline({
      vertex: vert,
      fragment: frag,
      targets: {
        format: presentationFormat,
      },
      primitive: {
        topology: 'triangle-list',
        cullMode: 'none',
      },
      depthStencil: {
        format: 'depth24plus',
        depthWriteEnabled: true,
        depthCompare: 'less',
      },
      multisample: { count: RESULTS_MSAA_SAMPLE_COUNT },
    })
    .withIndexBuffer(root.createBuffer(arrayOf(u16, tagQuadIndexU16.length), tagQuadIndexU16).$usage('index'))

  const indexCount = tagQuadIndexU16.length
  const encodeToPass = (pass: GPURenderPassEncoder, cameraBg: ResultsCameraBindGroup, tagCount: number) => {
    if (tagCount <= 0) {
      return
    }
    pipeline.with(pass).with(cameraBg).with(tagQuadsBg).drawIndexed(indexCount, tagCount)
  }
  return { tagQuadsBuf, encodeToPass }
}
