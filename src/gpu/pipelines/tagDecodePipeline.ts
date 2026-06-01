// GPU tag36h11: per-quad pixel histogram → linear peaks → deadband votes → pattern -1/-2 → parallel dict → canonicalize.
import type { ColorAttachment } from 'typegpu'
import type { TgpuRoot } from 'typegpu'
import { d, tgpu, std, common } from 'typegpu'
import { floor, max, min, mul, round, sqrt } from 'typegpu/std'
import { abs, atomicAdd, atomicMin, clamp, countOneBits, textureLoad } from 'typegpu/std'

import { profileComputePass, profileRenderPass } from '@/gpu/gpuProfiling'
import { MAX_QUADS } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import {
  GridDataSchema,
  type GridVizQuadBuffer,
} from '@/gpu/pipelines/gridVizPipeline'
import { tryHomographyFromCorners } from '@/gpu/shaders/homographyDlt'
import { Corners4, rotateStripCorners } from '@/gpu/shaders/quadCornerOrder'
import {
  DECODE_MIN_VOTE_FRACTION_OF_QUAD_EDGE,
  TAG_DECODE_MAX_DICT_ERROR,
  TAG_DECODE_MAX_WEAK_WILDCARD,
} from '@/gpu/tagDecodeThresholds'
import { TAG36H11_CODES, TAG36H11_COUNT } from '@/lib/tag36h11'

const TAG_MODULES = 8
const DATA_MODULES = 6
const MODULES_PER_QUAD = DATA_MODULES * DATA_MODULES
const COMPUTE_WG = 64
const CLEAR_WG = 256

const BIT_X = [
  1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2, 5, 4, 3, 4, 1, 1, 1, 1, 1, 2, 2, 2, 3,
] as const

const BIT_Y = [
  1, 1, 1, 1, 1, 2, 2, 2, 3, 1, 2, 3, 4, 5, 2, 3, 4, 3, 6, 6, 6, 6, 6, 5, 5, 5, 4, 6, 5, 4, 3, 2, 5, 4, 3, 4,
] as const

/** `moduleIdx` (row-major 6×6) → codeword bit index (inverse of BIT_X/BIT_Y). */
const MODULE_TO_BIT = (() => {
  const lut: number[] = Array.from<number>({ length: MODULES_PER_QUAD }).fill(0)
  for (let bit = 0; bit < 36; bit++) {
    const col = BIT_X[bit]! - 1
    const row = BIT_Y[bit]! - 1
    lut[row * DATA_MODULES + col] = bit
  }
  return lut
})()

const ROT_LUTS_0 = [
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
  32, 33, 34, 35,
] as const
const ROT_LUTS_1 = [
  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 3,
  4, 5, 6, 7, 8,
] as const
const ROT_LUTS_2 = [
  18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
  14, 15, 16, 17,
] as const
const ROT_LUTS_3 = [
  27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
  23, 24, 25, 26,
] as const

const ModuleToBitGpu = tgpu.const(d.arrayOf(d.u32, MODULES_PER_QUAD), MODULE_TO_BIT)
const BitXGpu = tgpu.const(d.arrayOf(d.u32, MODULES_PER_QUAD), [...BIT_X])
const BitYGpu = tgpu.const(d.arrayOf(d.u32, MODULES_PER_QUAD), [...BIT_Y])
const RotLut0Gpu = tgpu.const(d.arrayOf(d.u32, MODULES_PER_QUAD), [...ROT_LUTS_0])
const RotLut1Gpu = tgpu.const(d.arrayOf(d.u32, MODULES_PER_QUAD), [...ROT_LUTS_1])
const RotLut2Gpu = tgpu.const(d.arrayOf(d.u32, MODULES_PER_QUAD), [...ROT_LUTS_2])
const RotLut3Gpu = tgpu.const(d.arrayOf(d.u32, MODULES_PER_QUAD), [...ROT_LUTS_3])

const rotateBit = tgpu.fn([d.u32, d.u32], d.u32)((r, bitU) => {
  'use gpu'
  const v0 = RotLut0Gpu.$[bitU]!
  const v1 = RotLut1Gpu.$[bitU]!
  const v2 = RotLut2Gpu.$[bitU]!
  const v3 = RotLut3Gpu.$[bitU]!
  const lo = r === d.u32(0) ? v0 : v1
  const hi = r === d.u32(2) ? v2 : v3
  return r < d.u32(2) ? lo : hi
})

const PATTERN_BLACK = 0
const PATTERN_WHITE = 1
const PATTERN_WEAK = -1
const PATTERN_TIE = -2

const WORST_SCORE = ((TAG_DECODE_MAX_DICT_ERROR + 1) << 21) | (1 << 20) | TAG36H11_COUNT
const WorstScoreGpu = tgpu.const(d.u32, WORST_SCORE)

const CodewordPair = d.struct({ low: d.u32, high: d.u32 })
const CodewordBuffer = d.arrayOf(CodewordPair, TAG36H11_COUNT)

const MAX_CUSTOM_COUNT = 4096
const CustomCodewordBuffer = d.arrayOf(CodewordPair, MAX_CUSTOM_COUNT)
const CustomCountSchema = d.arrayOf(d.u32, 1)
const FlatI32Schema = d.arrayOf(d.i32, MAX_QUADS)

export const TagDecodeThresholdGpu = d.struct({
  blackBound: d.f32,
  whiteBound: d.f32,
  minVoteTotal: d.u32,
  valid: d.u32,
})

export const TagDecodeThresholdSchema = d.arrayOf(TagDecodeThresholdGpu, MAX_QUADS)

const QuadDecodeMetaGpu = d.struct({
  /** 0=ties, 1=weaks-no-ties, 2=clean (no ties, no weaks). */
  patternQuality: d.u32,
  weakCount: d.u32,
  weakBit0: d.u32,
  weakBit1: d.u32,
  weakBit2: d.u32,
  weakBit3: d.u32,
  canonRot: d.u32,
})

const QuadDecodeMetaSchema = d.arrayOf(QuadDecodeMetaGpu, MAX_QUADS)
export const PatternGrid = d.struct({ modules: d.arrayOf(d.i32, MODULES_PER_QUAD) })
const PatternSchema = d.arrayOf(PatternGrid, MAX_QUADS)
const ModuleVoteGrid = d.struct({ votes: d.arrayOf(d.atomic(d.u32), MODULES_PER_QUAD) })
const ModuleVoteSchema = d.arrayOf(ModuleVoteGrid, MAX_QUADS)
export const ModuleVoteReadonlyGrid = d.struct({ votes: d.arrayOf(d.u32, MODULES_PER_QUAD) })
const ModuleVoteReadonlySchema = d.arrayOf(ModuleVoteReadonlyGrid, MAX_QUADS)

const AtomicBestSchema = d.arrayOf(d.atomic(d.u32), MAX_QUADS)
const AtomicBestReadonlySchema = d.arrayOf(d.u32, MAX_QUADS)

const ActiveQuadCountSchema = d.arrayOf(d.u32, 1)

function capQuadCount(quadCount: number): number {
  return Math.max(0, Math.min(quadCount, MAX_QUADS))
}

function quadComputeWgs(quadCount: number): number {
  const n = capQuadCount(quadCount)
  return n > 0 ? Math.ceil(n / COMPUTE_WG) : 0
}

function allocCodewordBuffer(root: TgpuRoot) {
  const buf = root.createBuffer(CodewordBuffer).$usage('storage')
  const data: { low: number; high: number }[] = []
  for (const code of TAG36H11_CODES) {
    data.push({ low: Number(code & 0xffffffffn), high: Number((code >> 32n) & 0xffffffffn) })
  }
  buf.write(data)
  return buf
}

function createModuleVoteClear(
  root: TgpuRoot,
  deps: {
    moduleWhiteBuf: ReturnType<typeof root.createBuffer<typeof ModuleVoteSchema>>
    moduleBlackBuf: ReturnType<typeof root.createBuffer<typeof ModuleVoteSchema>>
  },
) {
  const voteClearLayout = tgpu.bindGroupLayout({
    moduleWhite: { storage: ModuleVoteReadonlySchema, access: 'mutable' },
    moduleBlack: { storage: ModuleVoteReadonlySchema, access: 'mutable' },
  })
  const voteClearKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [CLEAR_WG, 1, 1],
  })((input) => {
    const idx = d.u32(input.gid.x)
    const total = d.u32(MAX_QUADS * MODULES_PER_QUAD)
    if (idx >= total) {
      return
    }
    const quadId = idx / d.u32(MODULES_PER_QUAD)
    const vIdx = idx % d.u32(MODULES_PER_QUAD)
    voteClearLayout.$.moduleWhite[quadId]!.votes[vIdx] = d.u32(0)
    voteClearLayout.$.moduleBlack[quadId]!.votes[vIdx] = d.u32(0)
  })
  const voteClearPipeline = root.createComputePipeline({ compute: voteClearKernel })
  const voteClearBindGroup = root.createBindGroup(voteClearLayout, {
    moduleWhite: deps.moduleWhiteBuf as never,
    moduleBlack: deps.moduleBlackBuf as never,
  })
  const voteClearWgs = Math.ceil((MAX_QUADS * MODULES_PER_QUAD) / CLEAR_WG)

  return {
    encodeClearModuleVotes(pass: GPUComputePassEncoder) {
      voteClearPipeline.with(pass).with(voteClearBindGroup).dispatchWorkgroups(voteClearWgs)
    },
  }
}

function createAtomicBestClearStage(
  root: TgpuRoot,
  atomicBestBuf: ReturnType<typeof root.createBuffer<typeof AtomicBestSchema>>,
) {
  const layout = tgpu.bindGroupLayout({
    atomicBest: { storage: AtomicBestReadonlySchema, access: 'mutable' },
  })
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [CLEAR_WG, 1, 1],
  })((input) => {
    const idx = d.u32(input.gid.x)
    if (idx >= d.u32(MAX_QUADS)) {
      return
    }
    layout.$.atomicBest[idx] = WorstScoreGpu.$
  })
  const pipeline = root.createComputePipeline({ compute: kernel })
  const bindGroup = root.createBindGroup(layout, { atomicBest: atomicBestBuf as never })
  return {
    encodeClear(pass: GPUComputePassEncoder) {
      pipeline.with(pass).with(bindGroup).dispatchWorkgroups(Math.ceil(MAX_QUADS / CLEAR_WG))
    },
  }
}


function createModuleVoteStage(
  root: TgpuRoot,
  grayTexView: unknown,
  quadDataBuffer: GridVizQuadBuffer,
  thresholdBuf: ReturnType<typeof root.createBuffer<typeof TagDecodeThresholdSchema>>,
  moduleWhiteBuf: ReturnType<typeof root.createBuffer<typeof ModuleVoteSchema>>,
  moduleBlackBuf: ReturnType<typeof root.createBuffer<typeof ModuleVoteSchema>>,
  width: number,
  height: number,
) {
  const debugTex = root.device.createTexture({
    label: 'tag-vote-debug',
    size: [ width, height, 1 ],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
  })

  const layout = tgpu.bindGroupLayout({
    quads: { storage: GridDataSchema, access: 'readonly' },
    grayTex: { texture: d.texture2d() },
    thresholds: { storage: TagDecodeThresholdSchema, access: 'readonly' },
    moduleWhite: { storage: ModuleVoteSchema, access: 'mutable' },
    moduleBlack: { storage: ModuleVoteSchema, access: 'mutable' },
  })

  const vert = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: { outPos: d.builtin.position, uv: d.vec2f, quadId: d.interpolate('flat', d.u32) },
  })(({ vertexIndex, instanceIndex }) => {
    const quad = layout.$.quads[instanceIndex]!
    const H = quad.homography
    const c0 = H.columns[0]!
    const c1 = H.columns[1]!
    const hLen = c0.x * c0.x + c0.y * c0.y + c1.x * c1.x + c1.y * c1.y
    if (hLen < d.f32(1e-6)) {
      return { outPos: d.vec4f(-2, -2, 0, 1), uv: d.vec2f(0), quadId: instanceIndex }
    }
    const uvs = [d.vec2f(0, 0), d.vec2f(1, 0), d.vec2f(0, 1), d.vec2f(1, 1)]
    const uv = uvs[vertexIndex]!
    const imgPos = mul(H, d.vec3f(uv, 1))
    const w = imgPos.z
    const clipX = (2 * imgPos.x) / d.f32(width) - w
    const clipY = w - (2 * imgPos.y) / d.f32(height)
    return { outPos: d.vec4f(clipX, clipY, 0, w), uv, quadId: instanceIndex }
  })

  const frag = tgpu.fragmentFn({
    in: { pos: d.builtin.position, uv: d.vec2f, quadId: d.interpolate('flat', d.u32) },
    out: d.vec4f,
  })(({ pos, uv, quadId }) => {
    const thr = layout.$.thresholds[quadId]!
    if (thr.valid === d.u32(0)) {
      return d.vec4f(0, 0, 0, 0)
    }
    const mx = d.u32(floor(uv.x * d.f32(TAG_MODULES)))
    const my = d.u32(floor(uv.y * d.f32(TAG_MODULES)))
    const inInterior = mx >= d.u32(1) && mx <= d.u32(6) && my >= d.u32(1) && my <= d.u32(6)
    const gray = textureLoad(layout.$.grayTex, d.vec2u(d.u32(pos.x), d.u32(pos.y)), d.i32(0)).x
    if (gray <= thr.blackBound) {
      if (inInterior) {
        const cellIdx = (my - d.u32(1)) * d.u32(DATA_MODULES) + (mx - d.u32(1))
        atomicAdd(layout.$.moduleBlack[quadId]!.votes[cellIdx]!, d.u32(1))
      }
      return d.vec4f(0, 0, 0, 1)
    }
    if (gray >= thr.whiteBound) {
      if (inInterior) {
        const cellIdx = (my - d.u32(1)) * d.u32(DATA_MODULES) + (mx - d.u32(1))
        atomicAdd(layout.$.moduleWhite[quadId]!.votes[cellIdx]!, d.u32(1))
      }
      return d.vec4f(1, 1, 1, 1)
    }
    if (inInterior) {
      return d.vec4f(0.2, 0.3, 1, 1)
    }
    return d.vec4f(0, 0, 0, 1)
  })

  const pipeline = root
    .createRenderPipeline({
      vertex: vert,
      fragment: frag,
      targets: { format: 'rgba8unorm' },
      primitive: { topology: 'triangle-strip' },
    })
    .$name('tag-module-votes')

  const bindGroup = root.createBindGroup(layout, {
    quads: quadDataBuffer,
    grayTex: grayTexView as never,
    thresholds: thresholdBuf as never,
    moduleWhite: moduleWhiteBuf as never,
    moduleBlack: moduleBlackBuf as never,
  })

  return {
    debugTex,
    encodeModuleVotes(enc: GPUCommandEncoder, instanceCount: number) {
      if (instanceCount < 1) {
        return
      }
      const pass = profileRenderPass(enc, pipeline, {
        label: 'tag-module-votes',
        colorAttachments: [
          { view: debugTex.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0.35, 0.35, 0.35, 1] },
        ],
      })
      pass.setViewport(0, 0, width, height, 0, 1)
      pipeline.with(pass).with(bindGroup).draw(4, instanceCount)
      pass.end()
    },
  }
}

function createClassifyStage(
  root: TgpuRoot,
  moduleWhiteBuf: ReturnType<typeof root.createBuffer<typeof ModuleVoteSchema>>,
  moduleBlackBuf: ReturnType<typeof root.createBuffer<typeof ModuleVoteSchema>>,
  thresholdBuf: ReturnType<typeof root.createBuffer<typeof TagDecodeThresholdSchema>>,
  patternBuf: ReturnType<typeof root.createBuffer<typeof PatternSchema>>,
  metaBuf: ReturnType<typeof root.createBuffer<typeof QuadDecodeMetaSchema>>,
  activeQuadCountBuf: ReturnType<typeof root.createBuffer<typeof ActiveQuadCountSchema>>,
) {
  const layout = tgpu.bindGroupLayout({
    moduleWhite: { storage: ModuleVoteReadonlySchema, access: 'readonly' },
    moduleBlack: { storage: ModuleVoteReadonlySchema, access: 'readonly' },
    thresholds: { storage: TagDecodeThresholdSchema, access: 'readonly' },
    pattern: { storage: PatternSchema, access: 'mutable' },
    meta: { storage: QuadDecodeMetaSchema, access: 'mutable' },
    activeQuadCount: { storage: ActiveQuadCountSchema, access: 'readonly' },
  })

  const bindGroup = root.createBindGroup(layout, {
    moduleWhite: moduleWhiteBuf as never,
    moduleBlack: moduleBlackBuf as never,
    thresholds: thresholdBuf as never,
    pattern: patternBuf as never,
    meta: metaBuf as never,
    activeQuadCount: activeQuadCountBuf as never,
  })

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [COMPUTE_WG, 1, 1],
  })((input) => {
    const quadId = d.u32(input.gid.x)
    if (quadId >= layout.$.activeQuadCount[0]!) {
      return
    }

    const thr = layout.$.thresholds[quadId]!
    
    let tieCount = d.u32(0)
    let weakCount = d.u32(0)
    let whiteCount = d.u32(0)
    const weakBits = d.arrayOf(d.u32, TAG_DECODE_MAX_WEAK_WILDCARD)()

    // Unroll: runtime loop here × dict `mask` loop would explode work (TDR / “hang”) when tags appear.
    for (const i of tgpu.unroll(std.range(0, MODULES_PER_QUAD))) {
      const iu = d.u32(i)
      const w = layout.$.moduleWhite[quadId]!.votes[iu]!
      const b = layout.$.moduleBlack[quadId]!.votes[iu]!
      const sum = w + b
      let cell = d.i32(PATTERN_WEAK)
      if (thr.valid === d.u32(0) || sum < thr.minVoteTotal) {
        cell = d.i32(PATTERN_WEAK)
      } else {
        let diff = d.u32(abs(d.i32(b) - d.i32(w)))
        const minDiff = max(d.u32(2), sum / d.u32(6))
        if (diff >= minDiff) {
          if (b > w) { cell = d.i32(PATTERN_BLACK) }
          else { cell = d.i32(PATTERN_WHITE); whiteCount = whiteCount + d.u32(1) }
        } else {
          cell = d.i32(PATTERN_TIE)
          tieCount = tieCount + d.u32(1)
        }
      }
      layout.$.pattern[quadId]!.modules[iu] = cell
      if (cell === d.i32(PATTERN_WEAK) && weakCount < d.u32(TAG_DECODE_MAX_WEAK_WILDCARD)) {
        weakBits[weakCount] = ModuleToBitGpu.$[iu]!
        weakCount = weakCount + d.u32(1)
      }
    }

    let quality = d.u32(2) // clean by default
    if (tieCount > d.u32(0)) {
      quality = d.u32(0)
    } else if (weakCount > d.u32(0)) {
      quality = d.u32(1)
    }
    // Degenerate: all 36 cells are the same confident value → all-black or all-white.
    if (tieCount === d.u32(0) && weakCount === d.u32(0) && (whiteCount === d.u32(0) || whiteCount === d.u32(MODULES_PER_QUAD))) {
      quality = d.u32(0)
    }

    // Find canonical rotation: which rotation produces minimal 36-bit code.
    let canonRot = d.u32(0)
    let canonLow = d.u32(0xFFFFFFFF)
    let canonHigh = d.u32(0xFFFFFFFF)
    for (let r = d.u32(0); r < d.u32(4); r = r + d.u32(1)) {
      let codeLow = d.u32(0)
      let codeHigh = d.u32(0)
      for (const bit of tgpu.unroll(std.range(0, MODULES_PER_QUAD))) {
        const bitU = d.u32(bit)
        let srcBit = rotateBit(r, bitU)
        const bx = BitXGpu.$[srcBit]! - d.u32(1)
        const by = BitYGpu.$[srcBit]! - d.u32(1)
        const pIdx = by * d.u32(DATA_MODULES) + bx
        const cell = layout.$.pattern[quadId]!.modules[pIdx]!
        if (cell === d.i32(PATTERN_WHITE)) {
          if (bitU >= d.u32(32)) { codeHigh = codeHigh | (d.u32(1) << ((bitU - d.u32(32)) & d.u32(31))) }
          else { codeLow = codeLow | (d.u32(1) << (bitU & d.u32(31))) }
        }
      }
      if (codeHigh < canonHigh || (codeHigh === canonHigh && codeLow < canonLow)) {
        canonHigh = codeHigh
        canonLow = codeLow
        canonRot = r
      }
    }

    layout.$.meta[quadId] = QuadDecodeMetaGpu({
      patternQuality: quality,
      weakCount,
      weakBit0: weakBits[0]!,
      weakBit1: weakBits[1]!,
      weakBit2: weakBits[2]!,
      weakBit3: weakBits[3]!,
      canonRot,
    })
  })

  const pipeline = root.createComputePipeline({ compute: kernel }).$name('tag-classify')

  return {
    encodeClassify(pass: GPUComputePassEncoder, quadCount: number) {
      const wgs = quadComputeWgs(quadCount)
      if (wgs > 0) {
        pipeline.with(pass).with(bindGroup).dispatchWorkgroups(wgs)
      }
    },
  }
}

function createDictMatchStage(
  root: TgpuRoot,
  codewordBuffer: ReturnType<typeof root.createBuffer<typeof CodewordBuffer>>,
  customCodewordBuffer: ReturnType<typeof root.createBuffer<typeof CustomCodewordBuffer>>,
  customCountBuf: ReturnType<typeof root.createBuffer<typeof CustomCountSchema>>,
  patternBuf: ReturnType<typeof root.createBuffer<typeof PatternSchema>>,
  metaBuf: ReturnType<typeof root.createBuffer<typeof QuadDecodeMetaSchema>>,
  atomicBestBuf: ReturnType<typeof root.createBuffer<typeof AtomicBestSchema>>,
  activeQuadCountBuf: ReturnType<typeof root.createBuffer<typeof ActiveQuadCountSchema>>,
) {
  const layout = tgpu.bindGroupLayout({
    codewords: { storage: CodewordBuffer, access: 'readonly' },
    customCodewords: { storage: CustomCodewordBuffer, access: 'readonly' },
    customCount: { storage: CustomCountSchema, access: 'readonly' },
    pattern: { storage: PatternSchema, access: 'readonly' },
    meta: { storage: QuadDecodeMetaSchema, access: 'readonly' },
    atomicBest: { storage: AtomicBestSchema, access: 'mutable' },
    activeQuadCount: { storage: ActiveQuadCountSchema, access: 'readonly' },
  })

  const bindGroup = root.createBindGroup(layout, {
    codewords: codewordBuffer as never,
    customCodewords: customCodewordBuffer as never,
    customCount: customCountBuf as never,
    pattern: patternBuf as never,
    meta: metaBuf as never,
    atomicBest: atomicBestBuf as never,
    activeQuadCount: activeQuadCountBuf as never,
  })

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [COMPUTE_WG, 1, 1],
  })((input) => {
    const gid = d.u32(input.gid.x)
    const totalPerQuad = d.u32(TAG36H11_COUNT) + layout.$.customCount[0]!
    const cwIdx = gid % totalPerQuad
    const quadId = d.u32(gid / totalPerQuad)
    if (quadId >= layout.$.activeQuadCount[0]!) {
      return
    }

    const meta = layout.$.meta[quadId]!
    if (meta.patternQuality === d.u32(0)) {
      return
    }

    const isCustom = cwIdx >= d.u32(TAG36H11_COUNT)
    const customIdx = cwIdx - d.u32(TAG36H11_COUNT)
    const weakCount = meta.weakCount
    if (weakCount > d.u32(TAG_DECODE_MAX_WEAK_WILDCARD)) {
      return
    }

    // Both dictionaries are canonical (minimal) — only check canonical rotation.
    // Custom tags must match at canonRot; standard tags always match at canonRot since
    // codewords are preprocessed to canonical form.

    const maskCount = d.u32(1) << weakCount
    let localBest = d.u32(TAG_DECODE_MAX_DICT_ERROR + 1)
    let localRot = d.u32(0)

    for (let mask = d.u32(0); mask < maskCount; mask = mask + d.u32(1)) {
      let wildLow = d.u32(0)
      let wildHigh = d.u32(0)
      if (weakCount > d.u32(0) && (mask & d.u32(1)) !== d.u32(0)) {
        const bit = meta.weakBit0
        const pos = d.u32(35) - bit
        if (pos >= d.u32(32)) {
          wildHigh = wildHigh | (d.u32(1) << (pos - d.u32(32)))
        } else {
          wildLow = wildLow | (d.u32(1) << pos)
        }
      }
      if (weakCount > d.u32(1) && (mask & d.u32(2)) !== d.u32(0)) {
        const bit = meta.weakBit1
        const pos = d.u32(35) - bit
        if (pos >= d.u32(32)) {
          wildHigh = wildHigh | (d.u32(1) << (pos - d.u32(32)))
        } else {
          wildLow = wildLow | (d.u32(1) << pos)
        }
      }
      if (weakCount > d.u32(2) && (mask & d.u32(4)) !== d.u32(0)) {
        const bit = meta.weakBit2
        const pos = d.u32(35) - bit
        if (pos >= d.u32(32)) {
          wildHigh = wildHigh | (d.u32(1) << (pos - d.u32(32)))
        } else {
          wildLow = wildLow | (d.u32(1) << pos)
        }
      }
      if (weakCount > d.u32(3) && (mask & d.u32(8)) !== d.u32(0)) {
        const bit = meta.weakBit3
        const pos = d.u32(35) - bit
        if (pos >= d.u32(32)) {
          wildHigh = wildHigh | (d.u32(1) << (pos - d.u32(32)))
        } else {
          wildLow = wildLow | (d.u32(1) << pos)
        }
      }

      for (const rot of tgpu.unroll(std.range(0, 4))) {
          let knownLow = d.u32(0)
          let knownHigh = d.u32(0)
          for (const bit of tgpu.unroll(std.range(0, MODULES_PER_QUAD))) {
            const bitU = d.u32(bit)
            let srcBit = rotateBit(d.u32(rot), bitU)
            const bx = BitXGpu.$[srcBit]! - d.u32(1)
            const by = BitYGpu.$[srcBit]! - d.u32(1)
            const pIdx = by * d.u32(DATA_MODULES) + bx
            const cell = layout.$.pattern[quadId]!.modules[pIdx]!
            if (cell === d.i32(PATTERN_WHITE)) {
              const pos = d.u32(35) - bitU
              if (pos >= d.u32(32)) {
                knownHigh = knownHigh | (d.u32(1) << (pos - d.u32(32)))
              } else {
                knownLow = knownLow | (d.u32(1) << pos)
              }
            }
          }
          const customLo = layout.$.customCodewords[customIdx]!.low
          const customHi = layout.$.customCodewords[customIdx]!.high
          const stdLo = layout.$.codewords[cwIdx]!.low
          const stdHi = layout.$.codewords[cwIdx]!.high
          let cwLow = isCustom ? customLo : stdLo
          let cwHigh = isCustom ? customHi : stdHi
          const diffLow = knownLow ^ cwLow ^ wildLow
          const diffHigh = knownHigh ^ cwHigh ^ wildHigh
          const d0 = countOneBits(diffLow) + countOneBits(diffHigh)
          if (d0 < localBest) {
            localBest = d0
            localRot = d.u32(rot)
          } else if (d0 === localBest) {
            localRot = d.u32(rot)
          }
      }
    }

    if (localBest <= d.u32(TAG_DECODE_MAX_DICT_ERROR)) {
      // bit 20 = custom flag, bits 18-19 = rotation, bits 0-17 = cwIdx, bits 21+ = distance
      const customBitFlag = d.u32(1) << d.u32(20)
      let packedCwIdx = isCustom ? customIdx : cwIdx
      let customBit = isCustom ? customBitFlag : d.u32(0)
      const candidate = (localBest << d.u32(21)) | customBit | (localRot << d.u32(18)) | packedCwIdx
      atomicMin(layout.$.atomicBest[quadId]!, candidate)
    }
  })

  const pipeline = root.createComputePipeline({ compute: kernel }).$name('tag-dict')

  return {
    encodeDictMatch(pass: GPUComputePassEncoder, quadCount: number, customCount: number) {
      const n = capQuadCount(quadCount)
      const threads = n * (TAG36H11_COUNT + customCount)
      if (threads > 0) {
        pipeline
          .with(pass)
          .with(bindGroup)
          .dispatchWorkgroups(Math.ceil(threads / COMPUTE_WG))
      }
    },
  }
}

function createCanonicalizeStage(
  root: TgpuRoot,
  quadDataBuffer: GridVizQuadBuffer,
  atomicBestBuf: ReturnType<typeof root.createBuffer<typeof AtomicBestSchema>>,
  metaBuf: ReturnType<typeof root.createBuffer<typeof QuadDecodeMetaSchema>>,
  activeQuadCountBuf: ReturnType<typeof root.createBuffer<typeof ActiveQuadCountSchema>>,
) {
  const layout = tgpu.bindGroupLayout({
    quadData: { storage: GridDataSchema, access: 'mutable' },
    atomicBest: { storage: AtomicBestReadonlySchema, access: 'readonly' },
    meta: { storage: QuadDecodeMetaSchema, access: 'readonly' },
    activeQuadCount: { storage: ActiveQuadCountSchema, access: 'readonly' },
  })

  const bindGroup = root.createBindGroup(layout, {
    quadData: quadDataBuffer,
    atomicBest: atomicBestBuf as never,
    meta: metaBuf as never,
    activeQuadCount: activeQuadCountBuf as never,
  })

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [COMPUTE_WG, 1, 1],
  })((input) => {
    const quadId = d.u32(input.gid.x)
    if (quadId >= layout.$.activeQuadCount[0]!) {
      return
    }

    const quad = layout.$.quadData[quadId]!
    const H = quad.homography
    const c0 = H.columns[0]!
    const c1 = H.columns[1]!
    const hLen = c0.x * c0.x + c0.y * c0.y + c1.x * c1.x + c1.y * c1.y
    const quality = layout.$.meta[quadId]!.patternQuality

    if (hLen < d.f32(1e-6) || quality === d.u32(0)) {
      layout.$.quadData[quadId]!.tagKind = d.u32(0)
      layout.$.quadData[quadId]!.decodedTagId = d.i32(0)
      return
    }

    const score = layout.$.atomicBest[quadId]!
    const dist = score >> d.u32(21)
    const bestRot = (score >> d.u32(18)) & d.u32(3)
    const isCustom = ((score >> d.u32(20)) & d.u32(1)) !== d.u32(0)
    const bestId = score & d.u32(0x3ffff)

    if (dist > d.u32(TAG_DECODE_MAX_DICT_ERROR)) {
      if (quality === d.u32(2)) {
        layout.$.quadData[quadId]!.tagKind = d.u32(2) // clean undecoded
      } else {
        layout.$.quadData[quadId]!.tagKind = d.u32(0) // weaks, no match -> dead
      }
      layout.$.quadData[quadId]!.decodedTagId = d.i32(0)
      return
    }

    // Decoded
    layout.$.quadData[quadId]!.tagKind = d.u32(1)
    if (isCustom) {
      layout.$.quadData[quadId]!.decodedTagId = -(d.i32(bestId) + d.i32(1))
    } else {
      layout.$.quadData[quadId]!.decodedTagId = d.i32(bestId)
    }
    layout.$.quadData[quadId]!.decodedRotation = bestRot

    const strip = quad.screenCorners
    const rotated = rotateStripCorners(strip, (d.u32(4) - bestRot) & d.u32(3))
    const hRes = tryHomographyFromCorners(rotated[0]!, rotated[1]!, rotated[2]!, rotated[3]!)
    layout.$.quadData[quadId]!.screenCorners = Corners4(rotated)
    layout.$.quadData[quadId]!.homography = d.mat3x3f(
      hRes.homography.columns[0]!,
      hRes.homography.columns[1]!,
      hRes.homography.columns[2]!,
    )
    layout.$.quadData[quadId]!.decodedRotation = d.u32(0)
  })

  const pipeline = root.createComputePipeline({ compute: kernel }).$name('tag-canonicalize')

  return {
    encodeCanonicalize(pass: GPUComputePassEncoder, quadCount: number) {
      const wgs = quadComputeWgs(quadCount)
      if (wgs > 0) {
        pipeline.with(pass).with(bindGroup).dispatchWorkgroups(wgs)
      }
    },
  }
}


const voteDebugLayout = tgpu.bindGroupLayout({
  voteTex: { texture: d.texture2d() },
})

export function createVoteDebugDisplayStage(
  root: TgpuRoot,
  voteDebugTex: GPUTexture,
  presentationFormat: GPUTextureFormat,
) {
  const frag = tgpu.fragmentFn({
    in: { pos: d.builtin.position },
    out: d.vec4f,
  })((i) => {
    'use gpu'
    return textureLoad(voteDebugLayout.$.voteTex, d.vec2u(d.u32(i.pos.x), d.u32(i.pos.y)), d.i32(0))
  })

  const pipeline = root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: frag,
    targets: { format: presentationFormat },
  })

  const bindGroup = root.createBindGroup(voteDebugLayout, {
    voteTex: voteDebugTex as never,
  })

  function encodeDisplay(enc: GPUCommandEncoder, colorAttachment: ColorAttachment) {
    pipeline.with(enc).withColorAttachment(colorAttachment).with(bindGroup).draw(3)
  }

  return { encodeDisplay }
}

export type TagDecodeStage = ReturnType<typeof createTagDecodeStage>

export function createTagDecodeStage(
  root: TgpuRoot,
  deps: {
    grayTexView: unknown
    quadDataBuffer: GridVizQuadBuffer
    width: number
    height: number
    thresholdBuf?: ReturnType<typeof root.createBuffer<typeof TagDecodeThresholdSchema>>
  },
) {
  const thresholdBuf = deps.thresholdBuf ?? root.createBuffer(TagDecodeThresholdSchema).$usage('storage')
  const moduleWhiteBuf = root.createBuffer(ModuleVoteSchema).$usage('storage')
  const moduleBlackBuf = root.createBuffer(ModuleVoteSchema).$usage('storage')
  const patternBuf = root.createBuffer(PatternSchema).$usage('storage')
  const metaBuf = root.createBuffer(QuadDecodeMetaSchema).$usage('storage')
  const atomicBestBuf = root.createBuffer(AtomicBestSchema).$usage('storage')
  const activeQuadCountBuf = root.createBuffer(ActiveQuadCountSchema).$usage('storage')

  // Custom codeword buffer — always allocated, populated by updateCustomCodewords.
  const customCodewordBuffer = root.createBuffer(CustomCodewordBuffer).$usage('storage')
  const customCountBuf = root.createBuffer(CustomCountSchema).$usage('storage')
  customCountBuf.write([0])

  let currentCustomCount = 0

  function updateCustomCodewords(codes: bigint[]) {
    const data: { low: number; high: number }[] = []
    for (let i = 0; i < MAX_CUSTOM_COUNT; i++) {
      if (i < codes.length) {
        const code = codes[i]!
        data.push({ low: Number(code & 0xffffffffn), high: Number((code >> 32n) & 0xffffffffn) })
      } else {
        data.push({ low: 0, high: 0 })
      }
    }
    customCodewordBuffer.write(data)
    customCountBuf.write([codes.length])
    currentCustomCount = codes.length
  }

  const voteClear = createModuleVoteClear(root, { moduleWhiteBuf, moduleBlackBuf })

  const voteStage = createModuleVoteStage(
    root,
    deps.grayTexView,
    deps.quadDataBuffer,
    thresholdBuf,
    moduleWhiteBuf,
    moduleBlackBuf,
    deps.width,
    deps.height,
  )

  const codewordBuffer = allocCodewordBuffer(root)
  const classifyStage = createClassifyStage(
    root,
    moduleWhiteBuf,
    moduleBlackBuf,
    thresholdBuf,
    patternBuf,
    metaBuf,
    activeQuadCountBuf,
  )
  const dictStage = createDictMatchStage(
    root,
    codewordBuffer,
    customCodewordBuffer,
    customCountBuf,
    patternBuf,
    metaBuf,
    atomicBestBuf,
    activeQuadCountBuf,
  )
  const canonicalizeStage = createCanonicalizeStage(
    root,
    deps.quadDataBuffer,
    atomicBestBuf,
    metaBuf,
    activeQuadCountBuf,
  )

  const atomicBestClear = createAtomicBestClearStage(root, atomicBestBuf)

  function encodeModuleVotePasses(enc: GPUCommandEncoder, quadCount: number) {
    const n = capQuadCount(quadCount)
    if (n < 1) {
      return
    }
    const voteClearPass = profileComputePass(enc, 'tag-vote-clear', { label: 'tag-vote-clear' })
    voteClear.encodeClearModuleVotes(voteClearPass)
    voteClearPass.end()
    voteStage.encodeModuleVotes(enc, n)
  }

  function encodeVotePasses(enc: GPUCommandEncoder, quadCount: number) {
    encodeModuleVotePasses(enc, quadCount)
  }

  function encodeDecode(computePass: GPUComputePassEncoder, quadCount: number) {
    const n = capQuadCount(quadCount)
    if (n < 1) {
      return
    }
    classifyStage.encodeClassify(computePass, n)
    atomicBestClear.encodeClear(computePass)
    dictStage.encodeDictMatch(computePass, n, currentCustomCount)
    canonicalizeStage.encodeCanonicalize(computePass, n)
  }

  return {
    patternBuf,
    moduleWhiteBuf,
    moduleBlackBuf,
    activeQuadCountBuf,
    voteDebugTex: voteStage.debugTex,
    encodeModuleVotePasses,
    encodeVotePasses,
    encodeDecode,
    updateCustomCodewords,
  }
}
