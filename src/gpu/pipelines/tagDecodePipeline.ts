// GPU tag36h11: per-quad pixel histogram → linear peaks → deadband votes → pattern -1/-2 → parallel dict → canonicalize.
import type { ColorAttachment } from 'typegpu'
import type { TgpuRoot } from 'typegpu'
import { d, tgpu, std, common } from 'typegpu'
import { floor, max, min, mul, round, sqrt } from 'typegpu/std'
import { abs, atomicAdd, atomicMin, clamp, countOneBits, textureLoad } from 'typegpu/std'

import { profileComputePass, profileRenderPass } from '@/gpu/gpuProfiling'
import { MAX_QUADS } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import {
  DECODED_TAG_ID_DICT_MISS,
  DECODED_TAG_ID_UNKNOWN,
  GridDataSchema,
  type GridVizQuadBuffer,
} from '@/gpu/pipelines/gridVizPipeline'
import { tryHomographyFromCorners } from '@/gpu/shaders/homographyDlt'
import { Corners4, rotateStripCorners } from '@/gpu/shaders/quadCornerOrder'
import {
  DECODE_MIN_VOTE_FRACTION_OF_QUAD_EDGE,
  TAG_DECODE_HIST_BINS,
  TAG_DECODE_MAX_DICT_ERROR,
  TAG_DECODE_MAX_WEAK_WILDCARD,
  TAG_DECODE_MIN_PEAK_BIN_SEP,
  TAG_DECODE_MIN_PEAK_LUMA_BINS,
  TAG_DECODE_PEAK_GAP_FRAC,
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

const PATTERN_BLACK = 0
const PATTERN_WHITE = 1
const PATTERN_WEAK = -1
const PATTERN_TIE = -2

const PER_QUAD_HIST = MAX_QUADS * TAG_DECODE_HIST_BINS
const WORST_SCORE = ((TAG_DECODE_MAX_DICT_ERROR + 1) << 20) | TAG36H11_COUNT
const WorstScoreGpu = tgpu.const(d.u32, WORST_SCORE)

const CodewordPair = d.struct({ low: d.u32, high: d.u32 })
const CodewordBuffer = d.arrayOf(CodewordPair, TAG36H11_COUNT)

const TagDecodeThresholdGpu = d.struct({
  blackBound: d.f32,
  whiteBound: d.f32,
  minVoteTotal: d.u32,
  valid: d.u32,
  /** Luma histogram bins (for tag-hist debug viz; written in peak-threshold pass). */
  blackPeakBin: d.u32,
  whitePeakBin: d.u32,
  histMaxCount: d.u32,
})

const TagDecodeThresholdSchema = d.arrayOf(TagDecodeThresholdGpu, MAX_QUADS)

const QuadDecodeMetaGpu = d.struct({
  rejectDict: d.u32,
  weakCount: d.u32,
  weakBit0: d.u32,
  weakBit1: d.u32,
  weakBit2: d.u32,
  weakBit3: d.u32,
})

const QuadDecodeMetaSchema = d.arrayOf(QuadDecodeMetaGpu, MAX_QUADS)
export const PatternGrid = d.struct({ modules: d.arrayOf(d.i32, MODULES_PER_QUAD) })
const PatternSchema = d.arrayOf(PatternGrid, MAX_QUADS)
const ModuleVoteGrid = d.struct({ votes: d.arrayOf(d.atomic(d.u32), MODULES_PER_QUAD) })
const ModuleVoteSchema = d.arrayOf(ModuleVoteGrid, MAX_QUADS)
export const ModuleVoteReadonlyGrid = d.struct({ votes: d.arrayOf(d.u32, MODULES_PER_QUAD) })
const ModuleVoteReadonlySchema = d.arrayOf(ModuleVoteReadonlyGrid, MAX_QUADS)
const QuadPixelHistSchema = d.arrayOf(d.atomic(d.u32), PER_QUAD_HIST)
const QuadPixelHistReadonlySchema = d.arrayOf(d.u32, PER_QUAD_HIST)

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

/** GPU zero via plain `u32` mutable views (same buffers as atomic accum passes). */
function createTagDecodeBufferClears(
  root: TgpuRoot,
  deps: {
    histBuf: ReturnType<typeof root.createBuffer<typeof QuadPixelHistSchema>>
    moduleWhiteBuf: ReturnType<typeof root.createBuffer<typeof ModuleVoteSchema>>
    moduleBlackBuf: ReturnType<typeof root.createBuffer<typeof ModuleVoteSchema>>
  },
) {
  const histClearLayout = tgpu.bindGroupLayout({
    histogram: { storage: QuadPixelHistReadonlySchema, access: 'mutable' },
  })
  const histClearKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [CLEAR_WG, 1, 1],
  })((input) => {
    const idx = d.u32(input.gid.x)
    if (idx >= d.u32(PER_QUAD_HIST)) {
      return
    }
    histClearLayout.$.histogram[idx] = d.u32(0)
  })
  const histClearPipeline = root.createComputePipeline({ compute: histClearKernel })
  const histClearBindGroup = root.createBindGroup(histClearLayout, { histogram: deps.histBuf as never })

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

  const histClearWgs = Math.ceil(PER_QUAD_HIST / CLEAR_WG)
  const voteClearWgs = Math.ceil((MAX_QUADS * MODULES_PER_QUAD) / CLEAR_WG)

  return {
    encodeClearHist(pass: GPUComputePassEncoder) {
      histClearPipeline.with(pass).with(histClearBindGroup).dispatchWorkgroups(histClearWgs)
    },
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

function createHistAccumStage(
  root: TgpuRoot,
  grayTexView: unknown,
  quadDataBuffer: GridVizQuadBuffer,
  width: number,
  height: number,
) {
  const histBuf = root.createBuffer(QuadPixelHistSchema).$usage('storage')

  const layout = tgpu.bindGroupLayout({
    quads: { storage: GridDataSchema, access: 'readonly' },
    grayTex: { texture: d.texture2d() },
    histogram: { storage: QuadPixelHistSchema, access: 'mutable' },
  })

  const dummyTexture = root.device.createTexture({
    label: 'tag-decode-dummy',
    size: [width, height, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
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
    const mx = d.u32(floor(uv.x * d.f32(TAG_MODULES)))
    const my = d.u32(floor(uv.y * d.f32(TAG_MODULES)))
    if (mx < d.u32(1) || mx > d.u32(6) || my < d.u32(1) || my > d.u32(6)) {
      return d.vec4f(0, 0, 0, 0)
    }
    const gray = textureLoad(layout.$.grayTex, d.vec2u(d.u32(pos.x), d.u32(pos.y)), d.i32(0)).x
    const hBin = min(d.u32(floor(gray * d.f32(TAG_DECODE_HIST_BINS))), d.u32(TAG_DECODE_HIST_BINS - 1))
    atomicAdd(layout.$.histogram[quadId * d.u32(TAG_DECODE_HIST_BINS) + hBin]!, d.u32(1))
    return d.vec4f(0, 0, 0, 0)
  })

  const pipeline = root
    .createRenderPipeline({
      vertex: vert,
      fragment: frag,
      targets: { format: 'rgba8unorm' },
      primitive: { topology: 'triangle-strip' },
    })
    .$name('tag-hist-accum')

  const bindGroup = root.createBindGroup(layout, {
    quads: quadDataBuffer,
    grayTex: grayTexView as never,
    histogram: histBuf,
  })

  return {
    histBuf,
    dummyTexture,
    encodeHistAccum(enc: GPUCommandEncoder, instanceCount: number) {
      if (instanceCount < 1) {
        return
      }
      const pass = profileRenderPass(enc, pipeline, {
        label: 'tag-hist-accum',
        colorAttachments: [
          { view: dummyTexture.createView(), loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] },
        ],
      })
      pass.setViewport(0, 0, width, height, 0, 1)
      pipeline.with(pass).with(bindGroup).draw(4, instanceCount)
      pass.end()
    },
  }
}

function createPeakThresholdStage(
  root: TgpuRoot,
  histBuf: ReturnType<typeof root.createBuffer<typeof QuadPixelHistSchema>>,
  thresholdBuf: ReturnType<typeof root.createBuffer<typeof TagDecodeThresholdSchema>>,
  quadDataBuffer: GridVizQuadBuffer,
  activeQuadCountBuf: ReturnType<typeof root.createBuffer<typeof ActiveQuadCountSchema>>,
) {
  const layout = tgpu.bindGroupLayout({
    histogram: { storage: QuadPixelHistReadonlySchema, access: 'readonly' },
    thresholds: { storage: TagDecodeThresholdSchema, access: 'mutable' },
    quads: { storage: GridDataSchema, access: 'readonly' },
    activeQuadCount: { storage: ActiveQuadCountSchema, access: 'readonly' },
  })

  const bindGroup = root.createBindGroup(layout, {
    histogram: histBuf as never,
    thresholds: thresholdBuf as never,
    quads: quadDataBuffer,
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

    const base = quadId * d.u32(TAG_DECODE_HIST_BINS)
    const minSep = d.u32(TAG_DECODE_MIN_PEAK_BIN_SEP)
    const minBinSep = minSep + d.u32(1)

    let peak1Bin = d.u32(0)
    let peak1Val = d.u32(0)
    let peak2Bin = d.u32(0)
    let peak2Val = d.u32(0)
    let histMaxCount = d.u32(0)

    for (const b of tgpu.unroll(std.range(0, TAG_DECODE_HIST_BINS))) {
      const bu = d.u32(b)
      const v = layout.$.histogram[base + bu]!
      histMaxCount = max(histMaxCount, v)
      if (v > peak1Val) {
        peak1Val = v
        peak1Bin = bu
      }
    }

    for (const b of tgpu.unroll(std.range(0, TAG_DECODE_HIST_BINS))) {
      const bu = d.u32(b)
      const v = layout.$.histogram[base + bu]!
      let dist = bu - peak1Bin
      if (bu < peak1Bin) {
        dist = peak1Bin - bu
      }
      if (dist >= minBinSep && v > peak2Val) {
        peak2Val = v
        peak2Bin = bu
      }
    }

    let blackPeak = peak1Bin
    let whitePeak = peak1Bin
    let blackVal = peak1Val
    let whiteVal = d.u32(0)
    if (peak2Val > d.u32(0)) {
      blackPeak = min(peak1Bin, peak2Bin)
      whitePeak = max(peak1Bin, peak2Bin)
      blackVal = layout.$.histogram[base + blackPeak]!
      whiteVal = layout.$.histogram[base + whitePeak]!
    }

    const nBins = d.f32(TAG_DECODE_HIST_BINS)
    const blackLuma = (d.f32(blackPeak) + d.f32(0.5)) / nBins
    const whiteLuma = (d.f32(whitePeak) + d.f32(0.5)) / nBins
    const diff = whiteLuma - blackLuma
    const frac = d.f32(TAG_DECODE_PEAK_GAP_FRAC)
    const blackBound = blackLuma + diff * frac
    const whiteBound = whiteLuma - diff * frac

    const c = layout.$.quads[quadId]!.screenCorners
    const e01 = sqrt((c[1]!.x - c[0]!.x) * (c[1]!.x - c[0]!.x) + (c[1]!.y - c[0]!.y) * (c[1]!.y - c[0]!.y))
    const e13 = sqrt((c[3]!.x - c[1]!.x) * (c[3]!.x - c[1]!.x) + (c[3]!.y - c[1]!.y) * (c[3]!.y - c[1]!.y))
    const e32 = sqrt((c[2]!.x - c[3]!.x) * (c[2]!.x - c[3]!.x) + (c[2]!.y - c[3]!.y) * (c[2]!.y - c[3]!.y))
    const e20 = sqrt((c[0]!.x - c[2]!.x) * (c[0]!.x - c[2]!.x) + (c[0]!.y - c[2]!.y) * (c[0]!.y - c[2]!.y))
    let lMin = e01
    lMin = min(lMin, e13)
    lMin = min(lMin, e32)
    lMin = min(lMin, e20)
    const minVote = max(d.u32(2), d.u32(round(d.f32(DECODE_MIN_VOTE_FRACTION_OF_QUAD_EDGE) * lMin)))

    const peakSep = whitePeak - blackPeak
    let valid = d.u32(0)
    if (
      blackVal > d.u32(0) &&
      whiteVal > d.u32(0) &&
      peakSep >= d.u32(TAG_DECODE_MIN_PEAK_LUMA_BINS) &&
      whiteBound > blackBound
    ) {
      valid = d.u32(1)
    }

    layout.$.thresholds[quadId] = TagDecodeThresholdGpu({
      blackBound,
      whiteBound,
      minVoteTotal: minVote,
      valid,
      blackPeakBin: blackPeak,
      whitePeakBin: whitePeak,
      histMaxCount,
    })
  })

  const pipeline = root.createComputePipeline({ compute: kernel }).$name('tag-peaks')

  return {
    pipeline,
    encodePeakThresholds(pass: GPUComputePassEncoder, quadCount: number) {
      const wgs = quadComputeWgs(quadCount)
      if (wgs > 0) {
        pipeline.with(pass).with(bindGroup).dispatchWorkgroups(wgs)
      }
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
        let diff = b
        if (b > w) {
          diff = b - w
        } else {
          diff = w - b
        }
        const minDiff = max(d.u32(2), sum / d.u32(6))
        if (diff >= minDiff) {
          if (b > w) { cell = d.i32(PATTERN_BLACK) }
          else { cell = d.i32(PATTERN_WHITE) }
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

    let reject = d.u32(0)
    if (tieCount > d.u32(0)) {
      reject = d.u32(1)
    }
    layout.$.meta[quadId] = QuadDecodeMetaGpu({
      rejectDict: reject,
      weakCount,
      weakBit0: weakBits[0]!,
      weakBit1: weakBits[1]!,
      weakBit2: weakBits[2]!,
      weakBit3: weakBits[3]!,
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
  patternBuf: ReturnType<typeof root.createBuffer<typeof PatternSchema>>,
  metaBuf: ReturnType<typeof root.createBuffer<typeof QuadDecodeMetaSchema>>,
  atomicBestBuf: ReturnType<typeof root.createBuffer<typeof AtomicBestSchema>>,
  activeQuadCountBuf: ReturnType<typeof root.createBuffer<typeof ActiveQuadCountSchema>>,
) {
  const layout = tgpu.bindGroupLayout({
    codewords: { storage: CodewordBuffer, access: 'readonly' },
    pattern: { storage: PatternSchema, access: 'readonly' },
    meta: { storage: QuadDecodeMetaSchema, access: 'readonly' },
    atomicBest: { storage: AtomicBestSchema, access: 'mutable' },
    activeQuadCount: { storage: ActiveQuadCountSchema, access: 'readonly' },
  })

  const bindGroup = root.createBindGroup(layout, {
    codewords: codewordBuffer as never,
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
    const cwIdx = gid % d.u32(TAG36H11_COUNT)
    const quadId = d.u32(gid / d.u32(TAG36H11_COUNT))
    if (quadId >= layout.$.activeQuadCount[0]!) {
      return
    }

    const meta = layout.$.meta[quadId]!
    if (meta.rejectDict !== d.u32(0)) {
      return
    }

    const weakCount = meta.weakCount
    if (weakCount > d.u32(TAG_DECODE_MAX_WEAK_WILDCARD)) {
      return
    }

    const cw = layout.$.codewords[cwIdx]!
    

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
          let srcBit = d.u32(0)
          if (rot === 0) {
            srcBit = RotLut0Gpu.$[bitU]!
          } else if (rot === 1) {
            srcBit = RotLut1Gpu.$[bitU]!
          } else if (rot === 2) {
            srcBit = RotLut2Gpu.$[bitU]!
          } else {
            srcBit = RotLut3Gpu.$[bitU]!
          }
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
        const diffLow = knownLow ^ cw.low ^ wildLow
        const diffHigh = knownHigh ^ cw.high ^ wildHigh
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
      const candidate = (localBest << d.u32(20)) | (localRot << d.u32(18)) | cwIdx
      atomicMin(layout.$.atomicBest[quadId]!, candidate)
    }
  })

  const pipeline = root.createComputePipeline({ compute: kernel }).$name('tag-dict')

  return {
    encodeDictMatch(pass: GPUComputePassEncoder, quadCount: number) {
      const n = capQuadCount(quadCount)
      const threads = n * TAG36H11_COUNT
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
    if (hLen < d.f32(1e-6)) {
      layout.$.quadData[quadId]!.decodedTagId = d.u32(DECODED_TAG_ID_UNKNOWN)
      return
    }

    if (layout.$.meta[quadId]!.rejectDict !== d.u32(0)) {
      layout.$.quadData[quadId]!.decodedTagId = d.u32(DECODED_TAG_ID_UNKNOWN)
      return
    }

    const score = layout.$.atomicBest[quadId]!
    const dist = score >> d.u32(20)
    const bestRot = (score >> d.u32(18)) & d.u32(3)
    const bestId = score & d.u32(0x3ffff)

    if (dist > d.u32(TAG_DECODE_MAX_DICT_ERROR)) {
      layout.$.quadData[quadId]!.decodedTagId = d.u32(DECODED_TAG_ID_DICT_MISS)
      layout.$.quadData[quadId]!.decodedRotation = d.u32(0)
      return
    }

    layout.$.quadData[quadId]!.decodedTagId = bestId
    layout.$.quadData[quadId]!.decodedRotation = bestRot

    const strip = quad.screenCorners
    const rotated = rotateStripCorners(strip, bestRot)
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

// ------ Tag histogram debug (linear 32-bin) ------

const TAG_HIST_GRID_COLS = 6
/** Screen pixels per luma bin (matches orient hist `ORIENT_HIST_VIZ_PIXEL_SCALE`). */
const TAG_HIST_PIXEL_SCALE = 2
const TAG_HIST_CELL_W = TAG_DECODE_HIST_BINS * TAG_HIST_PIXEL_SCALE
/** Bar area height — same formula as orient `ORIENT_HIST_VIZ_BIN_H`. */
const TAG_HIST_CELL_H = 8 * TAG_HIST_PIXEL_SCALE
const TAG_HIST_GAP = 2
const TAG_HIST_STRIDE_X = TAG_HIST_CELL_W + TAG_HIST_GAP
const TAG_HIST_STRIDE_Y = TAG_HIST_CELL_H + TAG_HIST_GAP
const TAG_HIST_ROWS = Math.ceil(MAX_QUADS / 128)

export const TAG_HIST_CANVAS_W = TAG_HIST_GRID_COLS * TAG_HIST_STRIDE_X - TAG_HIST_GAP
export const TAG_HIST_CANVAS_H = TAG_HIST_ROWS * TAG_HIST_STRIDE_Y - TAG_HIST_GAP

const TagHistParams = d.struct({
  canvasSize: d.vec2u,
})

const tagHistLayout = tgpu.bindGroupLayout({
  params: { uniform: TagHistParams },
  histogram: { storage: QuadPixelHistReadonlySchema, access: 'readonly' },
  thresholds: { storage: TagDecodeThresholdSchema, access: 'readonly' },
})

function createTagHistogramRenderPipeline(root: TgpuRoot, presentationFormat: GPUTextureFormat) {
  const BG = d.vec4f(d.f32(0.08), d.f32(0.08), d.f32(0.1), d.f32(1))
  const GAP_COLOR = d.vec4f(d.f32(0.12), d.f32(0.12), d.f32(0.15), d.f32(1))

  const frag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    const size = tagHistLayout.$.params.canvasSize
    const wi = d.i32(size.x)
    const hi = d.i32(size.y)
    if (wi <= d.i32(0) || hi <= d.i32(0)) {
      return BG
    }

    const maxPx = d.f32(wi - d.i32(1))
    const maxPy = d.f32(hi - d.i32(1))
    const px = d.u32(floor(clamp(i.uv.x * d.f32(wi), d.f32(0), maxPx)))
    const py = d.u32(floor(clamp(i.uv.y * d.f32(hi), d.f32(0), maxPy)))

    const gridCols = d.u32(TAG_HIST_GRID_COLS)
    const cellW = d.u32(TAG_HIST_CELL_W)
    const cellH = d.u32(TAG_HIST_CELL_H)
    const gap = d.u32(TAG_HIST_GAP)
    const strideX = cellW + gap
    const strideY = cellH + gap

    const col = d.u32(d.f32(px) / d.f32(strideX))
    const row = d.u32(d.f32(py) / d.f32(strideY))
    const localX = px % strideX
    const localY = py % strideY

    if (col >= gridCols) {
      return BG
    }
    if (localX >= cellW || localY >= cellH) {
      return GAP_COLOR
    }

    const quadId = row * gridCols + col
    if (quadId >= d.u32(MAX_QUADS)) {
      return BG
    }

    const bin = d.u32(d.f32(localX) / d.f32(TAG_HIST_PIXEL_SCALE))
    const base = quadId * d.u32(TAG_DECODE_HIST_BINS)

    const thr = tagHistLayout.$.thresholds[quadId]!
    const blackPeak = thr.blackPeakBin
    const whitePeak = thr.whitePeakBin
    const maxCount = max(thr.histMaxCount, d.u32(1))
    const count = tagHistLayout.$.histogram[base + bin]!
    let barH = d.u32(0)
    if (count > d.u32(0)) {
      barH = max(d.u32(1), d.u32(d.f32(count * cellH) / d.f32(maxCount)))
    }

    const barTop = cellH - barH
    if (localY < barTop) {
      return BG
    }

    const PEAK = d.vec4f(d.f32(1), d.f32(1), d.f32(0.92), d.f32(1))
    if (bin === blackPeak || bin === whitePeak) {
      return PEAK
    }
    const t = d.f32(bin) / d.f32(TAG_DECODE_HIST_BINS - 1)
    return d.vec4f(t, d.f32(1) - d.f32(2) * abs(t - d.f32(0.5)), d.f32(1) - t, d.f32(1))
  })

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: frag,
    targets: { format: presentationFormat },
  })
}

export function createTagHistogramDisplayStage(
  root: TgpuRoot,
  histBuf: ReturnType<typeof createTagDecodeStage>['histBuf'],
  thresholdBuf: ReturnType<typeof createTagDecodeStage>['thresholdBuf'],
  presentationFormat: GPUTextureFormat,
) {
  const paramsBuffer = root.createBuffer(TagHistParams).$usage('uniform')
  const renderPipeline = createTagHistogramRenderPipeline(root, presentationFormat)

  const bindGroup = root.createBindGroup(tagHistLayout, {
    params: paramsBuffer,
    histogram: histBuf as never,
    thresholds: thresholdBuf as never,
  })

  function encodeDisplay(enc: GPUCommandEncoder, colorAttachment: ColorAttachment) {
    paramsBuffer.write({
      canvasSize: d.vec2u(TAG_HIST_CANVAS_W, TAG_HIST_CANVAS_H),
    })
    renderPipeline.with(enc).withColorAttachment(colorAttachment).with(bindGroup).draw(3)
  }

  return { encodeDisplay }
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
  },
) {
  const histStage = createHistAccumStage(root, deps.grayTexView, deps.quadDataBuffer, deps.width, deps.height)

  const thresholdBuf = root.createBuffer(TagDecodeThresholdSchema).$usage('storage')
  const moduleWhiteBuf = root.createBuffer(ModuleVoteSchema).$usage('storage')
  const moduleBlackBuf = root.createBuffer(ModuleVoteSchema).$usage('storage')
  const patternBuf = root.createBuffer(PatternSchema).$usage('storage')
  const metaBuf = root.createBuffer(QuadDecodeMetaSchema).$usage('storage')
  const atomicBestBuf = root.createBuffer(AtomicBestSchema).$usage('storage')
  const activeQuadCountBuf = root.createBuffer(ActiveQuadCountSchema).$usage('storage')

  const bufferClears = createTagDecodeBufferClears(root, {
    histBuf: histStage.histBuf,
    moduleWhiteBuf,
    moduleBlackBuf,
  })

  const peakStage = createPeakThresholdStage(
    root,
    histStage.histBuf,
    thresholdBuf,
    deps.quadDataBuffer,
    activeQuadCountBuf,
  )
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
  const dictStage = createDictMatchStage(root, codewordBuffer, patternBuf, metaBuf, atomicBestBuf, activeQuadCountBuf)
  const canonicalizeStage = createCanonicalizeStage(
    root,
    deps.quadDataBuffer,
    atomicBestBuf,
    metaBuf,
    activeQuadCountBuf,
  )

  const atomicBestClear = createAtomicBestClearStage(root, atomicBestBuf)

  function encodeHistAndPeaks(enc: GPUCommandEncoder, quadCount: number) {
    const n = capQuadCount(quadCount)
    if (n < 1) {
      return
    }
    const clearPass = profileComputePass(enc, 'tag-decode-clear', { label: 'tag-decode-clear' })
    bufferClears.encodeClearHist(clearPass)
    clearPass.end()
    histStage.encodeHistAccum(enc, n)
    const peakPass = profileComputePass(enc, peakStage.pipeline, { label: 'tag-peaks' })
    peakStage.encodePeakThresholds(peakPass, n)
    peakPass.end()
  }

  function encodeModuleVotePasses(enc: GPUCommandEncoder, quadCount: number) {
    const n = capQuadCount(quadCount)
    if (n < 1) {
      return
    }
    const voteClearPass = profileComputePass(enc, 'tag-vote-clear', { label: 'tag-vote-clear' })
    bufferClears.encodeClearModuleVotes(voteClearPass)
    voteClearPass.end()
    voteStage.encodeModuleVotes(enc, n)
  }

  function encodeVotePasses(enc: GPUCommandEncoder, quadCount: number) {
    encodeHistAndPeaks(enc, quadCount)
    encodeModuleVotePasses(enc, quadCount)
  }

  function encodeDecode(computePass: GPUComputePassEncoder, quadCount: number) {
    const n = capQuadCount(quadCount)
    if (n < 1) {
      return
    }
    classifyStage.encodeClassify(computePass, n)
    atomicBestClear.encodeClear(computePass)
    dictStage.encodeDictMatch(computePass, n)
    canonicalizeStage.encodeCanonicalize(computePass, n)
  }

  return {
    histBuf: histStage.histBuf,
    thresholdBuf,
    patternBuf,
    moduleWhiteBuf,
    moduleBlackBuf,
    activeQuadCountBuf,
    voteDebugTex: voteStage.debugTex,
    encodeHistAndPeaks,
    encodeModuleVotePasses,
    encodeVotePasses,
    encodeDecode,
  }
}
