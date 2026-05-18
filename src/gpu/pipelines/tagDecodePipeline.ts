// GPU tag36h11 quad decoder: render pass fills quads → fragment shader accumulates
// grayscale votes per module via atomics → compute pass builds histogram + matches dictionary.
import type { ColorAttachment } from 'typegpu'
import type { TgpuRoot } from 'typegpu'
import { d, tgpu, std, common } from 'typegpu'
import { floor, max, min, mul, round } from 'typegpu/std'
import { abs, atomicAdd, clamp, countOneBits, textureLoad } from 'typegpu/std'

import { TAG36H11_CODES, TAG36H11_COUNT } from '@/lib/tag36h11'
import {
  DECODED_TAG_ID_DICT_MISS,
  DECODED_TAG_ID_UNKNOWN,
  GridDataSchema,
  type GridVizQuadBuffer,
  MAX_INSTANCES,
} from '@/gpu/pipelines/gridVizPipeline'

const MAX_QUADS = MAX_INSTANCES
const TAG_MODULES = 8
const DATA_MODULES = 6
const MAX_DICT_ERROR = 3

const BIT_POS = [
  0, 1, 2, 3, 4, 9, 31, 5, 6, 7, 14, 10, 30, 34, 8, 17, 15, 11, 29, 33, 35, 26, 16, 12, 28, 32, 25, 24, 23, 13, 27, 22,
  21, 20, 19, 18,
] as const

const ROT_LUTS_0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35] as const
const ROT_LUTS_1 = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 3, 4, 5, 6, 7, 8] as const
const ROT_LUTS_2 = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] as const
const ROT_LUTS_3 = [27, 28, 29, 30, 31, 32, 33, 34, 35, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26] as const

const CodewordPair = d.struct({ low: d.u32, high: d.u32 })
const CodewordBuffer = d.arrayOf(CodewordPair, TAG36H11_COUNT)

function allocCodewordBuffer(root: TgpuRoot) {
  const buf = root.createBuffer(CodewordBuffer).$usage('storage')
  const data: { low: number; high: number }[] = []
  for (const code of TAG36H11_CODES) {
    data.push({ low: Number(code & 0xffffffffn), high: Number((code >> 32n) & 0xfn) })
  }
  buf.write(data)
  return buf
}

// ------ Stage 1: Quad fill + vote accumulation render pass ------
const MODULE_COUNT = MAX_QUADS * DATA_MODULES * DATA_MODULES
const HIST_BINS = 16
const PER_QUAD_HIST = MAX_QUADS * HIST_BINS

function createVoteAccumStage(
  root: TgpuRoot,
  grayTexView: unknown,
  quadDataBuffer: GridVizQuadBuffer,
  width: number,
  height: number,
) {
  const moduleSumBuf = root.createBuffer(d.arrayOf(d.atomic(d.u32), MODULE_COUNT)).$usage('storage')
  const moduleCountBuf = root.createBuffer(d.arrayOf(d.atomic(d.u32), MODULE_COUNT)).$usage('storage')
  const histBuf = root.createBuffer(d.arrayOf(d.atomic(d.u32), PER_QUAD_HIST)).$usage('storage')

  const voteLayout = tgpu.bindGroupLayout({
    quads: { storage: GridDataSchema, access: 'readonly' },
    grayTex: { texture: d.texture2d() },
    moduleSum: { storage: d.arrayOf(d.atomic(d.u32), MODULE_COUNT), access: 'mutable' },
    moduleCount: { storage: d.arrayOf(d.atomic(d.u32), MODULE_COUNT), access: 'mutable' },
    histogram: { storage: d.arrayOf(d.atomic(d.u32), PER_QUAD_HIST), access: 'mutable' },
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
    const quad = voteLayout.$.quads[instanceIndex]!
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
    // textureLoad first — uses pixel coords from the rasterizer, before any branching
    const gray = textureLoad(voteLayout.$.grayTex, d.vec2u(d.u32(pos.x), d.u32(pos.y)), d.i32(0)).x

    // DEBUG: accumulate grayscale values into per-quad 16-bin histogram
    const hBin = min(d.u32(floor(gray * d.f32(HIST_BINS))), d.u32(HIST_BINS - 1))
    atomicAdd(voteLayout.$.histogram[quadId * d.u32(HIST_BINS) + hBin]!, d.u32(1))

    const mx = d.u32(floor(uv.x * d.f32(TAG_MODULES)))
    const my = d.u32(floor(uv.y * d.f32(TAG_MODULES)))
    if (mx < d.u32(1) || mx > d.u32(6) || my < d.u32(1) || my > d.u32(6)) {
      return d.vec4f(0, 0, 0, 0)
    }
    const cellIdx = (my - d.u32(1)) * d.u32(DATA_MODULES) + (mx - d.u32(1))
    const bufIdx = quadId * d.u32(DATA_MODULES * DATA_MODULES) + cellIdx

    const fixedPoint = d.u32(round(gray * d.f32(65536)))
    atomicAdd(voteLayout.$.moduleSum[bufIdx]!, fixedPoint)
    atomicAdd(voteLayout.$.moduleCount[bufIdx]!, d.u32(1))

    return d.vec4f(0, 0, 0, 0)
  })

  const pipeline = root.createRenderPipeline({
    vertex: vert,
    fragment: frag,
    targets: { format: 'rgba8unorm' },
    primitive: { topology: 'triangle-strip' },
  })

  const voteBindGroup = root.createBindGroup(voteLayout, {
    quads: quadDataBuffer,
    grayTex: grayTexView as never,
    moduleSum: moduleSumBuf,
    moduleCount: moduleCountBuf,
    histogram: histBuf,
  })

  const zeroBuf = new Uint32Array(MODULE_COUNT as number)
  const zeroHist = new Uint32Array(PER_QUAD_HIST as number)

  function encodeVotes(enc: GPUCommandEncoder, instanceCount: number) {
    if (instanceCount < 1) return
    moduleSumBuf.write(zeroBuf)
    moduleCountBuf.write(zeroBuf)
    histBuf.write(zeroHist)
    const pass = enc.beginRenderPass({
      label: 'tag vote accum',
      colorAttachments: [
        { view: dummyTexture.createView(), loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] },
      ],
    })
    pass.setViewport(0, 0, width, height, 0, 1)
    pipeline.with(pass).with(voteBindGroup).draw(4, instanceCount)
    pass.end()
  }

  return { moduleSumBuf, moduleCountBuf, histBuf, encodeVotes, dummyTexture }
}

// ------ Stage 2: Per-quad histogram + decode compute pass ------

const WORKGROUP_SIZE = 64
const DECODE_SUM = d.arrayOf(d.u32, MODULE_COUNT)
const DECODE_CNT = d.arrayOf(d.u32, MODULE_COUNT)

function createQuadDecodeComputeStage(
  root: TgpuRoot,
  moduleSumBuf: ReturnType<typeof root.createBuffer>,
  moduleCountBuf: ReturnType<typeof root.createBuffer>,
  quadDataBuffer: GridVizQuadBuffer,
) {
  const codewordBuffer = allocCodewordBuffer(root)

  const decodeLayout = tgpu.bindGroupLayout({
    moduleSum: { storage: DECODE_SUM, access: 'readonly' },
    moduleCount: { storage: DECODE_CNT, access: 'readonly' },
    quadData: { storage: GridDataSchema, access: 'mutable' },
    codewords: { storage: CodewordBuffer, access: 'readonly' },
  })

  const decodeBindGroup = root.createBindGroup(decodeLayout, {
    moduleSum: moduleSumBuf as never,
    moduleCount: moduleCountBuf as never,
    quadData: quadDataBuffer,
    codewords: codewordBuffer,
  })

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    const quadId = d.u32(input.gid.x)
    if (quadId >= d.u32(MAX_QUADS)) return

    const quad = decodeLayout.$.quadData[quadId]!
    const H = quad.homography
    const c0 = H.columns[0]!
    const c1 = H.columns[1]!
    const hLen = c0.x * c0.x + c0.y * c0.y + c1.x * c1.x + c1.y * c1.y
    if (hLen < d.f32(1e-6)) {
      decodeLayout.$.quadData[quadId]!.decodedTagId = d.u32(DECODED_TAG_ID_UNKNOWN)
      return
    }

    // 1. Read 36 module averages, build 16-bin histogram
    let h0 = d.u32(0)
    let h1 = d.u32(0)
    let h2 = d.u32(0)
    let h3 = d.u32(0)
    let h4 = d.u32(0)
    let h5 = d.u32(0)
    let h6 = d.u32(0)
    let h7 = d.u32(0)
    let h8 = d.u32(0)
    let h9 = d.u32(0)
    let hA = d.u32(0)
    let hB = d.u32(0)
    let hC = d.u32(0)
    let hD = d.u32(0)
    let hE = d.u32(0)
    let hF = d.u32(0)

    const base = quadId * d.u32(DATA_MODULES * DATA_MODULES)
    for (const i of tgpu.unroll(std.range(0, DATA_MODULES * DATA_MODULES))) {
      const sum = decodeLayout.$.moduleSum[base + d.u32(i)]!
      const count = decodeLayout.$.moduleCount[base + d.u32(i)]!
      let avg = d.f32(0)
      if (count > d.u32(0)) {
        avg = d.f32(sum) / d.f32(65536) / d.f32(count)
      }
      const bin = min(d.u32(floor(avg * d.f32(16))), d.u32(15))
      if (bin === d.u32(0)) { h0 = h0 + d.u32(1) }
      if (bin === d.u32(1)) { h1 = h1 + d.u32(1) }
      if (bin === d.u32(2)) { h2 = h2 + d.u32(1) }
      if (bin === d.u32(3)) { h3 = h3 + d.u32(1) }
      if (bin === d.u32(4)) { h4 = h4 + d.u32(1) }
      if (bin === d.u32(5)) { h5 = h5 + d.u32(1) }
      if (bin === d.u32(6)) { h6 = h6 + d.u32(1) }
      if (bin === d.u32(7)) { h7 = h7 + d.u32(1) }
      if (bin === d.u32(8)) { h8 = h8 + d.u32(1) }
      if (bin === d.u32(9)) { h9 = h9 + d.u32(1) }
      if (bin === d.u32(10)) { hA = hA + d.u32(1) }
      if (bin === d.u32(11)) { hB = hB + d.u32(1) }
      if (bin === d.u32(12)) { hC = hC + d.u32(1) }
      if (bin === d.u32(13)) { hD = hD + d.u32(1) }
      if (bin === d.u32(14)) { hE = hE + d.u32(1) }
      if (bin === d.u32(15)) { hF = hF + d.u32(1) }
    }

    const hist = [h0, h1, h2, h3, h4, h5, h6, h7, h8, h9, hA, hB, hC, hD, hE, hF]

    // 2. Smooth with 3-wide kernel (circular)
    const s0 = hF + hist[0]! + hist[1]!
    const s1 = hist[0]! + hist[1]! + hist[2]!
    const s2 = hist[1]! + hist[2]! + hist[3]!
    const s3 = hist[2]! + hist[3]! + hist[4]!
    const s4 = hist[3]! + hist[4]! + hist[5]!
    const s5 = hist[4]! + hist[5]! + hist[6]!
    const s6 = hist[5]! + hist[6]! + hist[7]!
    const s7 = hist[6]! + hist[7]! + hist[8]!
    const s8 = hist[7]! + hist[8]! + hist[9]!
    const s9 = hist[8]! + hist[9]! + hist[10]!
    const sA = hist[9]! + hist[10]! + hist[11]!
    const sB = hist[10]! + hist[11]! + hist[12]!
    const sC = hist[11]! + hist[12]! + hist[13]!
    const sD = hist[12]! + hist[13]! + hist[14]!
    const sE = hist[13]! + hist[14]! + hist[15]!
    const sF = hist[14]! + hist[15]! + hist[0]!
    const smoothed = [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD, sE, sF]

    // 3. Find two peaks
    let blackPeak = d.u32(0)
    let blackPeakVal = d.u32(0)
    for (const i of tgpu.unroll(std.range(0, 16))) {
      if (smoothed[i]! > blackPeakVal) { blackPeakVal = smoothed[i]!; blackPeak = d.u32(i) }
    }

    let whitePeak = d.u32(0)
    let whitePeakVal = d.u32(0)
    for (const i of tgpu.unroll(std.range(0, 16))) {
      const iu = d.u32(i)
      const binDist = min(iu, blackPeak) + d.u32(16) - max(iu, blackPeak)
      const dist = min(binDist, d.u32(16) - binDist)
      const prev = smoothed[(i + 15) % 16]!
      const next = smoothed[(i + 1) % 16]!
      const v = smoothed[i]!
      if (v >= prev && v >= next && dist >= d.u32(4) && v > whitePeakVal) {
        whitePeakVal = v
        whitePeak = iu
      }
    }

    if (whitePeakVal === d.u32(0)) {
      let maxDist = d.u32(0)
      for (const i of tgpu.unroll(std.range(0, 16))) {
        const iu = d.u32(i)
        const binDist = min(iu, blackPeak) + d.u32(16) - max(iu, blackPeak)
        const dist = min(binDist, d.u32(16) - binDist)
        if (dist > maxDist) { maxDist = dist; whitePeak = iu }
      }
    }

    const threshold = (d.f32(blackPeak) + d.f32(whitePeak)) * d.f32(0.5) / d.f32(16)

    // 4. Classify 36 modules, pack codeword
    let codewordLow = d.u32(0)
    let codewordHigh = d.u32(0)
    for (const i of tgpu.unroll(std.range(0, DATA_MODULES * DATA_MODULES))) {
      const iu = d.u32(i)
      const sum = decodeLayout.$.moduleSum[base + iu]!
      const count = decodeLayout.$.moduleCount[base + iu]!
      let avg = d.f32(0.5)
      if (count > d.u32(0)) { avg = d.f32(sum) / d.f32(65536) / d.f32(count) }
      if (avg >= threshold) {
        const bitIdx = d.u32(BIT_POS[i]!)
        if (bitIdx === d.u32(32)) { codewordHigh = codewordHigh | d.u32(1) }
        else if (bitIdx === d.u32(33)) { codewordHigh = codewordHigh | d.u32(2) }
        else if (bitIdx === d.u32(34)) { codewordHigh = codewordHigh | d.u32(4) }
        else if (bitIdx === d.u32(35)) { codewordHigh = codewordHigh | d.u32(8) }
        else { codewordLow = codewordLow | (d.u32(1) << (bitIdx & d.u32(31))) }
      }
    }

    // 5. Build 4 rotated codewords (unrolled rotation, compile-time)
    let rLow0 = d.u32(0)
    let rHigh0 = d.u32(0)
    let rLow1 = d.u32(0)
    let rHigh1 = d.u32(0)
    let rLow2 = d.u32(0)
    let rHigh2 = d.u32(0)
    let rLow3 = d.u32(0)
    let rHigh3 = d.u32(0)

    for (const rot of tgpu.unroll(std.range(0, 4))) {
      let rLow = d.u32(0)
      let rHigh = d.u32(0)
      for (const bit of tgpu.unroll(std.range(0, 36))) {
        let srcBit = d.u32(0)
        if (rot === 0) { srcBit = d.u32(ROT_LUTS_0[bit]!) }
        else if (rot === 1) { srcBit = d.u32(ROT_LUTS_1[bit]!) }
        else if (rot === 2) { srcBit = d.u32(ROT_LUTS_2[bit]!) }
        else { srcBit = d.u32(ROT_LUTS_3[bit]!) }
        let srcVal = d.u32(0)
        if (srcBit === d.u32(32)) { srcVal = codewordHigh & d.u32(1) }
        else if (srcBit === d.u32(33)) { srcVal = (codewordHigh >> d.u32(1)) & d.u32(1) }
        else if (srcBit === d.u32(34)) { srcVal = (codewordHigh >> d.u32(2)) & d.u32(1) }
        else if (srcBit === d.u32(35)) { srcVal = (codewordHigh >> d.u32(3)) & d.u32(1) }
        else { srcVal = (codewordLow >> (srcBit & d.u32(31))) & d.u32(1) }
        if (srcVal !== d.u32(0)) {
          const dstBit = d.u32(bit)
          if (dstBit === d.u32(32)) { rHigh = rHigh | d.u32(1) }
          else if (dstBit === d.u32(33)) { rHigh = rHigh | d.u32(2) }
          else if (dstBit === d.u32(34)) { rHigh = rHigh | d.u32(4) }
          else if (dstBit === d.u32(35)) { rHigh = rHigh | d.u32(8) }
          else { rLow = rLow | (d.u32(1) << (dstBit & d.u32(31))) }
        }
      }
      if (rot === 0) { rLow0 = rLow; rHigh0 = rHigh }
      else if (rot === 1) { rLow1 = rLow; rHigh1 = rHigh }
      else if (rot === 2) { rLow2 = rLow; rHigh2 = rHigh }
      else { rLow3 = rLow; rHigh3 = rHigh }
    }

    // 6. Dictionary match
    let bestDist = d.u32(MAX_DICT_ERROR + 1)
    let bestId = d.u32(DECODED_TAG_ID_UNKNOWN)
    let bestRot = d.u32(0)

    for (let cwIdx = d.u32(0); cwIdx < d.u32(TAG36H11_COUNT); cwIdx = cwIdx + d.u32(1)) {
      const cw = decodeLayout.$.codewords[cwIdx]!

      let d0 = countOneBits(rLow0 ^ cw.low) + countOneBits(rHigh0 ^ cw.high)
      if (d0 < bestDist) { bestDist = d0; bestId = cwIdx; bestRot = d.u32(0) }
      else if (d0 === bestDist) { if (cwIdx < bestId) { bestId = cwIdx; bestRot = d.u32(0) } }

      let d1 = countOneBits(rLow1 ^ cw.low) + countOneBits(rHigh1 ^ cw.high)
      if (d1 < bestDist) { bestDist = d1; bestId = cwIdx; bestRot = d.u32(1) }
      else if (d1 === bestDist) { if (cwIdx < bestId) { bestId = cwIdx; bestRot = d.u32(1) } }

      let d2 = countOneBits(rLow2 ^ cw.low) + countOneBits(rHigh2 ^ cw.high)
      if (d2 < bestDist) { bestDist = d2; bestId = cwIdx; bestRot = d.u32(2) }
      else if (d2 === bestDist) { if (cwIdx < bestId) { bestId = cwIdx; bestRot = d.u32(2) } }

      let d3 = countOneBits(rLow3 ^ cw.low) + countOneBits(rHigh3 ^ cw.high)
      if (d3 < bestDist) { bestDist = d3; bestId = cwIdx; bestRot = d.u32(3) }
      else if (d3 === bestDist) { if (cwIdx < bestId) { bestId = cwIdx; bestRot = d.u32(3) } }
    }

    if (bestDist <= d.u32(MAX_DICT_ERROR)) {
      decodeLayout.$.quadData[quadId]!.decodedTagId = bestId
      decodeLayout.$.quadData[quadId]!.decodedRotation = bestRot
    } else {
      decodeLayout.$.quadData[quadId]!.decodedTagId = d.u32(DECODED_TAG_ID_DICT_MISS)
      decodeLayout.$.quadData[quadId]!.decodedRotation = d.u32(0)
    }
  })

  const pipeline = root.createComputePipeline({ compute: kernel })

  return {
    codewordBuffer,
    decodeBindGroup,
    encodeDecode(computePass: GPUComputePassEncoder) {
      pipeline.with(computePass).with(decodeBindGroup).dispatchWorkgroups(16, 1, 1)
    },
  }
}

// ------ Stage 3: Per-quad histogram visualization render pass ------

const TAG_HIST_GRID_COLS = 6
const TAG_HIST_PIXEL_SCALE = 8
const TAG_HIST_CELL_W = HIST_BINS * TAG_HIST_PIXEL_SCALE
const TAG_HIST_CELL_H = 4 * TAG_HIST_PIXEL_SCALE
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
  histogram: { storage: d.arrayOf(d.u32, PER_QUAD_HIST), access: 'readonly' },
})

function createTagHistogramRenderPipeline(root: TgpuRoot, presentationFormat: GPUTextureFormat) {
  const BG = d.vec4f(d.f32(0.08), d.f32(0.08), d.f32(0.1), d.f32(1))
  const GAP_COLOR = d.vec4f(d.f32(0.12), d.f32(0.12), d.f32(0.15), d.f32(1))

  const frag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu'
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

    const col = d.u32(px / strideX)
    const row = d.u32(py / strideY)
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

    const bin = d.u32(localX / d.u32(TAG_HIST_PIXEL_SCALE))
    const base = quadId * d.u32(HIST_BINS)

    // Find black peak (highest bin) — histograms track per-module grayscale averages
    let blackPeak = d.u32(0)
    let blackVal = d.u32(0)
    for (const b of tgpu.unroll(std.range(0, HIST_BINS))) {
      const v = tagHistLayout.$.histogram[base + d.u32(b)]!
      if (v > blackVal) {
        blackVal = v
        blackPeak = d.u32(b)
      }
    }

    // Find white peak (must be local max, ≥4 bins from black)
    let whitePeak = d.u32(0)
    let whiteVal = d.u32(0)
    for (const b of tgpu.unroll(std.range(0, HIST_BINS))) {
      const iu = d.u32(b)
      const dist = min(iu, blackPeak) + d.u32(HIST_BINS) - max(iu, blackPeak)
      const sep = min(dist, d.u32(HIST_BINS) - dist)
      const v = tagHistLayout.$.histogram[base + iu]!
      const prev = tagHistLayout.$.histogram[base + d.u32((b + HIST_BINS - 1) % HIST_BINS)]!
      const next = tagHistLayout.$.histogram[base + d.u32((b + 1) % HIST_BINS)]!
      if (v >= prev && v >= next && sep >= d.u32(4) && v > whiteVal) {
        whiteVal = v
        whitePeak = iu
      }
    }

    // Fallback white peak if no local max found
    if (whiteVal === d.u32(0) && blackVal > d.u32(0)) {
      let maxSep = d.u32(0)
      for (const b of tgpu.unroll(std.range(0, HIST_BINS))) {
        const iu = d.u32(b)
        const dist = min(iu, blackPeak) + d.u32(HIST_BINS) - max(iu, blackPeak)
        const sep = min(dist, d.u32(HIST_BINS) - dist)
        if (sep > maxSep) {
          maxSep = sep
          whitePeak = iu
        }
      }
    }

    const maxCount = max(blackVal, d.u32(1))
    const count = tagHistLayout.$.histogram[base + bin]!
    let barH = d.u32(0)
    if (count > d.u32(0)) {
      barH = max(d.u32(1), d.u32((count * cellH) / maxCount))
    }

    const barTop = cellH - barH
    if (localY < barTop) {
      return BG
    }

    // Color: peaks in white, otherwise gradient blue→white→red
    const PEAK = d.vec4f(d.f32(1), d.f32(1), d.f32(0.92), d.f32(1))
    if (bin === blackPeak || bin === whitePeak) {
      return PEAK
    }
    const t = d.f32(bin) / d.f32(HIST_BINS - 1)
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
  presentationFormat: GPUTextureFormat,
) {
  const paramsBuffer = root.createBuffer(TagHistParams).$usage('uniform')
  const renderPipeline = createTagHistogramRenderPipeline(root, presentationFormat)

  const bindGroup = root.createBindGroup(tagHistLayout, {
    params: paramsBuffer,
    histogram: histBuf,
  })

  function encodeDisplay(enc: GPUCommandEncoder, colorAttachment: ColorAttachment) {
    paramsBuffer.write({
      canvasSize: d.vec2u(TAG_HIST_CANVAS_W, TAG_HIST_CANVAS_H),
    })
    renderPipeline.with(enc).withColorAttachment(colorAttachment).with(bindGroup).draw(3)
  }

  return { encodeDisplay }
}

// ------ Public API ------

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
  const voteAccum = createVoteAccumStage(root, deps.grayTexView, deps.quadDataBuffer, deps.width, deps.height)
  const decode = createQuadDecodeComputeStage(
    root,
    voteAccum.moduleSumBuf,
    voteAccum.moduleCountBuf,
    deps.quadDataBuffer,
  )

  return {
    histBuf: voteAccum.histBuf,
    encodeVotes: voteAccum.encodeVotes,
    encodeDecode: decode.encodeDecode,
  }
}
