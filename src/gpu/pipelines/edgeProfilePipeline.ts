// Per-edge 64-bin grayscale profiles along edge normal (black → white), via quad rasterization.
import type { TgpuRoot } from 'typegpu'
import { d, std, tgpu } from 'typegpu'
import { atomicAdd, atomicLoad, atomicStore, floor, max, min, round, select, sqrt, textureLoad } from 'typegpu/std'

import { profileComputePass, profileRenderPass } from '@/gpu/gpuProfiling'
import { DECODE_MIN_VOTE_FRACTION_OF_QUAD_EDGE } from '@/gpu/tagDecodeThresholds'
import { MAX_EDGES_PER_LABEL } from '@/gpu/lineFitThresholds'
import { MAX_QUADS } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import { GridDataSchema, type GridVizQuadBuffer } from '@/gpu/pipelines/gridVizPipeline'
import {
  EdgeLineEntry,
  PROFILE_BUCKET_COUNT,
  PROFILE_NEIGHBORHOOD_HALF,
  type EdgeLineOutBuffer,
} from '@/gpu/pipelines/edgeLineFitPipeline'
import {
  TagDecodeThresholdGpu,
  TagDecodeThresholdSchema,
} from '@/gpu/pipelines/tagDecodePipeline'

export { PROFILE_BUCKET_COUNT, PROFILE_NEIGHBORHOOD_HALF }

const BUCKET_WORKGROUP_SIZE = 256
const MINMAX_WG = 64
const BUCKETS_PER_EDGE = PROFILE_BUCKET_COUNT
const GRAY_FIXED_SCALE = 100000
const HALF_W_F = d.f32(PROFILE_NEIGHBORHOOD_HALF)
const NORM_SPAN = d.f32(2) * HALF_W_F
const BUCKET_COUNT_F = d.f32(PROFILE_BUCKET_COUNT)

const ProfileBucketAtomic = d.struct({
  sumGrayFixed: d.atomic(d.i32),
  count: d.atomic(d.u32),
})

export const ProfileBucketGpu = d.struct({
  sumGrayFixed: d.i32,
  count: d.u32,
})

// ---- reset ----

function createProfileResetPipeline(root: TgpuRoot, bucketCount: number) {
  const resetLayout = tgpu.bindGroupLayout({
    profileBuckets: { storage: d.arrayOf(ProfileBucketAtomic), access: 'mutable' },
    profileAvg: { storage: d.arrayOf(d.f32), access: 'mutable' },
  })

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [BUCKET_WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const bid = d.u32(input.gid.x)
    if (bid >= d.u32(bucketCount)) {
      return
    }
    atomicStore(resetLayout.$.profileBuckets[bid]!.sumGrayFixed, d.i32(0))
    atomicStore(resetLayout.$.profileBuckets[bid]!.count, d.u32(0))
    resetLayout.$.profileAvg[bid] = d.f32(0)
  })
  const pipeline = root.createComputePipeline({ compute: kernel })
  return { resetLayout, pipeline }
}

// ---- accum (render pass: rasterize edge quads) ----

function createProfileAccumStage(
  root: TgpuRoot,
  grayTexView: unknown,
  lineOut: EdgeLineOutBuffer,
  profileBuckets: ReturnType<typeof root.createBuffer>,
  width: number,
  height: number,
  maxFlatEdges: number,
  presentationFormat: GPUTextureFormat,
) {
  const accumLayout = tgpu.bindGroupLayout({
    grayTex: { texture: d.texture2d(d.f32) },
    lineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'readonly' },
    profileBuckets: { storage: d.arrayOf(ProfileBucketAtomic), access: 'mutable' },
  }).$name('profile-accum-bgl')

  const vert = tgpu.vertexFn({
    in: { vertexIndex: d.builtin.vertexIndex, instanceIndex: d.builtin.instanceIndex },
    out: {
      outPos: d.builtin.position,
      flatIdx: d.interpolate('flat', d.u32),
      s: d.vec2f,
    },
  })(({ vertexIndex, instanceIndex }) => {
    const line = accumLayout.$.lineOut[instanceIndex]!
    if (line.valid === d.u32(0)) {
      return { outPos: d.vec4f(-2, -2, 0, 1), flatIdx: instanceIndex, s: d.vec2f(0, 0) }
    }

    const gLen = sqrt(line.sumGx * line.sumGx + line.sumGy * line.sumGy)
    let nnx = d.f32(0)
    let nny = d.f32(0)
    if (gLen >= d.f32(1e-8)) {
      nnx = line.sumGx / gLen
      nny = line.sumGy / gLen
    }

    // Quad strip: expand endpoints p0/p1 along normal by ±halfW
    // v0=TL(p0,+half), v1=TR(p1,+half), v2=BL(p0,-half), v3=BR(p1,-half)
    const isTop = vertexIndex === d.u32(0) || vertexIndex === d.u32(1)
    const isP1 = vertexIndex === d.u32(1) || vertexIndex === d.u32(3)
    const side = select(d.f32(-PROFILE_NEIGHBORHOOD_HALF), HALF_W_F, isTop)
    const ex = select(line.p0x, line.p1x, isP1)
    const ey = select(line.p0y, line.p1y, isP1)

    const px = ex + nnx * side
    const py = ey + nny * side

    const fw = d.f32(width)
    const fh = d.f32(height)
    const clipX = (d.f32(2) * px) / fw - d.f32(1)
    const clipY = d.f32(1) - (d.f32(2) * py) / fh

    return {
      outPos: d.vec4f(clipX, clipY, d.f32(0), d.f32(1)),
      flatIdx: instanceIndex,
      s: d.vec2f(side, d.f32(0)),
    }
  })

  const frag = tgpu.fragmentFn({
    in: {
      pos: d.builtin.position,
      flatIdx: d.interpolate('flat', d.u32),
      s: d.vec2f,
    },
    out: d.vec4f,
  })(({ pos, flatIdx, s }) => {
    const signedDist = s.x
    const frac_ = (signedDist + HALF_W_F) / NORM_SPAN
    let b = d.u32(floor(frac_ * BUCKET_COUNT_F))
    b = min(max(b, d.u32(0)), d.u32(BUCKETS_PER_EDGE - 1))

    const px = d.u32(pos.x)
    const py = d.u32(pos.y)
    const gray = textureLoad(accumLayout.$.grayTex, d.vec2u(px, py), d.i32(0)).x
    const bucketIdx = flatIdx * d.u32(BUCKETS_PER_EDGE) + b
    atomicAdd(accumLayout.$.profileBuckets[bucketIdx]!.sumGrayFixed, d.i32(gray * d.f32(GRAY_FIXED_SCALE)))
    atomicAdd(accumLayout.$.profileBuckets[bucketIdx]!.count, d.u32(1))
    return d.vec4f(0, 0, 0, 0)
  })

  const pipeline = root
    .createRenderPipeline({
      vertex: vert,
      fragment: frag,
      targets: { format: presentationFormat },
      primitive: { topology: 'triangle-strip' },
    })
    .$name('profile-accum-render')

  const bindGroup = root.createBindGroup(accumLayout, {
    grayTex: grayTexView as never,
    lineOut,
    profileBuckets: profileBuckets as never,
  })

  const dummyTexture = root.device.createTexture({
    label: 'profile-accum-dummy',
    size: [width, height, 1],
    format: presentationFormat,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  })

  function encodeAccum(enc: GPUCommandEncoder) {
    const pass = profileRenderPass(enc, pipeline, {
      label: 'profile-accum',
      colorAttachments: [
        { view: dummyTexture.createView(), loadOp: 'clear', storeOp: 'discard', clearValue: [0, 0, 0, 0] },
      ],
    })
    pass.setViewport(0, 0, width, height, 0, 1)
    pipeline.with(pass).with(bindGroup).draw(4, maxFlatEdges)
    pass.end()
  }

  return { encodeAccum }
}

// ---- normalize ----

function createProfileNormalizePipeline(root: TgpuRoot, bucketCount: number) {
  const normalizeLayout = tgpu.bindGroupLayout({
    profileBuckets: { storage: d.arrayOf(ProfileBucketAtomic), access: 'mutable' },
    profileAvg: { storage: d.arrayOf(d.f32), access: 'mutable' },
  })

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [BUCKET_WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const bid = d.u32(input.gid.x)
    if (bid >= d.u32(bucketCount)) {
      return
    }
    const slot = normalizeLayout.$.profileBuckets[bid]!
    const c = atomicLoad(slot.count)
    const sumFixed = atomicLoad(slot.sumGrayFixed)
    if (c > d.u32(0)) {
      normalizeLayout.$.profileAvg[bid] = d.f32(sumFixed) / d.f32(c) / d.f32(GRAY_FIXED_SCALE)
    } else {
      normalizeLayout.$.profileAvg[bid] = d.f32(0)
    }
  })
  const pipeline = root.createComputePipeline({ compute: kernel })
  return { normalizeLayout, pipeline }
}

// ---- min/max extraction per quad ----

function createProfileMinMaxStage(
  root: TgpuRoot,
  profileBuckets: ReturnType<typeof root.createBuffer>,
  profileAvg: ReturnType<typeof root.createBuffer>,
  quadDataBuffer: GridVizQuadBuffer,
  quadCountBuf: unknown,
) {
  const minmaxLayout = tgpu.bindGroupLayout({
    profileBuckets: { storage: d.arrayOf(ProfileBucketGpu), access: 'readonly' },
    profileAvg: { storage: d.arrayOf(d.f32), access: 'readonly' },
    quads: { storage: GridDataSchema, access: 'readonly' },
    thresholds: { storage: TagDecodeThresholdSchema, access: 'mutable' },
    activeQuadCount: { storage: d.arrayOf(d.u32, 1), access: 'readonly' },
  }).$name('profile-minmax-bgl')

  const thresholdBuf = root.createBuffer(TagDecodeThresholdSchema).$usage('storage').$name('tag-decode-thresholds')

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [MINMAX_WG, 1, 1],
  })((input) => {
    'use gpu'
    const quadId = d.u32(input.gid.x)
    if (quadId >= minmaxLayout.$.activeQuadCount[0]!) {
      return
    }

    let minGray = d.f32(1)
    let maxGray = d.f32(0)

    for (const b of tgpu.unroll(std.range(0, BUCKETS_PER_EDGE))) {
      const bu = d.u32(b)
      let bucketSum = d.f32(0)
      let edgeCount = d.u32(0)

      for (const e of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
        const flatIdx = quadId * d.u32(MAX_EDGES_PER_LABEL) + d.u32(e)
        const idx = flatIdx * d.u32(BUCKETS_PER_EDGE) + bu
        if (minmaxLayout.$.profileBuckets[idx]!.count > d.u32(0)) {
          bucketSum = bucketSum + minmaxLayout.$.profileAvg[idx]!
          edgeCount = edgeCount + d.u32(1)
        }
      }

      if (edgeCount > d.u32(0)) {
        const avg = bucketSum / d.f32(edgeCount)
        minGray = min(minGray, avg)
        maxGray = max(maxGray, avg)
      }
    }

    const diff = maxGray - minGray
    const gapFrac = d.f32(0.375)
    const blackBound = minGray + diff * gapFrac
    const whiteBound = maxGray - diff * gapFrac

    const c = minmaxLayout.$.quads[quadId]!.screenCorners
    const d01 = sqrt((c[1]!.x - c[0]!.x) * (c[1]!.x - c[0]!.x) + (c[1]!.y - c[0]!.y) * (c[1]!.y - c[0]!.y))
    const d13 = sqrt((c[3]!.x - c[1]!.x) * (c[3]!.x - c[1]!.x) + (c[3]!.y - c[1]!.y) * (c[3]!.y - c[1]!.y))
    const d32 = sqrt((c[2]!.x - c[3]!.x) * (c[2]!.x - c[3]!.x) + (c[2]!.y - c[3]!.y) * (c[2]!.y - c[3]!.y))
    const d20 = sqrt((c[0]!.x - c[2]!.x) * (c[0]!.x - c[2]!.x) + (c[0]!.y - c[2]!.y) * (c[0]!.y - c[2]!.y))
    let lMin = d01
    lMin = min(lMin, d13)
    lMin = min(lMin, d32)
    lMin = min(lMin, d20)
    const minVote = max(d.u32(2), d.u32(round(d.f32(DECODE_MIN_VOTE_FRACTION_OF_QUAD_EDGE) * lMin)))

    let valid = d.u32(0)
    if (diff > d.f32(0.05) && whiteBound > blackBound) {
      valid = d.u32(1)
    }

    minmaxLayout.$.thresholds[quadId] = TagDecodeThresholdGpu({
      blackBound,
      whiteBound,
      minVoteTotal: minVote,
      valid,
    })
  })

  const pipeline = root.createComputePipeline({ compute: kernel }).$name('profile-minmax')

  const bindGroup = root.createBindGroup(minmaxLayout, {
    profileBuckets: profileBuckets as never,
    profileAvg: profileAvg as never,
    quads: quadDataBuffer as never,
    thresholds: thresholdBuf,
    activeQuadCount: quadCountBuf as never,
  })

  function encodeMinMax(pass: GPUComputePassEncoder) {
    pipeline.with(pass).with(bindGroup).dispatchWorkgroups(Math.ceil(MAX_QUADS / MINMAX_WG))
  }

  return { thresholdBuf, encodeMinMax }
}

// ---- main stage ----

export function createEdgeProfileStage(
  root: TgpuRoot,
  width: number,
  height: number,
  maxFlatEdges: number,
  grayTexView: unknown,
  lineOut: EdgeLineOutBuffer,
  quadDataBuffer: GridVizQuadBuffer,
  quadCountBuf: unknown,
  presentationFormat: GPUTextureFormat,
) {
  const bucketCount = maxFlatEdges * BUCKETS_PER_EDGE
  const profileBuckets = root.createBuffer(d.arrayOf(ProfileBucketAtomic, bucketCount)).$usage('storage')
  const profileAvg = root.createBuffer(d.arrayOf(d.f32, bucketCount)).$usage('storage')

  const { resetLayout, pipeline: resetPipeline } = createProfileResetPipeline(root, bucketCount)
  const { encodeAccum } = createProfileAccumStage(
    root, grayTexView, lineOut, profileBuckets, width, height, maxFlatEdges, presentationFormat,
  )
  const { normalizeLayout, pipeline: normalizePipeline } = createProfileNormalizePipeline(root, bucketCount)
  const { thresholdBuf, encodeMinMax } = createProfileMinMaxStage(
    root, profileBuckets, profileAvg, quadDataBuffer, quadCountBuf,
  )

  const resetBindGroup = root.createBindGroup(resetLayout, { profileBuckets, profileAvg })
  const normalizeBindGroup = root.createBindGroup(normalizeLayout, { profileBuckets, profileAvg })

  const bucketWg = Math.ceil(bucketCount / BUCKET_WORKGROUP_SIZE)

  const encodeCompute = (enc: GPUCommandEncoder) => {
    // reset
    runStage(enc, 'profile-reset', (p) => {
      resetPipeline.with(p).with(resetBindGroup).dispatchWorkgroups(bucketWg)
    })
    // accum (render pass)
    encodeAccum(enc)
    // normalize
    runStage(enc, 'profile-normalize', (p) => {
      normalizePipeline.with(p).with(normalizeBindGroup).dispatchWorkgroups(bucketWg)
    })
  }

  return {
    profileAvg,
    profileBuckets,
    thresholdBuf,
    encodeCompute,
    encodeMinMaxPass: encodeMinMax,
  }
}

function runStage(
  enc: GPUCommandEncoder,
  label: string,
  dispatch: (pass: GPUComputePassEncoder) => void,
) {
  const pass = profileComputePass(enc, label, { label })
  dispatch(pass)
  pass.end()
}

export type ProfileAvgBuffer = ReturnType<typeof createEdgeProfileStage>['profileAvg']
export type ProfileBucketsBuffer = ReturnType<typeof createEdgeProfileStage>['profileBuckets']
