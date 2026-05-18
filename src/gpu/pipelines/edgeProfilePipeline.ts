// Per-label 64-bin grayscale profiles along edge normal (black → white).
import type { TgpuRoot } from 'typegpu'
import { tgpu, d, std } from 'typegpu'
import { atomicAdd, atomicLoad, atomicStore, length, select, sqrt } from 'typegpu/std'

import { COMPONENT_LABEL_INVALID } from '@/gpu/detectedQuad'
import type { CompactLabelMapBuffer } from '@/gpu/pipelines/compactLabelPipeline'
import type { EdgeFilterBindResources } from '@/gpu/pipelines/edgeFilterPipeline'
import {
  EdgeLineEntry,
  PROFILE_BUCKET_COUNT,
  PROFILE_NEIGHBORHOOD_HALF,
  type EdgeLineOutBuffer,
} from '@/gpu/pipelines/edgeLineFitPipeline'
import { lineDirDot } from '@/gpu/shaders/linePca'
import type { GrayTexToBufferBindResources } from '@/gpu/pipelines/grayPipeline'

export { PROFILE_BUCKET_COUNT, PROFILE_NEIGHBORHOOD_HALF } from '@/gpu/pipelines/edgeLineFitPipeline'

const WORKGROUP_SIZE = 16
/** 1D bucket reset/normalize: WG sized so MAX_FLAT_EDGES×64 buckets stay under 65535 workgroups/dim. */
const BUCKET_WORKGROUP_SIZE = 256
const BUCKETS_PER_LABEL = PROFILE_BUCKET_COUNT

const GRAY_FIXED_SCALE = 100000

const ProfileBucketAtomic = d.struct({
  sumGrayFixed: d.atomic(d.i32),
  count: d.atomic(d.u32),
})

/** Same layout as {@link ProfileBucketAtomic}; plain types for readonly vertex reads. */
export const ProfileBucketGpu = d.struct({
  sumGrayFixed: d.i32,
  count: d.u32,
})

const FrameSizeUniform = d.struct({
  width: d.u32,
  height: d.u32,
})

function createProfileLayouts() {
  const resetLayout = tgpu.bindGroupLayout({
    profileBuckets: { storage: d.arrayOf(ProfileBucketAtomic), access: 'mutable' },
    profileAvg: { storage: d.arrayOf(d.f32), access: 'mutable' },
  })
  const accumLayout = tgpu.bindGroupLayout({
    frame: { uniform: FrameSizeUniform },
    grayBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
    edgeBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
    compactLabels: { storage: d.arrayOf(d.u32), access: 'readonly' },
    lineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'readonly' },
    profileBuckets: { storage: d.arrayOf(ProfileBucketAtomic), access: 'mutable' },
  })
  const normalizeLayout = tgpu.bindGroupLayout({
    profileBuckets: { storage: d.arrayOf(ProfileBucketAtomic), access: 'mutable' },
    profileAvg: { storage: d.arrayOf(d.f32), access: 'mutable' },
  })
  return { resetLayout, accumLayout, normalizeLayout }
}

export function createEdgeProfileStage(
  root: TgpuRoot,
  width: number,
  height: number,
  maxComponents: number,
  grayBuffer: GrayTexToBufferBindResources['grayBuffer'],
  filteredBuffer: EdgeFilterBindResources['filteredBuffer'],
  compactLabels: CompactLabelMapBuffer,
  lineOut: EdgeLineOutBuffer,
) {
  const bucketCount = maxComponents * BUCKETS_PER_LABEL
  const profileBuckets = root.createBuffer(d.arrayOf(ProfileBucketAtomic, bucketCount)).$usage('storage')
  const profileAvg = root.createBuffer(d.arrayOf(d.f32, bucketCount)).$usage('storage')
  const frameUniform = root.createBuffer(FrameSizeUniform).$usage('uniform')

  const layouts = createProfileLayouts()
  const resetPipeline = createProfileResetPipeline(root, layouts.resetLayout, bucketCount)
  const accumPipeline = createProfileAccumPipeline(root, layouts.accumLayout, width, height)
  const normalizePipeline = createProfileNormalizePipeline(root, layouts.normalizeLayout, bucketCount)

  const resetBindGroup = root.createBindGroup(layouts.resetLayout, {
    profileBuckets,
    profileAvg,
  })
  const accumBindGroup = root.createBindGroup(layouts.accumLayout, {
    frame: frameUniform,
    grayBuffer,
    edgeBuffer: filteredBuffer,
    compactLabels,
    lineOut,
    profileBuckets,
  })
  const normalizeBindGroup = root.createBindGroup(layouts.normalizeLayout, {
    profileBuckets,
    profileAvg,
  })

  const wgX = Math.ceil(width / WORKGROUP_SIZE)
  const wgY = Math.ceil(height / WORKGROUP_SIZE)
  const bucketWg = Math.ceil(bucketCount / BUCKET_WORKGROUP_SIZE)

  const encodeCompute = (pass: GPUComputePassEncoder) => {
    frameUniform.write({ width: d.u32(width), height: d.u32(height) })
    resetPipeline.with(pass).with(resetBindGroup).dispatchWorkgroups(bucketWg)
    accumPipeline.with(pass).with(accumBindGroup).dispatchWorkgroups(wgX, wgY)
    normalizePipeline.with(pass).with(normalizeBindGroup).dispatchWorkgroups(bucketWg)
  }

  return {
    profileAvg,
    profileBuckets,
    encodeCompute,
  }
}

function createProfileResetPipeline(
  root: TgpuRoot,
  resetLayout: ReturnType<typeof createProfileLayouts>['resetLayout'],
  bucketCount: number,
) {
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
  return root.createComputePipeline({ compute: kernel })
}

function createProfileAccumPipeline(
  root: TgpuRoot,
  accumLayout: ReturnType<typeof createProfileLayouts>['accumLayout'],
  width: number,
  height: number,
) {
  const halfW = PROFILE_NEIGHBORHOOD_HALF
  const bucketCountF = d.f32(PROFILE_BUCKET_COUNT)
  const normSpan = d.f32(2) * d.f32(halfW)

  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, WORKGROUP_SIZE, 1],
  })((input) => {
    'use gpu'
    const x = d.i32(input.gid.x)
    const y = d.i32(input.gid.y)
    const fw = d.i32(accumLayout.$.frame.width)
    const fh = d.i32(accumLayout.$.frame.height)
    if (x >= fw || y >= fh) {
      return
    }

    const px = d.f32(x) + d.f32(0.5)
    const py = d.f32(y) + d.f32(0.5)

    let bestLabel = d.u32(COMPONENT_LABEL_INVALID)
    let bestAbsS = d.f32(1e30)
    let bestNDotMean = d.f32(0)
    let bestTSampleMin = d.f32(0)
    let bestTSampleMax = d.f32(0)
    let bestSumGx = d.f32(0)
    let bestSumGy = d.f32(0)

    for (const dy of tgpu.unroll(std.range(-2, 3))) {
      for (const dx of tgpu.unroll(std.range(-2, 3))) {
        const nx = x + dx
        const ny = y + dy
        if (nx >= d.i32(0) && nx < fw && ny >= d.i32(0) && ny < fh) {
          const nIdx = d.u32(ny * fw + nx)
          if (length(accumLayout.$.edgeBuffer[nIdx]!) > d.f32(0)) {
            const label = accumLayout.$.compactLabels[nIdx]!
            if (label !== d.u32(COMPONENT_LABEL_INVALID)) {
              const line = accumLayout.$.lineOut[label]!
              if (line.valid !== d.u32(0)) {
                const gLen = sqrt(line.sumGx * line.sumGx + line.sumGy * line.sumGy)
                if (gLen >= d.f32(1e-8)) {
                  const nnx = line.sumGx / gLen
                  const nny = line.sumGy / gLen
                  const s = px * nnx + py * nny - line.nDotMean
                  const absS = std.abs(s)
                  const pick =
                    bestLabel === d.u32(COMPONENT_LABEL_INVALID) ||
                    absS < bestAbsS ||
                    (absS === bestAbsS && label < bestLabel)
                  if (pick) {
                    bestLabel = label
                    bestAbsS = absS
                    bestSumGx = line.sumGx
                    bestSumGy = line.sumGy
                    bestNDotMean = line.nDotMean
                    bestTSampleMin = line.tSampleMin
                    bestTSampleMax = line.tSampleMax
                  }
                }
              }
            }
          }
        }
      }
    }

    if (bestLabel === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }

    const gLen = sqrt(bestSumGx * bestSumGx + bestSumGy * bestSumGy)
    const nnx = bestSumGx / gLen
    const nny = bestSumGy / gLen
    const s = px * nnx + py * nny - bestNDotMean
    if (std.abs(s) > d.f32(halfW)) {
      return
    }
    const t = lineDirDot(px, py, nnx, nny)
    if (t < bestTSampleMin || t > bestTSampleMax) {
      return
    }

    const bFloat = ((s + d.f32(halfW)) / normSpan) * bucketCountF
    let b = d.u32(std.floor(bFloat))
    b = std.min(std.max(b, d.u32(0)), d.u32(PROFILE_BUCKET_COUNT - 1))

    const gray = accumLayout.$.grayBuffer[d.u32(y * fw + x)]!
    const bucketIdx = bestLabel * d.u32(BUCKETS_PER_LABEL) + b
    const slot = accumLayout.$.profileBuckets[bucketIdx]!
    atomicAdd(slot.sumGrayFixed, d.i32(gray * d.f32(GRAY_FIXED_SCALE)))
    atomicAdd(slot.count, d.u32(1))
  })
  return root.createComputePipeline({ compute: kernel })
}

function createProfileNormalizePipeline(
  root: TgpuRoot,
  normalizeLayout: ReturnType<typeof createProfileLayouts>['normalizeLayout'],
  bucketCount: number,
) {
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
    const avg = select(d.f32(0), d.f32(sumFixed) / d.f32(c) / d.f32(GRAY_FIXED_SCALE), c > d.u32(0))
    normalizeLayout.$.profileAvg[bid] = avg
  })
  return root.createComputePipeline({ compute: kernel })
}

export type ProfileAvgBuffer = ReturnType<typeof createEdgeProfileStage>['profileAvg']
export type ProfileBucketsBuffer = ReturnType<typeof createEdgeProfileStage>['profileBuckets']
