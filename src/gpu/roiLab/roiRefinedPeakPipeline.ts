import type { SampledFlag, StorageFlag, TgpuBuffer, TgpuRoot, TgpuSampler, TgpuTexture } from 'typegpu'
import { d, std, tgpu } from 'typegpu'
import { abs, atomicAdd, atomicLoad, atomicStore, clamp, min, sqrt } from 'typegpu/std'

import { ROI_CIRCLE_VERTEX_COUNT } from '@/gpu/roiLab/roiCircleConstants'
import {
  ROI_PEAK_MAX_CANDIDATES,
  ROI_PEAK_MAX_CIRCLES,
  ROI_REFINED_PEAK_RECORD_BYTES,
  ROI_REJECT_FIT,
  ROI_REJECT_INDEFINITE,
  RoiCircleInstance,
  RoiCircleSchema,
  RoiDrawIndirect,
  RoiPeakCandidateGpu,
  RoiPeakCandidateSchema,
  RoiRefinedPeakGpu,
  RoiRefinedPeakSchema,
  type RoiRefinedPeakReadback,
} from '@/gpu/roiLab/roiPeakTypes'
import {
  CUBIC_FIT_COEFFICIENTS,
  GAUSSIAN_BLUR_RADIUS,
  GAUSSIAN_BLUR_WEIGHTS,
  HESSIAN_EPSILON,
  NEWTON_DET_EPSILON,
  NEWTON_MAX_ITERATIONS,
  NEWTON_STEP_EPSILON,
  NEWTON_VERTEX_EPSILON,
} from '@/lib/roiPeakFit'

const GaussianWeights = tgpu.const(d.arrayOf(d.f32, GAUSSIAN_BLUR_WEIGHTS.length), [...GAUSSIAN_BLUR_WEIGHTS])
const CubicFitCoefficients = tgpu.const(d.arrayOf(d.f32, CUBIC_FIT_COEFFICIENTS.length), [...CUBIC_FIT_COEFFICIENTS])

const detectionParams = d.struct({
  blackCut: d.u32,
  whiteCut: d.u32,
  valid: d.u32,
})

const COUNTER_CANDIDATES = 0
const COUNTER_REFINED = 1
const COUNTER_CIRCLES = 2
const COUNTER_REJECTS = 3
const COUNTER_COUNT = 4
const RoiCounterSchema = d.arrayOf(d.atomic(d.u32), COUNTER_COUNT)

const horizontalBlurLayout = tgpu.bindGroupLayout({
  cameraTexture: { texture: d.texture2d() },
  backgroundTexture: { texture: d.texture2d() },
  sampler: { sampler: 'filtering' },
  horizontal: { storage: d.arrayOf(d.f32), access: 'mutable' },
})

const verticalBlurLayout = tgpu.bindGroupLayout({
  horizontal: { storage: d.arrayOf(d.f32), access: 'readonly' },
  blurred: { storage: d.arrayOf(d.f32), access: 'mutable' },
})

const candidateLayout = tgpu.bindGroupLayout({
  cameraTexture: { texture: d.texture2d() },
  backgroundTexture: { texture: d.texture2d() },
  maskTexture: { texture: d.texture2d() },
  sampler: { sampler: 'filtering' },
  blurred: { storage: d.arrayOf(d.f32), access: 'readonly' },
  candidates: { storage: RoiPeakCandidateSchema, access: 'mutable' },
  counters: { storage: RoiCounterSchema, access: 'mutable' },
  params: { uniform: detectionParams },
})

const fitLayout = tgpu.bindGroupLayout({
  blurred: { storage: d.arrayOf(d.f32), access: 'readonly' },
  candidates: { storage: RoiPeakCandidateSchema, access: 'readonly' },
  counters: { storage: RoiCounterSchema, access: 'mutable' },
  refined: { storage: RoiRefinedPeakSchema, access: 'mutable' },
  circles: { storage: RoiCircleSchema, access: 'mutable' },
  rejects: { storage: RoiCircleSchema, access: 'mutable' },
})

const publishLayout = tgpu.bindGroupLayout({
  counters: { storage: RoiCounterSchema, access: 'mutable' },
  drawIndirect: { storage: RoiDrawIndirect, access: 'mutable' },
  rejectDrawIndirect: { storage: RoiDrawIndirect, access: 'mutable' },
})

const flattenedAt = tgpu.fn(
  [d.u32, d.u32, d.f32, d.f32],
  d.f32,
)((px, py, width, height) => {
  'use gpu'
  const color = std.textureLoad(horizontalBlurLayout.$.cameraTexture, d.vec2u(px, py), d.i32(0))
  const luma = std.dot(color.rgb, d.vec3f(0.299, 0.587, 0.114))
  const uv = (d.vec2f(d.f32(px), d.f32(py)) + d.vec2f(0.5, 0.5)).div(d.vec2f(width, height))
  const background = std.textureSampleLevel(
    horizontalBlurLayout.$.backgroundTexture,
    horizontalBlurLayout.$.sampler,
    uv,
    d.f32(0),
  ).r
  return std.clamp(d.f32(0.5) + (luma - background), d.f32(0), d.f32(1))
})

const sampledMask = tgpu.fn(
  [d.u32, d.u32, d.f32, d.f32],
  d.f32,
)((px, py, width, height) => {
  'use gpu'
  const uv = (d.vec2f(d.f32(px), d.f32(py)) + d.vec2f(0.5, 0.5)).div(d.vec2f(width, height))
  return std.textureSampleLevel(candidateLayout.$.maskTexture, candidateLayout.$.sampler, uv, d.f32(0)).r
})

/**
 * Raw high-pass value, matching the ROI Lab "Blobs" view. Candidate membership
 * is decided on this signal; the blurred buffer is only used for the 3x3
 * extremum and the cubic fit.
 */
const rawFlattenedAt = tgpu.fn(
  [d.u32, d.u32, d.f32, d.f32],
  d.f32,
)((px, py, width, height) => {
  'use gpu'
  const color = std.textureLoad(candidateLayout.$.cameraTexture, d.vec2u(px, py), d.i32(0))
  const luma = std.dot(color.rgb, d.vec3f(0.299, 0.587, 0.114))
  const uv = (d.vec2f(d.f32(px), d.f32(py)) + d.vec2f(0.5, 0.5)).div(d.vec2f(width, height))
  const background = std.textureSampleLevel(
    candidateLayout.$.backgroundTexture,
    candidateLayout.$.sampler,
    uv,
    d.f32(0),
  ).r
  return std.clamp(d.f32(0.5) + (luma - background), d.f32(0), d.f32(1))
})

const supportAt = tgpu.fn(
  [d.u32, d.u32, d.u32, d.f32, d.f32],
  d.bool,
)((px, py, polarity, width, height) => {
  'use gpu'
  const params = candidateLayout.$.params
  if (params.valid === d.u32(0)) {
    return false
  }
  if (sampledMask(px, py, width, height) < d.f32(0.5)) {
    return false
  }
  const value = rawFlattenedAt(px, py, width, height)
  const bin = d.u32(clamp(value, d.f32(0), d.f32(1)) * d.f32(255) + d.f32(0.5))
  if (polarity === d.u32(0)) {
    return bin < params.blackCut
  }
  return bin >= params.whiteCut
})

const plateauBlocked = tgpu.fn(
  [d.u32, d.u32, d.f32, d.u32, d.f32, d.f32],
  d.bool,
)((px, py, value, polarity, width, height) => {
  'use gpu'
  const left = d.i32(px) - d.i32(1)
  const up = d.i32(py) - d.i32(1)
  const right = d.i32(px) + d.i32(1)
  const maxX = d.i32(width) - d.i32(1)
  if (up >= d.i32(0)) {
    const upRow = d.u32(up) * d.u32(width)
    if (left >= d.i32(0)) {
      if (supportAt(d.u32(left), d.u32(up), polarity, width, height)) {
        if (abs(candidateLayout.$.blurred[upRow + d.u32(left)]! - value) <= d.f32(1e-7)) {
          return true
        }
      }
    }
    if (supportAt(px, d.u32(up), polarity, width, height)) {
      if (abs(candidateLayout.$.blurred[upRow + px]! - value) <= d.f32(1e-7)) {
        return true
      }
    }
    if (right <= maxX) {
      if (supportAt(d.u32(right), d.u32(up), polarity, width, height)) {
        if (abs(candidateLayout.$.blurred[upRow + d.u32(right)]! - value) <= d.f32(1e-7)) {
          return true
        }
      }
    }
  }
  if (left >= d.i32(0)) {
    if (supportAt(d.u32(left), py, polarity, width, height)) {
      if (abs(candidateLayout.$.blurred[py * d.u32(width) + d.u32(left)]! - value) <= d.f32(1e-7)) {
        return true
      }
    }
  }
  return false
})

export type RoiRefinedPeakStage = {
  encode: () => void
  /**
   * Queues the counter/peak copy and starts mapping it.
   *
   * The returned promise resolves once the copy has run and the records are on
   * the CPU. Callers queue the copy on one frame and await the promise on the
   * next: `mapAsync` only resolves after every pass queued before the copy, so
   * awaiting it on the frame that queued it stalls the CPU for the whole GPU
   * pipeline.
   */
  submitPeaksReadback: () => Promise<RoiRefinedPeakReadback>
  setThresholds: (blackCut: number, whiteCut: number, valid: boolean) => void
  blurredBuffer: TgpuBuffer<d.WgslArray<d.F32>> & StorageFlag
  circleBuffer: ReturnType<TgpuRoot['createBuffer']>
  drawIndirectBuffer: ReturnType<TgpuRoot['createBuffer']>
  rejectBuffer: ReturnType<TgpuRoot['createBuffer']>
  rejectDrawIndirectBuffer: ReturnType<TgpuRoot['createBuffer']>
  destroy: () => void
}

export function createRoiRefinedPeakStage(
  root: TgpuRoot,
  frameWidth: number,
  frameHeight: number,
  cameraTexture: TgpuTexture & SampledFlag,
  maskTexture: TgpuTexture & SampledFlag,
  backgroundTexture: TgpuTexture & SampledFlag,
  sampler: TgpuSampler,
): RoiRefinedPeakStage {
  const area = frameWidth * frameHeight
  const horizontalBuffer = root.createBuffer(d.arrayOf(d.f32, area)).$usage('storage')
  const blurredBuffer = root.createBuffer(d.arrayOf(d.f32, area)).$usage('storage')
  const candidateBuffer = root.createBuffer(RoiPeakCandidateSchema).$usage('storage')
  const refinedBuffer = root.createBuffer(RoiRefinedPeakSchema).$usage('storage')
  const circleBuffer = root.createBuffer(RoiCircleSchema).$usage('storage')
  const rejectBuffer = root.createBuffer(RoiCircleSchema).$usage('storage')
  const counterBuffer = root.createBuffer(RoiCounterSchema).$usage('storage')
  const drawIndirectBuffer = root.createBuffer(RoiDrawIndirect).$usage('storage', 'indirect')
  const rejectDrawIndirectBuffer = root.createBuffer(RoiDrawIndirect).$usage('storage', 'indirect')
  drawIndirectBuffer.write({
    vertexCount: ROI_CIRCLE_VERTEX_COUNT,
    instanceCount: 0,
    firstVertex: 0,
    firstInstance: 0,
  })
  rejectDrawIndirectBuffer.write({
    vertexCount: ROI_CIRCLE_VERTEX_COUNT,
    instanceCount: 0,
    firstVertex: 0,
    firstInstance: 0,
  })
  const readbackBuffer = root.device.createBuffer({
    label: 'roi-refined-peak-readback',
    size: COUNTER_COUNT * 4 + ROI_PEAK_MAX_CANDIDATES * ROI_REFINED_PEAK_RECORD_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  const detectionUniform = root.createUniform(detectionParams, {
    blackCut: 0,
    whiteCut: 256,
    valid: 0,
  })

  const horizontalPipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    'use gpu'
    const px = d.u32(x)
    const py = d.u32(y)
    let sum = d.f32(0)
    for (let offset = -GAUSSIAN_BLUR_RADIUS; offset <= GAUSSIAN_BLUR_RADIUS; offset += 1) {
      let sampleX = d.i32(px) + d.i32(offset)
      if (sampleX < d.i32(0)) {
        sampleX = -sampleX
      }
      if (sampleX >= d.i32(frameWidth)) {
        sampleX = d.i32(2 * (frameWidth - 1)) - sampleX
      }
      sum +=
        flattenedAt(d.u32(sampleX), py, d.f32(frameWidth), d.f32(frameHeight)) *
        GaussianWeights.$[d.u32(offset + GAUSSIAN_BLUR_RADIUS)]!
    }
    horizontalBlurLayout.$.horizontal[py * d.u32(frameWidth) + px] = sum
  })

  const verticalPipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    'use gpu'
    const px = d.u32(x)
    const py = d.u32(y)
    let sum = d.f32(0)
    for (let offset = -GAUSSIAN_BLUR_RADIUS; offset <= GAUSSIAN_BLUR_RADIUS; offset += 1) {
      let sampleY = d.i32(py) + d.i32(offset)
      if (sampleY < d.i32(0)) {
        sampleY = -sampleY
      }
      if (sampleY >= d.i32(frameHeight)) {
        sampleY = d.i32(2 * (frameHeight - 1)) - sampleY
      }
      sum +=
        verticalBlurLayout.$.horizontal[d.u32(sampleY) * d.u32(frameWidth) + px]! *
        GaussianWeights.$[d.u32(offset + GAUSSIAN_BLUR_RADIUS)]!
    }
    verticalBlurLayout.$.blurred[py * d.u32(frameWidth) + px] = sum
  })

  const candidatePipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    'use gpu'
    const params = candidateLayout.$.params
    if (params.valid === d.u32(0)) {
      return
    }
    const px = d.u32(x)
    const py = d.u32(y)
    if (sampledMask(px, py, d.f32(frameWidth), d.f32(frameHeight)) < d.f32(0.5)) {
      return
    }
    // Blob membership is decided on the raw high-pass signal, exactly like the
    // "Blobs" view. Do not re-test the blurred value here: blurring can move a
    // small blob's peak below the luma cut even though it is a valid blob.
    const rawValue = rawFlattenedAt(px, py, d.f32(frameWidth), d.f32(frameHeight))
    const rawBin = d.u32(clamp(rawValue, d.f32(0), d.f32(1)) * d.f32(255) + d.f32(0.5))
    let polarity = d.u32(0)
    if (rawBin < params.blackCut) {
      polarity = d.u32(0)
    } else if (rawBin >= params.whiteCut) {
      polarity = d.u32(1)
    } else {
      return
    }
    const value = candidateLayout.$.blurred[py * d.u32(frameWidth) + px]!
    for (let oy = -1; oy <= 1; oy += 1) {
      for (let ox = -1; ox <= 1; ox += 1) {
        if (ox === 0 && oy === 0) {
          continue
        }
        const sx = d.i32(px) + d.i32(ox)
        const sy = d.i32(py) + d.i32(oy)
        if (sx < d.i32(0) || sx >= d.i32(frameWidth) || sy < d.i32(0) || sy >= d.i32(frameHeight)) {
          continue
        }
        const neighbour = candidateLayout.$.blurred[d.u32(sy) * d.u32(frameWidth) + d.u32(sx)]!
        if (polarity === d.u32(1)) {
          if (neighbour > value + d.f32(1e-7)) {
            return
          }
        } else if (neighbour < value - d.f32(1e-7)) {
          return
        }
      }
    }
    if (plateauBlocked(px, py, value, polarity, d.f32(frameWidth), d.f32(frameHeight))) {
      return
    }
    const slot = atomicAdd(candidateLayout.$.counters[d.u32(COUNTER_CANDIDATES)]!, d.u32(1))
    if (slot < d.u32(ROI_PEAK_MAX_CANDIDATES)) {
      candidateLayout.$.candidates[slot] = RoiPeakCandidateGpu({
        x: px,
        y: py,
        polarity,
      })
    }
  })

  const fitPipeline = root.createGuardedComputePipeline((index: number) => {
    'use gpu'
    const slot = d.u32(index)
    if (slot >= atomicLoad(fitLayout.$.counters[d.u32(COUNTER_CANDIDATES)]!)) {
      return
    }
    const candidate = fitLayout.$.candidates[slot]!
    const values = d.arrayOf(d.f32, 25)()
    for (let dy = -2; dy <= 2; dy += 1) {
      for (let dx = -2; dx <= 2; dx += 1) {
        let sx = d.i32(candidate.x) + d.i32(dx)
        let sy = d.i32(candidate.y) + d.i32(dy)
        if (sx < d.i32(0)) {
          sx = d.i32(0)
        }
        if (sx >= d.i32(frameWidth)) {
          sx = d.i32(frameWidth) - d.i32(1)
        }
        if (sy < d.i32(0)) {
          sy = d.i32(0)
        }
        if (sy >= d.i32(frameHeight)) {
          sy = d.i32(frameHeight) - d.i32(1)
        }
        values[d.u32((dy + 2) * 5 + (dx + 2))] = fitLayout.$.blurred[d.u32(sy) * d.u32(frameWidth) + d.u32(sx)]!
      }
    }
    const coefficients = d.arrayOf(d.f32, 10)()
    for (let term = 0; term < 10; term += 1) {
      let sum = d.f32(0)
      for (let sample = 0; sample < 25; sample += 1) {
        sum += CubicFitCoefficients.$[d.u32(term * 25 + sample)]! * values[d.u32(sample)]!
      }
      coefficients[d.u32(term)] = sum
    }
    const c0 = coefficients[d.u32(0)]!
    const c1 = coefficients[d.u32(1)]!
    const c2 = coefficients[d.u32(2)]!
    const c3 = coefficients[d.u32(3)]!
    const c4 = coefficients[d.u32(4)]!
    const c5 = coefficients[d.u32(5)]!
    const c6 = coefficients[d.u32(6)]!
    const c7 = coefficients[d.u32(7)]!
    const c8 = coefficients[d.u32(8)]!
    const c9 = coefficients[d.u32(9)]!

    let u = d.f32(0)
    let v = d.f32(0)
    let converged = false
    for (let iteration = 0; iteration < NEWTON_MAX_ITERATIONS; iteration += 1) {
      const gx = c1 + d.f32(2) * c3 * u + c4 * v + d.f32(3) * c6 * u * u + d.f32(2) * c7 * u * v + c8 * v * v
      const gy = c2 + c4 * u + d.f32(2) * c5 * v + c7 * u * u + d.f32(2) * c8 * u * v + d.f32(3) * c9 * v * v
      const hxx = d.f32(2) * c3 + d.f32(6) * c6 * u + d.f32(2) * c7 * v
      const hxy = c4 + d.f32(2) * c7 * u + d.f32(2) * c8 * v
      const hyy = d.f32(2) * c5 + d.f32(2) * c8 * u + d.f32(6) * c9 * v
      const det = hxx * hyy - hxy * hxy
      if (!(abs(det) >= d.f32(NEWTON_DET_EPSILON))) {
        break
      }
      const stepU = (-hyy * gx + hxy * gy) / det
      const stepV = (hxy * gx - hxx * gy) / det
      const stepLength = sqrt(stepU * stepU + stepV * stepV)
      let scale = d.f32(1)
      if (stepLength > d.f32(0.5)) {
        scale = d.f32(0.5) / stepLength
      }
      u += stepU * scale
      v += stepV * scale
      if (stepLength * scale < d.f32(NEWTON_STEP_EPSILON)) {
        converged = true
        break
      }
    }

    let refinedX = d.f32(candidate.x) + d.f32(0.5)
    let refinedY = d.f32(candidate.y) + d.f32(0.5)
    let peak = d.f32(0)
    let lambda1 = d.f32(0)
    let lambda2 = d.f32(0)
    let hessianXY = d.f32(0)
    let reason = d.u32(0)
    const vertexOutside = abs(u) > d.f32(1 + NEWTON_VERTEX_EPSILON) || abs(v) > d.f32(1 + NEWTON_VERTEX_EPSILON)
    if (!converged || vertexOutside) {
      reason = d.u32(ROI_REJECT_FIT)
    } else {
      refinedX = d.f32(candidate.x) + d.f32(2) * u + d.f32(0.5)
      refinedY = d.f32(candidate.y) + d.f32(2) * v + d.f32(0.5)
      const hxx = d.f32(2) * c3 + d.f32(6) * c6 * u + d.f32(2) * c7 * v
      const hxy = c4 + d.f32(2) * c7 * u + d.f32(2) * c8 * v
      const hyy = d.f32(2) * c5 + d.f32(2) * c8 * u + d.f32(6) * c9 * v
      const physicalXX = hxx / d.f32(4)
      const physicalXY = hxy / d.f32(4)
      const physicalYY = hyy / d.f32(4)
      const mean = (physicalXX + physicalYY) / d.f32(2)
      const difference = (physicalXX - physicalYY) / d.f32(2)
      const rootTerm = sqrt(difference * difference + physicalXY * physicalXY)
      lambda1 = mean - rootTerm
      lambda2 = mean + rootTerm
      hessianXY = physicalXY
      peak =
        c0 +
        c1 * u +
        c2 * v +
        c3 * u * u +
        c4 * u * v +
        c5 * v * v +
        c6 * u * u * u +
        c7 * u * u * v +
        c8 * u * v * v +
        c9 * v * v * v
      const finite =
        peak === peak &&
        lambda1 === lambda1 &&
        lambda2 === lambda2 &&
        hessianXY === hessianXY &&
        abs(peak) < d.f32(1e20) &&
        abs(lambda1) < d.f32(1e20) &&
        abs(lambda2) < d.f32(1e20) &&
        abs(hessianXY) < d.f32(1e20)
      if (!finite) {
        reason = d.u32(ROI_REJECT_FIT)
      } else if (candidate.polarity === d.u32(1)) {
        if (!(lambda1 < d.f32(-HESSIAN_EPSILON) && lambda2 < d.f32(-HESSIAN_EPSILON))) {
          reason = d.u32(ROI_REJECT_INDEFINITE)
        }
      } else if (!(lambda1 > d.f32(HESSIAN_EPSILON) && lambda2 > d.f32(HESSIAN_EPSILON))) {
        reason = d.u32(ROI_REJECT_INDEFINITE)
      }
    }

    if (reason === d.u32(0)) {
      const refinedSlot = atomicAdd(fitLayout.$.counters[d.u32(COUNTER_REFINED)]!, d.u32(1))
      if (refinedSlot < d.u32(ROI_PEAK_MAX_CANDIDATES)) {
        fitLayout.$.refined[refinedSlot] = RoiRefinedPeakGpu({
          x: refinedX,
          y: refinedY,
          peak,
          lambda1,
          lambda2,
          hxy: hessianXY,
          polarity: candidate.polarity,
          pad0: d.u32(0),
        })
      }
      const circleSlot = atomicAdd(fitLayout.$.counters[d.u32(COUNTER_CIRCLES)]!, d.u32(1))
      if (circleSlot < d.u32(ROI_PEAK_MAX_CIRCLES)) {
        fitLayout.$.circles[circleSlot] = RoiCircleInstance({
          x: refinedX,
          y: refinedY,
          polarity: candidate.polarity,
          reason: d.u32(0),
        })
      }
    } else {
      const rejectSlot = atomicAdd(fitLayout.$.counters[d.u32(COUNTER_REJECTS)]!, d.u32(1))
      if (rejectSlot < d.u32(ROI_PEAK_MAX_CIRCLES)) {
        fitLayout.$.rejects[rejectSlot] = RoiCircleInstance({
          x: refinedX,
          y: refinedY,
          polarity: candidate.polarity,
          reason,
        })
      }
    }
  })

  const resetPipeline = root.createGuardedComputePipeline(() => {
    'use gpu'
    atomicStore(fitLayout.$.counters[d.u32(COUNTER_CANDIDATES)]!, d.u32(0))
    atomicStore(fitLayout.$.counters[d.u32(COUNTER_REFINED)]!, d.u32(0))
    atomicStore(fitLayout.$.counters[d.u32(COUNTER_CIRCLES)]!, d.u32(0))
    atomicStore(fitLayout.$.counters[d.u32(COUNTER_REJECTS)]!, d.u32(0))
  })

  const publishPipeline = root.createGuardedComputePipeline(() => {
    'use gpu'
    publishLayout.$.drawIndirect.instanceCount = min(
      atomicLoad(publishLayout.$.counters[d.u32(COUNTER_CIRCLES)]!),
      d.u32(ROI_PEAK_MAX_CIRCLES),
    )
    publishLayout.$.rejectDrawIndirect.instanceCount = min(
      atomicLoad(publishLayout.$.counters[d.u32(COUNTER_REJECTS)]!),
      d.u32(ROI_PEAK_MAX_CIRCLES),
    )
  })

  const horizontalBindGroup = root.createBindGroup(horizontalBlurLayout, {
    cameraTexture,
    backgroundTexture,
    sampler,
    horizontal: horizontalBuffer,
  })
  const verticalBindGroup = root.createBindGroup(verticalBlurLayout, {
    horizontal: horizontalBuffer,
    blurred: blurredBuffer,
  })
  const candidateBindGroup = root.createBindGroup(candidateLayout, {
    cameraTexture,
    backgroundTexture,
    maskTexture,
    sampler,
    blurred: blurredBuffer,
    candidates: candidateBuffer,
    counters: counterBuffer,
    params: detectionUniform,
  })
  const fitBindGroup = root.createBindGroup(fitLayout, {
    blurred: blurredBuffer,
    candidates: candidateBuffer,
    counters: counterBuffer,
    refined: refinedBuffer,
    circles: circleBuffer,
    rejects: rejectBuffer,
  })
  const publishBindGroup = root.createBindGroup(publishLayout, {
    counters: counterBuffer,
    drawIndirect: drawIndirectBuffer,
    rejectDrawIndirect: rejectDrawIndirectBuffer,
  })

  let readbackBusy = false
  let readbackPending = false

  const headerBytes = COUNTER_COUNT * 4
  const peakBytes = ROI_PEAK_MAX_CANDIDATES * ROI_REFINED_PEAK_RECORD_BYTES
  const bytesCopied = headerBytes + peakBytes

  const emptyReadback = (): RoiRefinedPeakReadback => ({
    peaks: [],
    bytesCopied: 0,
    truncated: false,
  })

  const mapReadback = async (): Promise<RoiRefinedPeakReadback> => {
    try {
      await readbackBuffer.mapAsync(GPUMapMode.READ, 0, bytesCopied)
      const view = new DataView(readbackBuffer.getMappedRange(0, bytesCopied))
      const rawCandidates = view.getUint32(COUNTER_CANDIDATES * 4, true)
      const rawRefined = view.getUint32(COUNTER_REFINED * 4, true)
      const count = Math.min(rawRefined, ROI_PEAK_MAX_CANDIDATES)
      const peaks = []
      for (let index = 0; index < count; index += 1) {
        const offset = headerBytes + index * ROI_REFINED_PEAK_RECORD_BYTES
        peaks.push({
          x: view.getFloat32(offset, true),
          y: view.getFloat32(offset + 4, true),
          peak: view.getFloat32(offset + 8, true),
          eigenvalue1: view.getFloat32(offset + 12, true),
          eigenvalue2: view.getFloat32(offset + 16, true),
          hessianXY: view.getFloat32(offset + 20, true),
          polarity: (view.getUint32(offset + 24, true) === 0 ? 0 : 1) as 0 | 1,
        })
      }
      peaks.sort((first, second) => first.y - second.y || first.x - second.x)
      return {
        peaks,
        bytesCopied,
        truncated: rawRefined > ROI_PEAK_MAX_CANDIDATES || rawCandidates > ROI_PEAK_MAX_CANDIDATES,
      }
    } finally {
      if (readbackBuffer.mapState === 'mapped') {
        readbackBuffer.unmap()
      }
    }
  }

  return {
    blurredBuffer,
    circleBuffer,
    drawIndirectBuffer,
    rejectBuffer,
    rejectDrawIndirectBuffer,
    setThresholds(blackCut, whiteCut, valid) {
      detectionUniform.write({
        blackCut,
        whiteCut,
        valid: valid ? 1 : 0,
      })
    },
    encode() {
      resetPipeline.with(fitBindGroup).dispatchThreads()
      horizontalPipeline.with(horizontalBindGroup).dispatchThreads(frameWidth, frameHeight)
      verticalPipeline.with(verticalBindGroup).dispatchThreads(frameWidth, frameHeight)
      candidatePipeline.with(candidateBindGroup).dispatchThreads(frameWidth, frameHeight)
      fitPipeline.with(fitBindGroup).dispatchThreads(ROI_PEAK_MAX_CANDIDATES)
      publishPipeline.with(publishBindGroup).dispatchThreads()
    },
    submitPeaksReadback() {
      // One readback per stage at a time. Slots alternate, so a stage is only
      // touched every other frame; a conflict here means a caller awaited too
      // late and the previous map is still outstanding.
      if (readbackBusy || readbackPending) {
        return Promise.resolve(emptyReadback())
      }
      readbackPending = true
      readbackBusy = true
      const encoder = root.device.createCommandEncoder({ label: 'roi-refined-peak-readback' })
      encoder.copyBufferToBuffer(root.unwrap(counterBuffer), 0, readbackBuffer, 0, headerBytes)
      encoder.copyBufferToBuffer(root.unwrap(refinedBuffer), 0, readbackBuffer, headerBytes, peakBytes)
      root.device.queue.submit([encoder.finish()])
      return mapReadback().finally(() => {
        readbackPending = false
        readbackBusy = false
      })
    },
    destroy() {
      horizontalBuffer.destroy()
      blurredBuffer.destroy()
      candidateBuffer.destroy()
      refinedBuffer.destroy()
      circleBuffer.destroy()
      rejectBuffer.destroy()
      counterBuffer.destroy()
      drawIndirectBuffer.destroy()
      rejectDrawIndirectBuffer.destroy()
      readbackBuffer.destroy()
    },
  }
}
