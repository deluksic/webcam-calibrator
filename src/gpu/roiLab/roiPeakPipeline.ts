import type { SampledFlag, TgpuRoot, TgpuTexture } from 'typegpu'
import { d, std, tgpu } from 'typegpu'
import {
  abs,
  atomicAdd,
  atomicLoad,
  atomicStore,
  clamp,
  intdiv,
  min,
  sqrt,
} from 'typegpu/std'

import { ROI_CIRCLE_VERTEX_COUNT } from '@/gpu/roiLab/roiCircleConstants'

export const ROI_PEAK_MAX_CANDIDATES = 4096
export const ROI_PEAK_MIN_RESPONSE = 0.004
export const ROI_PEAK_BLUR_RADIUS = 4
export const ROI_PEAK_GAP_FRACTION = 0.07
export const ROI_PEAK_MAX_CIRCLES = 4096

export const ROI_REJECT_LOW_RESPONSE = 1
export const ROI_REJECT_FIT = 2
export const ROI_REJECT_NMS = 3
export const ROI_REJECT_LOW_AREA = 4
export const ROI_REJECT_GATE = 5

const BLUR_SIGMA = 2.5
const BLUR_WEIGHTS = (() => {
  const weights: number[] = []
  let sum = 0
  for (let i = -ROI_PEAK_BLUR_RADIUS; i <= ROI_PEAK_BLUR_RADIUS; i++) {
    const weight = Math.exp(-(i * i) / (2 * BLUR_SIGMA * BLUR_SIGMA))
    weights.push(weight)
    sum += weight
  }
  return weights.map((weight) => weight / sum)
})()

const BlurWeights = tgpu.const(d.arrayOf(d.f32, BLUR_WEIGHTS.length), BLUR_WEIGHTS)

const RoiPeakCandidateGpu = d.struct({
  x: d.u32,
  y: d.u32,
  polarity: d.u32,
  values: d.arrayOf(d.f32, 49),
})
const RoiCandidateSchema = d.arrayOf(RoiPeakCandidateGpu, ROI_PEAK_MAX_CANDIDATES)

const FittedPeak = d.struct({
  x: d.f32,
  y: d.f32,
  peak: d.f32,
  polarity: d.u32,
  suppressed: d.u32,
  support: d.u32,
})
const FittedSchema = d.arrayOf(FittedPeak, ROI_PEAK_MAX_CANDIDATES)

const DistanceStats = d.struct({
  meanDistance: d.f32,
  radius: d.f32,
})
const NearestSchema = d.arrayOf(d.f32, ROI_PEAK_MAX_CANDIDATES)

export const RoiCircleInstance = d.struct({
  x: d.f32,
  y: d.f32,
  polarity: d.u32,
  reason: d.u32,
})
export const RoiCircleSchema = d.arrayOf(RoiCircleInstance, ROI_PEAK_MAX_CIRCLES)

export const RoiDrawIndirect = d.struct({
  vertexCount: d.u32,
  instanceCount: d.u32,
  firstVertex: d.u32,
  firstInstance: d.u32,
})

const detectionParams = d.struct({
  blackCut: d.u32,
  whiteCut: d.u32,
  valid: d.u32,
  minResponse: d.f32,
  blackMinPeak: d.f32,
  whiteMinPeak: d.f32,
  minSupport: d.u32,
})

const candidateLayout = tgpu.bindGroupLayout({
  cameraTexture: { texture: d.texture2d() },
  blurred: { storage: d.arrayOf(d.f32), access: 'readonly' },
  maskTexture: { texture: d.texture2d() },
  candidates: { storage: RoiCandidateSchema, access: 'mutable' },
  candidateCount: { storage: d.atomic(d.u32), access: 'mutable' },
  rejects: { storage: RoiCircleSchema, access: 'mutable' },
  rejectCount: { storage: d.atomic(d.u32), access: 'mutable' },
  params: { uniform: detectionParams },
})

const fitLayout = tgpu.bindGroupLayout({
  candidates: { storage: RoiCandidateSchema, access: 'readonly' },
  fitted: { storage: FittedSchema, access: 'mutable' },
  candidateCount: { storage: d.atomic(d.u32), access: 'mutable' },
})

const suppressionLayout = tgpu.bindGroupLayout({
  fitted: { storage: FittedSchema, access: 'mutable' },
  candidateCount: { storage: d.atomic(d.u32), access: 'mutable' },
  stats: { storage: DistanceStats, access: 'readonly' },
})

const nearestLayout = tgpu.bindGroupLayout({
  fitted: { storage: FittedSchema, access: 'readonly' },
  candidateCount: { storage: d.atomic(d.u32), access: 'mutable' },
  nearest: { storage: NearestSchema, access: 'mutable' },
})

const statsLayout = tgpu.bindGroupLayout({
  nearest: { storage: NearestSchema, access: 'readonly' },
  candidateCount: { storage: d.atomic(d.u32), access: 'mutable' },
  stats: { storage: DistanceStats, access: 'mutable' },
})

const appendLayout = tgpu.bindGroupLayout({
  fitted: { storage: FittedSchema, access: 'readonly' },
  candidateCount: { storage: d.atomic(d.u32), access: 'mutable' },
  circles: { storage: RoiCircleSchema, access: 'mutable' },
  circleCount: { storage: d.atomic(d.u32), access: 'mutable' },
  rejects: { storage: RoiCircleSchema, access: 'mutable' },
  rejectCount: { storage: d.atomic(d.u32), access: 'mutable' },
  params: { uniform: detectionParams },
})

const publishLayout = tgpu.bindGroupLayout({
  circleCount: { storage: d.atomic(d.u32), access: 'mutable' },
  rejectCount: { storage: d.atomic(d.u32), access: 'mutable' },
  drawIndirect: { storage: RoiDrawIndirect, access: 'mutable' },
  rejectDrawIndirect: { storage: RoiDrawIndirect, access: 'mutable' },
})

const candidateKernel = (x: number, y: number) => {
  'use gpu'
  const size = std.textureDimensions(candidateLayout.$.cameraTexture)
  const maskSize = std.textureDimensions(candidateLayout.$.maskTexture)
  const px = d.u32(x)
  const py = d.u32(y)
  const params = candidateLayout.$.params
  if (params.valid === d.u32(0)) {
    return
  }
  let maskX = intdiv(px * maskSize.x, size.x)
  let maskY = intdiv(py * maskSize.y, size.y)
  if (maskX >= maskSize.x) {
    maskX = maskSize.x - d.u32(1)
  }
  if (maskY >= maskSize.y) {
    maskY = maskSize.y - d.u32(1)
  }
  const mask = std.textureLoad(
    candidateLayout.$.maskTexture,
    d.vec2u(maskX, maskY),
    d.i32(0),
  ).r
  if (mask < d.f32(0.5)) {
    return
  }
  const color = std.textureLoad(
    candidateLayout.$.cameraTexture,
    d.vec2u(px, py),
    d.i32(0),
  )
  const gray = std.dot(color.rgb, d.vec3f(0.299, 0.587, 0.114))
  const bin = d.u32(std.clamp(gray, d.f32(0), d.f32(1)) * d.f32(255) + d.f32(0.5))
  let polarity = d.u32(0)
  if (bin < params.blackCut) {
    polarity = d.u32(0)
  } else if (bin >= params.whiteCut) {
    polarity = d.u32(1)
  } else {
    return
  }

  const patch = d.arrayOf(d.f32, 49)()
  for (let oy = -3; oy <= 3; oy += 1) {
    for (let ox = -3; ox <= 3; ox += 1) {
      const sx = clamp(d.i32(px) + d.i32(ox), d.i32(0), d.i32(size.x) - d.i32(1))
      const sy = clamp(d.i32(py) + d.i32(oy), d.i32(0), d.i32(size.y) - d.i32(1))
      const sample = std.textureLoad(
        candidateLayout.$.cameraTexture,
        d.vec2u(d.u32(sx), d.u32(sy)),
        d.i32(0),
      )
      const sampleGray = std.dot(sample.rgb, d.vec3f(0.299, 0.587, 0.114))
      const blurred = candidateLayout.$.blurred[d.u32(sy) * size.x + d.u32(sx)]!
      const contrast = sampleGray - blurred
      patch[d.u32((oy + 3) * 7 + (ox + 3))] =
        polarity === d.u32(0) ? -contrast : contrast
    }
  }

  const center = patch[24]!
  let isMax = true
  for (let oy = -2; oy <= 2; oy += 1) {
    for (let ox = -2; ox <= 2; ox += 1) {
      const index = (oy + 3) * 7 + (ox + 3)
      if (index === 24) {
        continue
      }
      const value = patch[d.u32(index)]!
      if (value > center || (value === center && index < 24)) {
        isMax = false
      }
    }
  }
  if (!isMax) {
    return
  }
  if (center < params.minResponse) {
    if (atomicLoad(candidateLayout.$.rejectCount) < d.u32(1024)) {
      const debugIndex = atomicAdd(candidateLayout.$.rejectCount, d.u32(1))
      if (debugIndex < d.u32(1024)) {
        candidateLayout.$.rejects[debugIndex] = RoiCircleInstance({
          x: d.f32(px) + d.f32(0.5),
          y: d.f32(py) + d.f32(0.5),
          polarity,
          reason: d.u32(ROI_REJECT_LOW_RESPONSE),
        })
      }
    }
    return
  }

  const slotIndex = atomicAdd(candidateLayout.$.candidateCount, d.u32(1))
  if (slotIndex >= d.u32(ROI_PEAK_MAX_CANDIDATES)) {
    return
  }
  const slot = candidateLayout.$.candidates[slotIndex]!
  slot.x = px
  slot.y = py
  slot.polarity = polarity
  for (let oy = -3; oy <= 3; oy += 1) {
    for (let ox = -3; ox <= 3; ox += 1) {
      slot.values[d.u32((oy + 3) * 7 + (ox + 3))] =
        patch[d.u32((oy + 3) * 7 + (ox + 3))]!
    }
  }
}

const fitKernel = (index: number) => {
  'use gpu'
  const slot = d.u32(index)
  if (slot >= atomicLoad(fitLayout.$.candidateCount)) {
    return
  }
  const candidate = fitLayout.$.candidates[slot]!
  let s0 = d.f32(0)
  let sx = d.f32(0)
  let sy = d.f32(0)
  let sxx = d.f32(0)
  let syy = d.f32(0)
  let sxy = d.f32(0)
  for (let dy = -3; dy <= 3; dy += 1) {
    for (let dx = -3; dx <= 3; dx += 1) {
      const z = candidate.values[d.u32((dy + 3) * 7 + (dx + 3))]!
      s0 += z
      sx += d.f32(dx) * z
      sy += d.f32(dy) * z
      sxx += d.f32(dx * dx) * z
      syy += d.f32(dy * dy) * z
      sxy += d.f32(dx * dy) * z
    }
  }
  const b = sx / d.f32(196)
  const dd = sy / d.f32(196)
  const e = sxy / d.f32(784)
  const sum = (sxx + syy - d.f32(8) * s0) / d.f32(588)
  const delta = (sxx - syy) / d.f32(588)
  const a = (sum + delta) / d.f32(2)
  const c = (sum - delta) / d.f32(2)
  const f = (s0 - d.f32(196) * sum) / d.f32(49)
  const det = d.f32(4) * a * c - e * e
  let valid = a < d.f32(0) && c < d.f32(0) && det > d.f32(1e-9)
  let vx = d.f32(0)
  let vy = d.f32(0)
  let peak = d.f32(0)
  if (valid) {
    vx = (d.f32(-2) * c * b + e * dd) / det
    vy = (d.f32(-2) * a * dd + e * b) / det
    valid = abs(vx) <= d.f32(2.5) && abs(vy) <= d.f32(2.5)
  }
  if (valid) {
    peak = f + b * vx + dd * vy + a * vx * vx + e * vx * vy + c * vy * vy
    valid = peak >= d.f32(ROI_PEAK_MIN_RESPONSE)
  }
  let support = d.u32(0)
  if (valid) {
    for (let dy = -1; dy <= 1; dy += 1) {
      for (let dx = -1; dx <= 1; dx += 1) {
        if (candidate.values[d.u32((dy + 3) * 7 + (dx + 3))]! >= peak * d.f32(0.5)) {
          support += d.u32(1)
        }
      }
    }
  }
  if (valid) {
    fitLayout.$.fitted[slot] = FittedPeak({
      x: d.f32(candidate.x) + vx,
      y: d.f32(candidate.y) + vy,
      peak,
      polarity: candidate.polarity,
      suppressed: d.u32(0),
      support,
    })
  } else {
    fitLayout.$.fitted[slot] = FittedPeak({
      x: d.f32(candidate.x),
      y: d.f32(candidate.y),
      peak: d.f32(0),
      polarity: candidate.polarity,
      suppressed: d.u32(0),
      support: d.u32(0),
    })
  }
}

const circleResetKernel = () => {
  'use gpu'
  atomicStore(appendLayout.$.circleCount, d.u32(0))
}

const appendKernel = (index: number) => {
  'use gpu'
  const slot = d.u32(index)
  if (slot >= atomicLoad(appendLayout.$.candidateCount)) {
    return
  }
  const fitted = appendLayout.$.fitted[slot]!
  let rejectReason = d.u32(0)
  if (fitted.peak <= d.f32(0)) {
    rejectReason = d.u32(ROI_REJECT_FIT)
  } else if (fitted.suppressed === d.u32(1)) {
    rejectReason = d.u32(ROI_REJECT_NMS)
  } else if (fitted.support < appendLayout.$.params.minSupport) {
    rejectReason = d.u32(ROI_REJECT_LOW_AREA)
  } else {
    const minPeak = fitted.polarity === d.u32(0)
      ? appendLayout.$.params.blackMinPeak
      : appendLayout.$.params.whiteMinPeak
    if (fitted.peak < minPeak) {
      rejectReason = d.u32(ROI_REJECT_GATE)
    }
  }
  if (rejectReason !== d.u32(0)) {
    const rejectIndex = atomicAdd(appendLayout.$.rejectCount, d.u32(1))
    if (rejectIndex >= d.u32(ROI_PEAK_MAX_CIRCLES)) {
      return
    }
    appendLayout.$.rejects[rejectIndex] = RoiCircleInstance({
      x: fitted.x + d.f32(0.5),
      y: fitted.y + d.f32(0.5),
      polarity: fitted.polarity,
      reason: rejectReason,
    })
    return
  }
  const circleIndex = atomicAdd(appendLayout.$.circleCount, d.u32(1))
  if (circleIndex >= d.u32(ROI_PEAK_MAX_CIRCLES)) {
    return
  }
  appendLayout.$.circles[circleIndex] = RoiCircleInstance({
    x: fitted.x + d.f32(0.5),
    y: fitted.y + d.f32(0.5),
    polarity: fitted.polarity,
    reason: d.u32(0),
  })
}

const publishKernel = () => {
  'use gpu'
  const count = atomicLoad(publishLayout.$.circleCount)
  publishLayout.$.drawIndirect.instanceCount = min(count, d.u32(ROI_PEAK_MAX_CIRCLES))
  const rejects = atomicLoad(publishLayout.$.rejectCount)
  publishLayout.$.rejectDrawIndirect.instanceCount = min(rejects, d.u32(ROI_PEAK_MAX_CIRCLES))
}

export type RoiPeakStage = {
  setThresholds: (
    blackCut: number,
    whiteCut: number,
    valid: boolean,
    blackGap: number,
    whiteGap: number,
  ) => void
  encodeDetection: () => void
  clear: () => void
  destroy: () => void
  circleBuffer: ReturnType<TgpuRoot['createBuffer']>
  drawIndirectBuffer: ReturnType<TgpuRoot['createBuffer']>
  rejectBuffer: ReturnType<TgpuRoot['createBuffer']>
  rejectDrawIndirectBuffer: ReturnType<TgpuRoot['createBuffer']>
}

export function createRoiPeakStage(
  root: TgpuRoot,
  frameWidth: number,
  frameHeight: number,
  cameraTexture: TgpuTexture & SampledFlag,
  maskTexture: TgpuTexture & SampledFlag,
): RoiPeakStage {
  const count = frameWidth * frameHeight
  const horizontalBuffer = root.createBuffer(d.arrayOf(d.f32, count)).$usage('storage')
  const blurredBuffer = root.createBuffer(d.arrayOf(d.f32, count)).$usage('storage')
  const candidateBuffer = root.createBuffer(RoiCandidateSchema).$usage('storage')
  const candidateCountBuffer = root.createBuffer(d.atomic(d.u32)).$usage('storage')
  const fittedBuffer = root.createBuffer(FittedSchema).$usage('storage')
  const nearestBuffer = root.createBuffer(NearestSchema).$usage('storage')
  const distanceStatsBuffer = root.createBuffer(DistanceStats).$usage('storage')
  distanceStatsBuffer.write({ meanDistance: 0, radius: 2 })
  const circleBuffer = root.createBuffer(RoiCircleSchema).$usage('storage')
  const circleCountBuffer = root.createBuffer(d.atomic(d.u32)).$usage('storage')
  const rejectBuffer = root.createBuffer(RoiCircleSchema).$usage('storage')
  const rejectCountBuffer = root.createBuffer(d.atomic(d.u32)).$usage('storage')
  const drawIndirectBuffer = root
    .createBuffer(RoiDrawIndirect)
    .$usage('storage', 'indirect')
  const rejectDrawIndirectBuffer = root
    .createBuffer(RoiDrawIndirect)
    .$usage('storage', 'indirect')
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
  const detectionUniform = root.createUniform(detectionParams, {
    blackCut: 0,
    whiteCut: 256,
    valid: 0,
    minResponse: ROI_PEAK_MIN_RESPONSE,
    blackMinPeak: ROI_PEAK_MIN_RESPONSE,
    whiteMinPeak: ROI_PEAK_MIN_RESPONSE,
    minSupport: 1,
  })

  const horizontalLayout = tgpu.bindGroupLayout({
    cameraTexture: { texture: d.texture2d() },
    horizontal: { storage: d.arrayOf(d.f32, count), access: 'mutable' },
  })
  const verticalLayout = tgpu.bindGroupLayout({
    horizontal: { storage: d.arrayOf(d.f32, count), access: 'readonly' },
    blurred: { storage: d.arrayOf(d.f32, count), access: 'mutable' },
  })
  const horizontalPipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    'use gpu'
    const size = std.textureDimensions(horizontalLayout.$.cameraTexture)
    const px = d.u32(x)
    const py = d.u32(y)
    let sum = d.f32(0)
    for (let offset = -ROI_PEAK_BLUR_RADIUS; offset <= ROI_PEAK_BLUR_RADIUS; offset += 1) {
      const sx = clamp(d.i32(px) + d.i32(offset), d.i32(0), d.i32(size.x) - d.i32(1))
      const color = std.textureLoad(
        horizontalLayout.$.cameraTexture,
        d.vec2u(d.u32(sx), py),
        d.i32(0),
      )
      const luma = std.dot(color.rgb, d.vec3f(0.299, 0.587, 0.114))
      sum += luma * BlurWeights.$[d.u32(offset + ROI_PEAK_BLUR_RADIUS)]!
    }
    horizontalLayout.$.horizontal[py * size.x + px] = sum
  })
  const verticalPipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    'use gpu'
    const px = d.u32(x)
    const py = d.u32(y)
    let sum = d.f32(0)
    for (let offset = -ROI_PEAK_BLUR_RADIUS; offset <= ROI_PEAK_BLUR_RADIUS; offset += 1) {
      const sy = clamp(d.i32(py) + d.i32(offset), d.i32(0), d.i32(frameHeight) - d.i32(1))
      sum +=
        verticalLayout.$.horizontal[d.u32(sy) * d.u32(frameWidth) + px]! *
        BlurWeights.$[d.u32(offset + ROI_PEAK_BLUR_RADIUS)]!
    }
    verticalLayout.$.blurred[py * d.u32(frameWidth) + px] = sum
  })
  const candidatePipeline = root.createGuardedComputePipeline(candidateKernel)
  const candidateResetPipeline = root.createGuardedComputePipeline(() => {
    'use gpu'
    atomicStore(candidateLayout.$.candidateCount, d.u32(0))
    atomicStore(candidateLayout.$.rejectCount, d.u32(0))
  })
  // Suppression radius is measured, not resolution-derived: use mean
  // nearest-neighbor distance across all fitted peaks, then reject anything
  // closer than 60% of that when a higher-quality neighbor exists.
  const nearestKernel = (index: number) => {
    'use gpu'
    const slot = d.u32(index)
    if (slot >= atomicLoad(nearestLayout.$.candidateCount)) {
      return
    }
    const mine = nearestLayout.$.fitted[slot]!
    if (mine.peak <= d.f32(0)) {
      nearestLayout.$.nearest[slot] = d.f32(0)
      return
    }
    let bestSq = d.f32(1e12)
    for (let otherIndex = 0; otherIndex < ROI_PEAK_MAX_CANDIDATES; otherIndex += 1) {
      const otherSlot = d.u32(otherIndex)
      if (otherSlot >= atomicLoad(nearestLayout.$.candidateCount)) {
        break
      }
      if (otherSlot === slot) {
        continue
      }
      const other = nearestLayout.$.fitted[otherSlot]!
      if (other.peak <= d.f32(0)) {
        continue
      }
      const dx = other.x - mine.x
      const dy = other.y - mine.y
      const distSq = dx * dx + dy * dy
      if (distSq < bestSq) {
        bestSq = distSq
      }
    }
    nearestLayout.$.nearest[slot] = bestSq < d.f32(1e11) ? sqrt(bestSq) : d.f32(0)
  }
  const meanDistanceKernel = () => {
    'use gpu'
    const count = atomicLoad(statsLayout.$.candidateCount)
    let sum = d.f32(0)
    let valid = d.u32(0)
    for (let index = 0; index < ROI_PEAK_MAX_CANDIDATES; index += 1) {
      if (d.u32(index) >= count) {
        break
      }
      const distance = statsLayout.$.nearest[d.u32(index)]!
      if (distance > d.f32(0)) {
        sum += distance
        valid += d.u32(1)
      }
    }
    if (valid < d.u32(2)) {
      statsLayout.$.stats.meanDistance = d.f32(0)
      statsLayout.$.stats.radius = d.f32(2)
      return
    }
    const mean = sum / d.f32(valid)
    statsLayout.$.stats.meanDistance = mean
    statsLayout.$.stats.radius = clamp(mean * d.f32(0.6), d.f32(1.5), d.f32(16))
  }
  const suppressionKernel = (index: number) => {
    'use gpu'
    const slot = d.u32(index)
    if (slot >= atomicLoad(suppressionLayout.$.candidateCount)) {
      return
    }
    const mine = suppressionLayout.$.fitted[slot]!
    if (mine.peak <= d.f32(0)) {
      return
    }
    const radius = suppressionLayout.$.stats.radius
    const radiusSq = radius * radius
    for (let otherIndex = 0; otherIndex < ROI_PEAK_MAX_CANDIDATES; otherIndex += 1) {
      const otherSlot = d.u32(otherIndex)
      if (otherSlot >= atomicLoad(suppressionLayout.$.candidateCount)) {
        break
      }
      if (otherSlot === slot) {
        continue
      }
      const other = suppressionLayout.$.fitted[otherSlot]!
      if (other.peak <= d.f32(0)) {
        continue
      }
      const dx = other.x - mine.x
      const dy = other.y - mine.y
      if (dx * dx + dy * dy >= radiusSq) {
        continue
      }
      const better =
        other.peak > mine.peak ||
        (other.peak === mine.peak && other.support > mine.support) ||
        (other.peak === mine.peak && other.support === mine.support && otherSlot < slot)
      if (better) {
        suppressionLayout.$.fitted[slot]!.suppressed = d.u32(1)
        return
      }
    }
  }
  const nearestPipeline = root.createGuardedComputePipeline(nearestKernel)
  const meanDistancePipeline = root.createGuardedComputePipeline(meanDistanceKernel)
  const suppressionPipeline = root.createGuardedComputePipeline(suppressionKernel)
  const fitPipeline = root.createGuardedComputePipeline(fitKernel)
  const circleResetPipeline = root.createGuardedComputePipeline(circleResetKernel)
  const appendPipeline = root.createGuardedComputePipeline(appendKernel)
  const publishPipeline = root.createGuardedComputePipeline(publishKernel)

  const horizontalBindGroup = root.createBindGroup(horizontalLayout, {
    cameraTexture,
    horizontal: horizontalBuffer,
  })
  const verticalBindGroup = root.createBindGroup(verticalLayout, {
    horizontal: horizontalBuffer,
    blurred: blurredBuffer,
  })
  const candidateBindGroup = root.createBindGroup(candidateLayout, {
    cameraTexture,
    blurred: blurredBuffer,
    maskTexture,
    candidates: candidateBuffer,
    candidateCount: candidateCountBuffer,
    rejects: rejectBuffer,
    rejectCount: rejectCountBuffer,
    params: detectionUniform,
  })
  const nearestBindGroup = root.createBindGroup(nearestLayout, {
    fitted: fittedBuffer,
    candidateCount: candidateCountBuffer,
    nearest: nearestBuffer,
  })
  const statsBindGroup = root.createBindGroup(statsLayout, {
    nearest: nearestBuffer,
    candidateCount: candidateCountBuffer,
    stats: distanceStatsBuffer,
  })
  const suppressionBindGroup = root.createBindGroup(suppressionLayout, {
    fitted: fittedBuffer,
    candidateCount: candidateCountBuffer,
    stats: distanceStatsBuffer,
  })
  const fitBindGroup = root.createBindGroup(fitLayout, {
    candidates: candidateBuffer,
    fitted: fittedBuffer,
    candidateCount: candidateCountBuffer,
  })
  const appendBindGroup = root.createBindGroup(appendLayout, {
    fitted: fittedBuffer,
    candidateCount: candidateCountBuffer,
    circles: circleBuffer,
    circleCount: circleCountBuffer,
    rejects: rejectBuffer,
    rejectCount: rejectCountBuffer,
    params: detectionUniform,
  })
  const publishBindGroup = root.createBindGroup(publishLayout, {
    circleCount: circleCountBuffer,
    rejectCount: rejectCountBuffer,
    drawIndirect: drawIndirectBuffer,
    rejectDrawIndirect: rejectDrawIndirectBuffer,
  })

  let thresholdBlackCut = 0
  let thresholdWhiteCut = 256
  let thresholdValid = false
  let thresholdBlackMinPeak = ROI_PEAK_MIN_RESPONSE
  let thresholdWhiteMinPeak = ROI_PEAK_MIN_RESPONSE
  const minSupport = Math.max(1, Math.round(frameWidth / 640))
  const writeDetectionParams = () => {
    detectionUniform.write({
      blackCut: thresholdBlackCut,
      whiteCut: thresholdWhiteCut,
      valid: thresholdValid ? 1 : 0,
      minResponse: ROI_PEAK_MIN_RESPONSE,
      blackMinPeak: thresholdBlackMinPeak,
      whiteMinPeak: thresholdWhiteMinPeak,
      minSupport,
    })
  }

  return {
    circleBuffer,
    drawIndirectBuffer,
    rejectBuffer,
    rejectDrawIndirectBuffer,
    setThresholds(blackCut, whiteCut, valid, blackGap, whiteGap) {
      const minPeak = (gap: number) =>
        Math.max(
          ROI_PEAK_MIN_RESPONSE,
          (Math.max(0, gap) / 255) * ROI_PEAK_GAP_FRACTION,
        )
      thresholdBlackCut = blackCut
      thresholdWhiteCut = whiteCut
      thresholdValid = valid
      thresholdBlackMinPeak = minPeak(blackGap)
      thresholdWhiteMinPeak = minPeak(whiteGap)
      writeDetectionParams()
    },
    encodeDetection() {
      candidateResetPipeline.with(candidateBindGroup).dispatchThreads()
      horizontalPipeline.with(horizontalBindGroup).dispatchThreads(frameWidth, frameHeight)
      verticalPipeline.with(verticalBindGroup).dispatchThreads(frameWidth, frameHeight)
      candidatePipeline.with(candidateBindGroup).dispatchThreads(frameWidth, frameHeight)
      fitPipeline.with(fitBindGroup).dispatchThreads(ROI_PEAK_MAX_CANDIDATES)
      nearestPipeline.with(nearestBindGroup).dispatchThreads(ROI_PEAK_MAX_CANDIDATES)
      meanDistancePipeline.with(statsBindGroup).dispatchThreads()
      suppressionPipeline
        .with(suppressionBindGroup)
        .dispatchThreads(ROI_PEAK_MAX_CANDIDATES)
      circleResetPipeline.with(appendBindGroup).dispatchThreads()
      appendPipeline.with(appendBindGroup).dispatchThreads(ROI_PEAK_MAX_CANDIDATES)
      publishPipeline.with(publishBindGroup).dispatchThreads()
    },
    clear() {
      circleResetPipeline.with(appendBindGroup).dispatchThreads()
      publishPipeline.with(publishBindGroup).dispatchThreads()
    },
    destroy() {
      horizontalBuffer.destroy()
      blurredBuffer.destroy()
      candidateBuffer.destroy()
      candidateCountBuffer.destroy()
      fittedBuffer.destroy()
      nearestBuffer.destroy()
      distanceStatsBuffer.destroy()
      circleBuffer.destroy()
      circleCountBuffer.destroy()
      rejectBuffer.destroy()
      rejectCountBuffer.destroy()
      drawIndirectBuffer.destroy()
      rejectDrawIndirectBuffer.destroy()
    },
  }
}
