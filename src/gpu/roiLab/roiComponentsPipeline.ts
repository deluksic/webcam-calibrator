import type { SampledFlag, TgpuSampler, TgpuRoot, TgpuTexture } from 'typegpu'
import { d, std, tgpu } from 'typegpu'
import { atomicAdd, atomicLoad, atomicMax, atomicStore, clamp, intdiv, max, min, sqrt } from 'typegpu/std'

import { createPointerJumpLabeling } from '@/gpu/pipelines/pointerJumpPipeline'
import { createTileLabeling } from '@/gpu/roiLab/tileLabeling'
import { ROI_CIRCLE_VERTEX_COUNT } from '@/gpu/roiLab/roiCircleConstants'
import { RoiCircleInstance, RoiCircleSchema, RoiDrawIndirect } from '@/gpu/roiLab/roiPeakPipeline'

const COMPONENT_LABEL_INVALID = 0xffff_ffff
const POINTER_JUMP_STEPS = 6
const MIN_COMPONENT_AREA = 4
// Fixed compact-component cap. The default target has ~476 markers; 2048
// leaves headroom while keeping the NMS loops and fixed readback small.
const MAX_COMPONENTS = 2048
const SUPPRESSED_BIT = 0x8000_0000
// TEMPORARY: disabled to measure how much the radius filter was dropping at
// oblique angles. With it off nothing is suppressed, the raw component set is
// drawn and read back, and `encodeNms` skips its four dispatches.
const NMS_ENABLED = false
// `reason` doubles as the component area for the NMS passes. Nothing reads it
// while the filter is off, and a non-zero value would draw every circle as a
// rejection marker, so the raw set zeroes it by multiplying the area by 0.
const AREA_SCALE = NMS_ENABLED ? 1 : 0

const componentParams = d.struct({
  blackCut: d.u32,
  whiteCut: d.u32,
  valid: d.u32,
  residualGain: d.f32,
})

const NmsStats = d.struct({
  meanDistance: d.f32,
  radius: d.f32,
})

const ComponentStats = d.struct({
  count: d.atomic(d.u32),
  sumX: d.atomic(d.u32),
  sumY: d.atomic(d.u32),
  color: d.atomic(d.u32),
})

const rawLayout = tgpu.bindGroupLayout({
  cameraTexture: { texture: d.texture2d() },
  maskTexture: { texture: d.texture2d() },
  backgroundTexture: { texture: d.texture2d() },
  sampler: { sampler: 'filtering' },
  raw: { storage: d.arrayOf(d.u32), access: 'mutable' },
  params: { uniform: componentParams },
})

const erodeLayout = tgpu.bindGroupLayout({
  raw: { storage: d.arrayOf(d.u32), access: 'readonly' },
  eroded: { storage: d.arrayOf(d.u32), access: 'mutable' },
})

const dilateLayout = tgpu.bindGroupLayout({
  eroded: { storage: d.arrayOf(d.u32), access: 'readonly' },
  raw: { storage: d.arrayOf(d.u32), access: 'mutable' },
})

const guardLayout = tgpu.bindGroupLayout({
  raw: { storage: d.arrayOf(d.u32), access: 'readonly' },
  colors: { storage: d.arrayOf(d.u32), access: 'mutable' },
  edges: { storage: d.arrayOf(d.vec2f), access: 'mutable' },
})

const statsResetLayout = tgpu.bindGroupLayout({
  labels: { storage: d.arrayOf(d.u32), access: 'readonly' },
  colors: { storage: d.arrayOf(d.u32), access: 'readonly' },
  stats: { storage: d.arrayOf(ComponentStats), access: 'mutable' },
  componentCount: { storage: d.atomic(d.u32), access: 'mutable' },
  filteredCount: { storage: d.atomic(d.u32), access: 'mutable' },
})

const accumulateLayout = tgpu.bindGroupLayout({
  colors: { storage: d.arrayOf(d.u32), access: 'readonly' },
  labels: { storage: d.arrayOf(d.u32), access: 'readonly' },
  stats: { storage: d.arrayOf(ComponentStats), access: 'mutable' },
})

const finalizeLayout = tgpu.bindGroupLayout({
  colors: { storage: d.arrayOf(d.u32), access: 'readonly' },
  labels: { storage: d.arrayOf(d.u32), access: 'readonly' },
  stats: { storage: d.arrayOf(ComponentStats), access: 'mutable' },
  circles: { storage: RoiCircleSchema, access: 'mutable' },
  componentCount: { storage: d.atomic(d.u32), access: 'mutable' },
})

const labelColorLayout = tgpu.bindGroupLayout({
  colors: { storage: d.arrayOf(d.u32), access: 'readonly' },
  labels: { storage: d.arrayOf(d.u32), access: 'readonly' },
  labelTexture: { storageTexture: d.textureStorage2d('rgba8unorm', 'write-only') },
})

const nmsNearestLayout = tgpu.bindGroupLayout({
  circles: { storage: RoiCircleSchema, access: 'readonly' },
  componentCount: { storage: d.atomic(d.u32), access: 'mutable' },
  nearest: { storage: d.arrayOf(d.f32, MAX_COMPONENTS), access: 'mutable' },
})

const nmsStatsLayout = tgpu.bindGroupLayout({
  nearest: { storage: d.arrayOf(d.f32, MAX_COMPONENTS), access: 'readonly' },
  componentCount: { storage: d.atomic(d.u32), access: 'mutable' },
  stats: { storage: NmsStats, access: 'mutable' },
})

const nmsSuppressLayout = tgpu.bindGroupLayout({
  circles: { storage: RoiCircleSchema, access: 'mutable' },
  componentCount: { storage: d.atomic(d.u32), access: 'mutable' },
  stats: { storage: NmsStats, access: 'readonly' },
})

const nmsCompactLayout = tgpu.bindGroupLayout({
  circles: { storage: RoiCircleSchema, access: 'readonly' },
  componentCount: { storage: d.atomic(d.u32), access: 'mutable' },
  filtered: { storage: RoiCircleSchema, access: 'mutable' },
  filteredCount: { storage: d.atomic(d.u32), access: 'mutable' },
})

const publishLayout = tgpu.bindGroupLayout({
  componentCount: { storage: d.atomic(d.u32), access: 'mutable' },
  drawIndirect: { storage: RoiDrawIndirect, access: 'mutable' },
})

export type RoiComponentPoint = {
  x: number
  y: number
  color: 0 | 1
}

export type RoiComponentReadback = {
  points: RoiComponentPoint[]
  count: number
  bytesCopied: number
  truncated: boolean
}

export type RoiComponentsStage = {
  encodeMorphology: () => void
  encodeLabeling: () => void
  encodeStats: () => void
  encodeNms: () => void
  setLabelOverlay: (enabled: boolean) => void
  readComponents: () => Promise<RoiComponentReadback>
  setThresholds: (blackCut: number, whiteCut: number, valid: boolean) => void
  destroy: () => void
  circleBuffer: ReturnType<TgpuRoot['createBuffer']>
  drawIndirectBuffer: ReturnType<TgpuRoot['createBuffer']>
  labelTexture: ReturnType<TgpuRoot['createTexture']>
}

export function createRoiComponentsStage(
  root: TgpuRoot,
  frameWidth: number,
  frameHeight: number,
  cameraTexture: TgpuTexture & SampledFlag,
  maskTexture: TgpuTexture & SampledFlag,
  backgroundTexture: TgpuTexture & SampledFlag,
  sampler: TgpuSampler,
): RoiComponentsStage {
  const area = frameWidth * frameHeight
  const rawBuffer = root.createBuffer(d.arrayOf(d.u32, area)).$usage('storage')
  const erodedBuffer = root.createBuffer(d.arrayOf(d.u32, area)).$usage('storage')
  const colorBuffer = root.createBuffer(d.arrayOf(d.u32, area)).$usage('storage')
  const edgeBuffer = root.createBuffer(d.arrayOf(d.vec2f, area)).$usage('storage')
  const statsBuffer = root.createBuffer(d.arrayOf(ComponentStats, area)).$usage('storage')
  const componentCountBuffer = root.createBuffer(d.atomic(d.u32)).$usage('storage')
  const circleBuffer = root.createBuffer(RoiCircleSchema).$usage('storage')
  const filteredCircleBuffer = root.createBuffer(RoiCircleSchema).$usage('storage')
  const nearestBuffer = root.createBuffer(d.arrayOf(d.f32, MAX_COMPONENTS)).$usage('storage')
  const nmsStatsBuffer = root.createBuffer(NmsStats).$usage('storage')
  nmsStatsBuffer.write({ meanDistance: 0, radius: 2 })
  const filteredCountBuffer = root.createBuffer(d.atomic(d.u32)).$usage('storage')
  // The circles that are drawn and read back. With NMS off the filtered
  // buffers are never written, so both fall back to the raw component set.
  const drawCountBuffer = NMS_ENABLED ? filteredCountBuffer : componentCountBuffer
  const drawCircleBuffer = NMS_ENABLED ? filteredCircleBuffer : circleBuffer
  const componentReadbackBuffer = root.device.createBuffer({
    label: 'roi-component-readback',
    size: 4 + MAX_COMPONENTS * 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  })
  let componentReadbackBusy = false
  const labelTexture = root
    .createTexture({
      size: [frameWidth, frameHeight],
      format: 'rgba8unorm',
    })
    .$usage('sampled', 'storage')
  const drawIndirectBuffer = root.createBuffer(RoiDrawIndirect).$usage('storage', 'indirect')
  drawIndirectBuffer.write({
    vertexCount: ROI_CIRCLE_VERTEX_COUNT,
    instanceCount: 0,
    firstVertex: 0,
    firstInstance: 0,
  })
  // The labels overlay is only reachable from the ROI lab's view modes; the
  // calibration pipeline turns it off for its whole lifetime. It stays on by
  // default because a paused redraw samples the texture without re-encoding.
  let labelOverlayEnabled = true
  const paramsBuffer = root.createUniform(componentParams, {
    blackCut: 0,
    whiteCut: 256,
    valid: 0,
    residualGain: 1,
  })

  const rawPipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    'use gpu'
    const px = d.u32(x)
    const py = d.u32(y)
    const index = py * d.u32(frameWidth) + px
    const params = rawLayout.$.params
    if (params.valid === d.u32(0)) {
      rawLayout.$.raw[index] = d.u32(0)
      return
    }
    const maskSize = std.textureDimensions(rawLayout.$.maskTexture)
    const maskX = min(intdiv(px * maskSize.x, d.u32(frameWidth)), maskSize.x - d.u32(1))
    const maskY = min(intdiv(py * maskSize.y, d.u32(frameHeight)), maskSize.y - d.u32(1))
    const mask = std.textureLoad(rawLayout.$.maskTexture, d.vec2u(maskX, maskY), d.i32(0)).r
    if (mask < d.f32(0.5)) {
      rawLayout.$.raw[index] = d.u32(0)
      return
    }
    const color = std.textureLoad(rawLayout.$.cameraTexture, d.vec2u(px, py), d.i32(0))
    const uv = (d.vec2f(px, py) + d.vec2f(0.5, 0.5)).div(d.vec2f(d.f32(frameWidth), d.f32(frameHeight)))
    const background = std.textureSampleLevel(rawLayout.$.backgroundTexture, rawLayout.$.sampler, uv, d.f32(0)).r
    const luma = std.dot(color.rgb, d.vec3f(0.299, 0.587, 0.114))
    const flattened = std.clamp(d.f32(0.5) + (luma - background) * params.residualGain, d.f32(0), d.f32(1))
    const bin = d.u32(flattened * d.f32(255) + d.f32(0.5))
    if (bin < params.blackCut) {
      rawLayout.$.raw[index] = d.u32(1)
    } else if (bin >= params.whiteCut) {
      rawLayout.$.raw[index] = d.u32(2)
    } else {
      rawLayout.$.raw[index] = d.u32(0)
    }
  })

  // 2x2 opening: removes one-pixel bridges, preserves a 2x2 blob.
  const erodePipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    'use gpu'
    const px = d.i32(x)
    const py = d.i32(y)
    const index = d.u32(y) * d.u32(frameWidth) + d.u32(x)
    const color = erodeLayout.$.raw[index]!
    if (color === d.u32(0)) {
      erodeLayout.$.eroded[index] = d.u32(0)
      return
    }
    for (let oy = 0; oy <= 1; oy += 1) {
      for (let ox = 0; ox <= 1; ox += 1) {
        const nx = px + d.i32(ox)
        const ny = py + d.i32(oy)
        if (
          nx >= d.i32(frameWidth) ||
          ny >= d.i32(frameHeight) ||
          erodeLayout.$.raw[d.u32(ny * d.i32(frameWidth) + nx)]! !== color
        ) {
          erodeLayout.$.eroded[index] = d.u32(0)
          return
        }
      }
    }
    erodeLayout.$.eroded[index] = color
  })

  const dilatePipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    'use gpu'
    const px = d.i32(x)
    const py = d.i32(y)
    const index = d.u32(y) * d.u32(frameWidth) + d.u32(x)
    for (let oy = -1; oy <= 0; oy += 1) {
      for (let ox = -1; ox <= 0; ox += 1) {
        const nx = px + d.i32(ox)
        const ny = py + d.i32(oy)
        if (nx < d.i32(0) || ny < d.i32(0)) {
          continue
        }
        const seed = dilateLayout.$.eroded[d.u32(ny * d.i32(frameWidth) + nx)]!
        if (seed !== d.u32(0)) {
          dilateLayout.$.raw[index] = seed
          return
        }
      }
    }
    dilateLayout.$.raw[index] = d.u32(0)
  })

  // Never let a black and white pixel enter the same edge component.
  const guardPipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    'use gpu'
    const px = d.i32(x)
    const py = d.i32(y)
    const index = d.u32(y) * d.u32(frameWidth) + d.u32(x)
    const color = guardLayout.$.raw[index]!
    if (color === d.u32(0)) {
      guardLayout.$.colors[index] = d.u32(0)
      guardLayout.$.edges[index] = d.vec2f(0, 0)
      return
    }
    let opposite = false
    for (let oy = -1; oy <= 1; oy += 1) {
      for (let ox = -1; ox <= 1; ox += 1) {
        const nx = px + d.i32(ox)
        const ny = py + d.i32(oy)
        if (nx < d.i32(0) || nx >= d.i32(frameWidth) || ny < d.i32(0) || ny >= d.i32(frameHeight)) {
          continue
        }
        const neighbor = guardLayout.$.raw[d.u32(ny * d.i32(frameWidth) + nx)]!
        if (neighbor !== d.u32(0) && neighbor !== color) {
          opposite = true
        }
      }
    }
    if (opposite) {
      guardLayout.$.colors[index] = d.u32(0)
      guardLayout.$.edges[index] = d.vec2f(0, 0)
    } else {
      guardLayout.$.colors[index] = color
      guardLayout.$.edges[index] = d.vec2f(1, 0)
    }
  })

  const statsResetPipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    'use gpu'
    const index = d.u32(y) * d.u32(frameWidth) + d.u32(x)
    if (index === d.u32(0)) {
      atomicStore(statsResetLayout.$.componentCount, d.u32(0))
      atomicStore(statsResetLayout.$.filteredCount, d.u32(0))
    }
    if (statsResetLayout.$.colors[index]! === d.u32(0) || statsResetLayout.$.labels[index]! !== index) {
      return
    }
    atomicStore(statsResetLayout.$.stats[index]!.count, d.u32(0))
    atomicStore(statsResetLayout.$.stats[index]!.sumX, d.u32(0))
    atomicStore(statsResetLayout.$.stats[index]!.sumY, d.u32(0))
    atomicStore(statsResetLayout.$.stats[index]!.color, d.u32(0))
  })

  const accumulatePipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    'use gpu'
    const px = d.u32(x)
    const py = d.u32(y)
    const index = py * d.u32(frameWidth) + px
    const color = accumulateLayout.$.colors[index]!
    if (color === d.u32(0)) {
      return
    }
    const root = accumulateLayout.$.labels[index]!
    if (root === d.u32(COMPONENT_LABEL_INVALID)) {
      return
    }
    atomicAdd(accumulateLayout.$.stats[root]!.count, d.u32(1))
    atomicAdd(accumulateLayout.$.stats[root]!.sumX, px)
    atomicAdd(accumulateLayout.$.stats[root]!.sumY, py)
    atomicMax(accumulateLayout.$.stats[root]!.color, color)
  })

  const finalizePipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    'use gpu'
    const index = d.u32(y) * d.u32(frameWidth) + d.u32(x)
    if (finalizeLayout.$.colors[index]! === d.u32(0)) {
      return
    }
    if (finalizeLayout.$.labels[index]! !== index) {
      return
    }
    const stats = finalizeLayout.$.stats[index]!
    const count = atomicLoad(stats.count)
    if (count < d.u32(MIN_COMPONENT_AREA)) {
      return
    }
    const slot = atomicAdd(finalizeLayout.$.componentCount, d.u32(1))
    if (slot >= d.u32(MAX_COMPONENTS)) {
      return
    }
    const sumX = atomicLoad(stats.sumX)
    const sumY = atomicLoad(stats.sumY)
    const color = atomicLoad(stats.color)
    const polarity = color === d.u32(1) ? d.u32(0) : d.u32(1)
    finalizeLayout.$.circles[slot] = RoiCircleInstance({
      x: d.f32(sumX) / d.f32(count) + d.f32(0.5),
      y: d.f32(sumY) / d.f32(count) + d.f32(0.5),
      polarity,
      reason: count * d.u32(AREA_SCALE),
    })
  })

  const labelColorPipeline = root.createGuardedComputePipeline((x: number, y: number) => {
    'use gpu'
    const px = d.u32(x)
    const py = d.u32(y)
    const index = py * d.u32(frameWidth) + px
    if (labelColorLayout.$.colors[index]! === d.u32(0)) {
      std.textureStore(
        labelColorLayout.$.labelTexture,
        d.vec2u(px, py),
        d.vec4f(d.f32(0.04), d.f32(0.04), d.f32(0.05), d.f32(1)),
      )
      return
    }
    const label = labelColorLayout.$.labels[index]!
    let hash = label * d.u32(2654435761)
    hash ^= hash >>> d.u32(16)
    hash *= d.u32(2246822519)
    hash ^= hash >>> d.u32(13)
    const red = d.f32(hash & d.u32(255)) / d.f32(255)
    const green = d.f32((hash >>> d.u32(8)) & d.u32(255)) / d.f32(255)
    const blue = d.f32((hash >>> d.u32(16)) & d.u32(255)) / d.f32(255)
    std.textureStore(
      labelColorLayout.$.labelTexture,
      d.vec2u(px, py),
      d.vec4f(d.f32(0.2) + red * d.f32(0.8), d.f32(0.2) + green * d.f32(0.8), d.f32(0.2) + blue * d.f32(0.8), d.f32(1)),
    )
  })

  const nmsNearestPipeline = root.createGuardedComputePipeline((index: number) => {
    'use gpu'
    const i = d.u32(index)
    const count = atomicLoad(nmsNearestLayout.$.componentCount)
    if (i >= count) {
      return
    }
    const mine = nmsNearestLayout.$.circles[i]!
    let best = d.f32(1e12)
    for (let jIndex = 0; jIndex < MAX_COMPONENTS; jIndex += 1) {
      const j = d.u32(jIndex)
      if (j >= count) {
        break
      }
      if (j === i) {
        continue
      }
      const other = nmsNearestLayout.$.circles[j]!
      const dx = other.x - mine.x
      const dy = other.y - mine.y
      const distance = dx * dx + dy * dy
      if (distance < best) {
        best = distance
      }
    }
    nmsNearestLayout.$.nearest[i] = best < d.f32(1e11) ? sqrt(best) : d.f32(0)
  })

  const nmsStatsPipeline = root.createGuardedComputePipeline(() => {
    'use gpu'
    const count = atomicLoad(nmsStatsLayout.$.componentCount)
    let sum = d.f32(0)
    let valid = d.u32(0)
    for (let index = 0; index < MAX_COMPONENTS; index += 1) {
      if (d.u32(index) >= count) {
        break
      }
      const distance = nmsStatsLayout.$.nearest[d.u32(index)]!
      if (distance > d.f32(0)) {
        sum += distance
        valid += d.u32(1)
      }
    }
    if (valid < d.u32(2)) {
      nmsStatsLayout.$.stats.meanDistance = d.f32(0)
      nmsStatsLayout.$.stats.radius = d.f32(2)
      return
    }
    const mean = sum / d.f32(valid)
    nmsStatsLayout.$.stats.meanDistance = mean
    nmsStatsLayout.$.stats.radius = clamp(mean * d.f32(0.6), d.f32(1.5), d.f32(16))
  })

  const nmsSuppressPipeline = root.createGuardedComputePipeline((index: number) => {
    'use gpu'
    const i = d.u32(index)
    const count = atomicLoad(nmsSuppressLayout.$.componentCount)
    if (i >= count) {
      return
    }
    const mine = nmsSuppressLayout.$.circles[i]!
    const mineArea = mine.reason & d.u32(0x7fff_ffff)
    const radius = nmsSuppressLayout.$.stats.radius
    for (let jIndex = 0; jIndex < MAX_COMPONENTS; jIndex += 1) {
      const j = d.u32(jIndex)
      if (j >= count) {
        break
      }
      if (j === i) {
        continue
      }
      const other = nmsSuppressLayout.$.circles[j]!
      const otherArea = other.reason & d.u32(0x7fff_ffff)
      const areaRadius = d.f32(0.7) * sqrt(d.f32(max(mineArea, otherArea)))
      const pairRadius = max(radius, areaRadius)
      const dx = other.x - mine.x
      const dy = other.y - mine.y
      if (dx * dx + dy * dy >= pairRadius * pairRadius) {
        continue
      }
      if (otherArea > mineArea || (otherArea === mineArea && j < i)) {
        nmsSuppressLayout.$.circles[i]!.reason = mine.reason | d.u32(SUPPRESSED_BIT)
        return
      }
    }
  })

  const nmsCompactPipeline = root.createGuardedComputePipeline((index: number) => {
    'use gpu'
    const i = d.u32(index)
    const count = atomicLoad(nmsCompactLayout.$.componentCount)
    if (i >= count) {
      return
    }
    const circle = nmsCompactLayout.$.circles[i]!
    if ((circle.reason & d.u32(SUPPRESSED_BIT)) !== d.u32(0)) {
      return
    }
    const slot = atomicAdd(nmsCompactLayout.$.filteredCount, d.u32(1))
    if (slot >= d.u32(MAX_COMPONENTS)) {
      return
    }
    nmsCompactLayout.$.filtered[slot] = RoiCircleInstance({
      x: circle.x,
      y: circle.y,
      polarity: circle.polarity,
      reason: d.u32(0),
    })
  })

  const publishPipeline = root.createGuardedComputePipeline(() => {
    'use gpu'
    publishLayout.$.drawIndirect.instanceCount = min(atomicLoad(publishLayout.$.componentCount), d.u32(MAX_COMPONENTS))
  })

  const pointerJump = createPointerJumpLabeling(
    root,
    frameWidth,
    frameHeight,
    edgeBuffer,
    POINTER_JUMP_STEPS,
    true,
  )
  // Tile labeling is three dispatches (tile roots, border merge, resolve)
  // instead of init + POINTER_JUMP_STEPS. The pointer-jump path stays wired up
  // behind this switch so it can be compared or fallen back to on device.
  const LABELING_MODE: 'tile' | 'pointerJump' = 'tile'
  const tileLabeling =
    LABELING_MODE === 'tile'
      ? createTileLabeling(root, frameWidth, frameHeight, edgeBuffer)
      : undefined
  const labelBuffer =
    tileLabeling === undefined ? pointerJump.pointerJumpBuffer0 : tileLabeling.labels

  const rawBindGroup = root.createBindGroup(rawLayout, {
    cameraTexture,
    maskTexture,
    backgroundTexture,
    sampler,
    raw: rawBuffer,
    params: paramsBuffer,
  })
  const erodeBindGroup = root.createBindGroup(erodeLayout, {
    raw: rawBuffer,
    eroded: erodedBuffer,
  })
  const dilateBindGroup = root.createBindGroup(dilateLayout, {
    eroded: erodedBuffer,
    raw: rawBuffer,
  })
  const guardBindGroup = root.createBindGroup(guardLayout, {
    raw: rawBuffer,
    colors: colorBuffer,
    edges: edgeBuffer,
  })
  const statsResetBindGroup = root.createBindGroup(statsResetLayout, {
    labels: labelBuffer,
    colors: colorBuffer,
    stats: statsBuffer,
    componentCount: componentCountBuffer,
    filteredCount: filteredCountBuffer,
  })
  const accumulateBindGroup = root.createBindGroup(accumulateLayout, {
    colors: colorBuffer,
    labels: labelBuffer,
    stats: statsBuffer,
  })
  const finalizeBindGroup = root.createBindGroup(finalizeLayout, {
    colors: colorBuffer,
    labels: labelBuffer,
    stats: statsBuffer,
    circles: circleBuffer,
    componentCount: componentCountBuffer,
  })
  const labelColorBindGroup = root.createBindGroup(labelColorLayout, {
    colors: colorBuffer,
    labels: labelBuffer,
    labelTexture,
  })
  const nmsNearestBindGroup = root.createBindGroup(nmsNearestLayout, {
    circles: circleBuffer,
    componentCount: componentCountBuffer,
    nearest: nearestBuffer,
  })
  const nmsStatsBindGroup = root.createBindGroup(nmsStatsLayout, {
    nearest: nearestBuffer,
    componentCount: componentCountBuffer,
    stats: nmsStatsBuffer,
  })
  const nmsSuppressBindGroup = root.createBindGroup(nmsSuppressLayout, {
    circles: circleBuffer,
    componentCount: componentCountBuffer,
    stats: nmsStatsBuffer,
  })
  const nmsCompactBindGroup = root.createBindGroup(nmsCompactLayout, {
    circles: circleBuffer,
    componentCount: componentCountBuffer,
    filtered: filteredCircleBuffer,
    filteredCount: filteredCountBuffer,
  })
  const publishBindGroup = root.createBindGroup(publishLayout, {
    componentCount: drawCountBuffer,
    drawIndirect: drawIndirectBuffer,
  })

  return {
    circleBuffer: drawCircleBuffer,
    drawIndirectBuffer,
    labelTexture,
    async readComponents() {
      if (componentReadbackBusy) {
        return { points: [], count: 0, bytesCopied: 0, truncated: false }
      }
      componentReadbackBusy = true
      try {
        const bytesCopied = 4 + MAX_COMPONENTS * 16
        const encoder = root.device.createCommandEncoder({ label: 'roi-component-readback' })
        encoder.copyBufferToBuffer(root.unwrap(drawCountBuffer), 0, componentReadbackBuffer, 0, 4)
        encoder.copyBufferToBuffer(
          root.unwrap(drawCircleBuffer),
          0,
          componentReadbackBuffer,
          4,
          MAX_COMPONENTS * 16,
        )
        root.device.queue.submit([encoder.finish()])
        await componentReadbackBuffer.mapAsync(GPUMapMode.READ, 0, bytesCopied)
        const view = new DataView(componentReadbackBuffer.getMappedRange(0, bytesCopied))
        const rawCount = view.getUint32(0, true)
        const count = Math.min(rawCount, MAX_COMPONENTS)
        const truncated = rawCount > MAX_COMPONENTS
        const points: RoiComponentPoint[] = []
        for (let index = 0; index < count; index += 1) {
          const offset = 4 + index * 16
          const polarity = view.getUint32(offset + 8, true)
          points.push({
            x: view.getFloat32(offset, true),
            y: view.getFloat32(offset + 4, true),
            color: polarity === 0 ? 0 : 1,
          })
        }
        return { points, count, bytesCopied, truncated }
      } finally {
        if (componentReadbackBuffer.mapState === 'mapped') {
          componentReadbackBuffer.unmap()
        }
        componentReadbackBusy = false
      }
    },
    setThresholds(blackCut, whiteCut, valid) {
      paramsBuffer.write({
        blackCut,
        whiteCut,
        valid: valid ? 1 : 0,
        residualGain: 1,
      })
    },
    setLabelOverlay(enabled) {
      labelOverlayEnabled = enabled
    },
    encodeMorphology() {
      rawPipeline.with(rawBindGroup).dispatchThreads(frameWidth, frameHeight)
      erodePipeline.with(erodeBindGroup).dispatchThreads(frameWidth, frameHeight)
      dilatePipeline.with(dilateBindGroup).dispatchThreads(frameWidth, frameHeight)
      guardPipeline.with(guardBindGroup).dispatchThreads(frameWidth, frameHeight)
    },
    encodeLabeling() {
      const encoder = root.device.createCommandEncoder({ label: 'roi-component-ccl' })
      const pass = encoder.beginComputePass({ label: 'roi-component-ccl' })
      if (tileLabeling !== undefined) {
        tileLabeling.encodeCompute(pass)
      } else {
        pointerJump.encodeCompute(pass)
      }
      pass.end()
      root.device.queue.submit([encoder.finish()])
    },
    encodeStats() {
      statsResetPipeline.with(statsResetBindGroup).dispatchThreads(frameWidth, frameHeight)
      accumulatePipeline.with(accumulateBindGroup).dispatchThreads(frameWidth, frameHeight)
      // Off only when the owning pipeline can never show this overlay; gating
      // it on the *current* view mode would leave paused redraws black.
      if (labelOverlayEnabled) {
        labelColorPipeline.with(labelColorBindGroup).dispatchThreads(frameWidth, frameHeight)
      }
      finalizePipeline.with(finalizeBindGroup).dispatchThreads(frameWidth, frameHeight)
    },
    encodeNms() {
      if (NMS_ENABLED) {
        nmsNearestPipeline.with(nmsNearestBindGroup).dispatchThreads(MAX_COMPONENTS)
        nmsStatsPipeline.with(nmsStatsBindGroup).dispatchThreads()
        nmsSuppressPipeline.with(nmsSuppressBindGroup).dispatchThreads(MAX_COMPONENTS)
        nmsCompactPipeline.with(nmsCompactBindGroup).dispatchThreads(MAX_COMPONENTS)
      }
      publishPipeline.with(publishBindGroup).dispatchThreads()
    },
    destroy() {
      rawBuffer.destroy()
      erodedBuffer.destroy()
      colorBuffer.destroy()
      edgeBuffer.destroy()
      statsBuffer.destroy()
      componentCountBuffer.destroy()
      circleBuffer.destroy()
      filteredCircleBuffer.destroy()
      nearestBuffer.destroy()
      nmsStatsBuffer.destroy()
      filteredCountBuffer.destroy()
      labelTexture.destroy()
      drawIndirectBuffer.destroy()
      componentReadbackBuffer.destroy()
      tileLabeling?.destroy()
      pointerJump.pointerJumpBuffer0.destroy()
      pointerJump.pointerJumpBuffer1.destroy()
      pointerJump.pointerJumpAtomicBuffer.destroy()
    },
  }
}
