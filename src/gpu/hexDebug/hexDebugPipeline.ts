import type { TgpuRoot } from 'typegpu'
import { common, d, tgpu } from 'typegpu'
import {
  abs,
  atomicAdd,
  bitcastU32toF32,
  atomicCompareExchangeWeak,
  atomicLoad,
  atomicMax,
  atomicMin,
  atomicStore,
  floor,
  max,
  min,
  pow,
  round,
  select,
  sqrt,
} from 'typegpu/std'

import { createMultiScaleDohStage } from '@/gpu/hexDebug/dohMultiScale'
import { createCopyIngest } from '@/gpu/pipelines/copyPipeline'
import { createGrayStage } from '@/gpu/pipelines/grayPipeline'
import {
  createPointerJumpAtomicToLabelsPipeline,
  createPointerJumpLayouts,
  createPointerJumpStepPipeline,
} from '@/gpu/pipelines/pointerJumpPipeline'
import type { HexDebugComponent, HexDebugFrame, HexDebugRejection, HexDebugTuning } from '@/lib/hexDebug/types'

const WORKGROUP_SIZE = 64
const MAX_COMPONENTS = 65536
const ROOT_JUMP_ITERATIONS = 12
const GRAY_SCALE = 31

const BlockStats = d.struct({
  count: d.u32,
  sumGray: d.u32,
  sumGraySq: d.u32,
  sumX: d.u32,
  sumY: d.u32,
  minX: d.u32,
  maxX: d.u32,
  minY: d.u32,
  maxY: d.u32,
})

const ComponentStatsAtomic = d.struct({
  count: d.atomic(d.u32),
  sumGray: d.atomic(d.u32),
  sumGraySq: d.atomic(d.u32),
  sumX: d.atomic(d.u32),
  sumY: d.atomic(d.u32),
  minX: d.atomic(d.u32),
  maxX: d.atomic(d.u32),
  minY: d.atomic(d.u32),
  maxY: d.atomic(d.u32),
})

const ComponentStatsReadonly = d.struct({
  count: d.u32,
  sumGray: d.u32,
  sumGraySq: d.u32,
  sumX: d.u32,
  sumY: d.u32,
  minX: d.u32,
  maxX: d.u32,
  minY: d.u32,
  maxY: d.u32,
})

const ComponentRow = d.struct({
  label: d.u32,
  count: d.u32,
  sumGray: d.u32,
  sumGraySq: d.u32,
  sumX: d.u32,
  sumY: d.u32,
  minX: d.u32,
  maxX: d.u32,
  minY: d.u32,
  maxY: d.u32,
  rejection: d.u32,
  fillRatio: d.f32,
  ringMean: d.f32,
  ringVariance: d.f32,
})

const DebugParams = d.struct({
  width: d.u32,
  height: d.u32,
  blockCols: d.u32,
  blockRows: d.u32,
  baseTolerance: d.f32,
  z: d.f32,
  maxTolerance: d.f32,
  minArea: d.u32,
  maxArea: d.u32,
  minFillPercent: d.u32,
  ringMargin: d.u32,
  minContrast: d.f32,
  maxRingVariance: d.f32,
})

const BoxParams = d.struct({
  width: d.u32,
  height: d.u32,
  activeCount: d.u32,
  showRejections: d.u32,
})

const RenderParams = d.struct({
  width: d.u32,
  height: d.u32,
  mode: d.u32,
})

export type HexDebugPipeline = {
  encode(enc: GPUCommandEncoder, video: HTMLVideoElement, tuning: HexDebugTuning, options?: { doh?: boolean }): void
  readComponents(): Promise<HexDebugFrame>
  render(
    enc: GPUCommandEncoder,
    view: GPUTextureView,
    mode: 'gray' | 'components' | 'doh' | 'boxes' | 'nms',
    activeCount: number,
    showRejections: boolean,
  ): void
  setNmsThreshold(threshold: number): void
  destroy(): void
}

function toComponent(row: d.Infer<typeof ComponentRow>): HexDebugComponent {
  const area = Math.max(1, row.count)
  const meanGray = row.sumGray / (area * GRAY_SCALE)
  const secondMoment = row.sumGraySq / (area * GRAY_SCALE * GRAY_SCALE)
  const rejection: HexDebugRejection =
    row.rejection === 1
      ? 'fill'
      : row.rejection === 2
        ? 'ring'
        : row.rejection === 3
          ? 'contrast'
          : row.rejection === 4
            ? 'noRing'
            : 'accepted'
  return {
    label: row.label,
    area,
    meanGray,
    variance: Math.max(0, secondMoment - meanGray * meanGray),
    bbox: { minX: row.minX, maxX: row.maxX, minY: row.minY, maxY: row.maxY },
    centroid: { x: row.sumX / area, y: row.sumY / area },
    color: meanGray < 0.5 ? 0 : 1,
    fillRatio: row.fillRatio,
    ringMean: row.ringMean,
    ringVariance: row.ringVariance,
    rejection,
  }
}

export function createHexDebugPipeline(
  root: TgpuRoot,
  width: number,
  height: number,
  initialTuning: HexDebugTuning,
  presentationFormat: GPUTextureFormat,
): HexDebugPipeline {
  const cols = width
  const rows = height
  const blockCount = width * height
  const paramsBuffer = root.createBuffer(DebugParams).$usage('uniform')
  const ingest = createCopyIngest(root, width, height)
  const gray = createGrayStage(root, width, height, ingest.grayTex.createView(d.texture2d(d.f32)), { linearize: true })
  const dohStage = createMultiScaleDohStage(root, width, height, gray.buffer)
  const dohResponseBuffer = dohStage.responseBuffer
  const dohMaxBuffer = dohStage.maxBuffer

  const blockStatsBuffer = root.createBuffer(d.arrayOf(BlockStats, blockCount)).$usage('storage')
  const componentStatsBuffer = root.createBuffer(d.arrayOf(ComponentStatsAtomic, blockCount)).$usage('storage')
  const parentBuffer = root.createBuffer(d.arrayOf(d.atomic(d.u32), blockCount)).$usage('storage')
  const locksBuffer = root.createBuffer(d.arrayOf(d.atomic(d.u32), blockCount)).$usage('storage')
  const rootsA = root.createBuffer(d.arrayOf(d.u32, blockCount)).$usage('storage')
  const rootsB = root.createBuffer(d.arrayOf(d.u32, blockCount)).$usage('storage')
  const componentCounter = root.createBuffer(d.atomic(d.u32)).$usage('storage')
  const componentRowsBuffer = root.createBuffer(d.arrayOf(ComponentRow, MAX_COMPONENTS)).$usage('storage')
  const boxMaskBuffer = root.createBuffer(d.arrayOf(d.u32, blockCount)).$usage('storage')
  const boxParamsBuffer = root.createBuffer(BoxParams).$usage('uniform')
  const renderParamsBuffer = root.createBuffer(RenderParams).$usage('uniform')

  let currentTuning = { ...initialTuning }
  let lastIterations = initialTuning.iterations
  const writeParams = (): void => {
    paramsBuffer.write({
      width,
      height,
      blockCols: cols,
      blockRows: rows,
      baseTolerance: currentTuning.baseTolerance,
      z: currentTuning.z,
      maxTolerance: currentTuning.maxTolerance,
      minArea: currentTuning.minArea,
      maxArea: currentTuning.maxArea,
      minFillPercent: currentTuning.minFillPercent,
      ringMargin: currentTuning.ringMargin,
      minContrast: currentTuning.minContrast,
      maxRingVariance: currentTuning.maxRingVariance,
    })
  }
  writeParams()

  const blockLayout = tgpu
    .bindGroupLayout({
      params: { uniform: DebugParams },
      grayBuffer: { storage: d.arrayOf(d.f32, width * height), access: 'readonly' },
      blocks: { storage: d.arrayOf(BlockStats, blockCount), access: 'mutable' },
    })
    .$name('hex-debug-block-init')
  const blockPipeline = root
    .createComputePipeline({
      compute: tgpu.computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [WORKGROUP_SIZE, 1, 1],
      })((input) => {
        'use gpu'
        const blockId = d.u32(input.gid.x)
        if (blockId >= d.u32(blockCount)) {
          return
        }
        const p = blockLayout.$.params
        const row = d.u32(floor(d.f32(blockId) / d.f32(p.blockCols)))
        const col = blockId - row * p.blockCols
        const grayValue = blockLayout.$.grayBuffer[blockId]!
        const q = min(d.u32(127), max(d.u32(0), d.u32(round(grayValue * d.f32(GRAY_SCALE)))))
        blockLayout.$.blocks[blockId] = BlockStats({
          count: d.u32(1),
          sumGray: q,
          sumGraySq: q * q,
          sumX: col,
          sumY: row,
          minX: col,
          maxX: col,
          minY: row,
          maxY: row,
        })
      }),
    })
    .$name('hex-debug-block-init')
  const blockBindGroup = root.createBindGroup(blockLayout, {
    params: paramsBuffer,
    grayBuffer: gray.buffer,
    blocks: blockStatsBuffer,
  })

  const componentInitLayout = tgpu
    .bindGroupLayout({
      blocks: { storage: d.arrayOf(BlockStats, blockCount), access: 'readonly' },
      components: { storage: d.arrayOf(ComponentStatsAtomic, blockCount), access: 'mutable' },
      parent: { storage: d.arrayOf(d.atomic(d.u32), blockCount), access: 'mutable' },
      locks: { storage: d.arrayOf(d.atomic(d.u32), blockCount), access: 'mutable' },
      counter: { storage: d.atomic(d.u32), access: 'mutable' },
    })
    .$name('hex-debug-component-init')
  const componentInitPipeline = root
    .createComputePipeline({
      compute: tgpu.computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [WORKGROUP_SIZE, 1, 1],
      })((input) => {
        'use gpu'
        const blockId = d.u32(input.gid.x)
        if (blockId >= d.u32(blockCount)) {
          return
        }
        const b = componentInitLayout.$.blocks[blockId]!
        const c = componentInitLayout.$.components[blockId]!
        atomicStore(c.count, b.count)
        atomicStore(c.sumGray, b.sumGray)
        atomicStore(c.sumGraySq, b.sumGraySq)
        atomicStore(c.sumX, b.sumX)
        atomicStore(c.sumY, b.sumY)
        atomicStore(c.minX, b.minX)
        atomicStore(c.maxX, b.maxX)
        atomicStore(c.minY, b.minY)
        atomicStore(c.maxY, b.maxY)
        atomicStore(componentInitLayout.$.parent[blockId]!, blockId)
        atomicStore(componentInitLayout.$.locks[blockId]!, d.u32(0))
        if (blockId === d.u32(0)) {
          atomicStore(componentInitLayout.$.counter, d.u32(0))
        }
      }),
    })
    .$name('hex-debug-component-init')
  const componentInitBindGroup = root.createBindGroup(componentInitLayout, {
    blocks: blockStatsBuffer,
    components: componentStatsBuffer,
    parent: parentBuffer,
    locks: locksBuffer,
    counter: componentCounter,
  })

  const mergeLayout = tgpu
    .bindGroupLayout({
      params: { uniform: DebugParams },
      components: { storage: d.arrayOf(ComponentStatsAtomic, blockCount), access: 'mutable' },
      parent: { storage: d.arrayOf(d.atomic(d.u32), blockCount), access: 'mutable' },
      locks: { storage: d.arrayOf(d.atomic(d.u32), blockCount), access: 'mutable' },
    })
    .$name('hex-debug-component-merge')
  const mergePipeline = root
    .createComputePipeline({
      compute: tgpu.computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [WORKGROUP_SIZE, 1, 1],
      })((input) => {
        'use gpu'
        const blockId = d.u32(input.gid.x)
        if (blockId >= d.u32(blockCount)) {
          return
        }
        const p = mergeLayout.$.params
        const row = d.u32(floor(d.f32(blockId) / d.f32(p.blockCols)))
        const col = blockId - row * p.blockCols
        let rootA = blockId
        for (let step = d.u32(0); step < d.u32(64); step = step + d.u32(1)) {
          const next = atomicLoad(mergeLayout.$.parent[rootA]!)
          if (next === rootA) {
            break
          }
          rootA = next
        }
        const a = mergeLayout.$.components[rootA]!
        const countA = atomicLoad(a.count)
        if (countA === d.u32(0)) {
          return
        }
        const meanA = d.f32(atomicLoad(a.sumGray)) / (d.f32(countA) * d.f32(GRAY_SCALE))
        const varianceA = max(
          d.f32(0),
          d.f32(atomicLoad(a.sumGraySq)) / (d.f32(countA) * d.f32(GRAY_SCALE * GRAY_SCALE)) - meanA * meanA,
        )

        for (let direction = d.u32(0); direction < d.u32(4); direction = direction + d.u32(1)) {
          let neighborCol = col
          let neighborRow = row
          if (direction === d.u32(0)) {
            neighborCol = col + d.u32(1)
          } else if (direction === d.u32(1)) {
            neighborRow = row + d.u32(1)
          } else if (direction === d.u32(2)) {
            if (col === d.u32(0)) {
              continue
            }
            neighborCol = col - d.u32(1)
            neighborRow = row + d.u32(1)
          } else {
            neighborCol = col + d.u32(1)
            neighborRow = row + d.u32(1)
          }
          if (neighborCol >= p.blockCols || neighborRow >= p.blockRows) {
            continue
          }
          const neighbor = neighborRow * p.blockCols + neighborCol
          let rootB = neighbor
          for (let step = d.u32(0); step < d.u32(64); step = step + d.u32(1)) {
            const next = atomicLoad(mergeLayout.$.parent[rootB]!)
            if (next === rootB) {
              break
            }
            rootB = next
          }
          if (rootA === rootB) {
            continue
          }
          const b = mergeLayout.$.components[rootB]!
          const countB0 = atomicLoad(b.count)
          if (countB0 === d.u32(0)) {
            continue
          }
          const meanB0 = d.f32(atomicLoad(b.sumGray)) / (d.f32(countB0) * d.f32(GRAY_SCALE))
          const varianceB0 = max(
            d.f32(0),
            d.f32(atomicLoad(b.sumGraySq)) / (d.f32(countB0) * d.f32(GRAY_SCALE * GRAY_SCALE)) - meanB0 * meanB0,
          )
          const tolerance0 = min(
            p.maxTolerance,
            p.baseTolerance + p.z * sqrt(max(min(varianceA, varianceB0), d.f32(0))),
          )
          if (abs(meanA - meanB0) > tolerance0) {
            continue
          }

          const lo = min(rootA, rootB)
          const hi = max(rootA, rootB)
          const lockLo = atomicCompareExchangeWeak(mergeLayout.$.locks[lo]!, d.u32(0), d.u32(1))
          if (!lockLo.exchanged) {
            continue
          }
          const lockHi = atomicCompareExchangeWeak(mergeLayout.$.locks[hi]!, d.u32(0), d.u32(1))
          if (!lockHi.exchanged) {
            atomicStore(mergeLayout.$.locks[lo]!, d.u32(0))
            continue
          }
          if (atomicLoad(mergeLayout.$.parent[lo]!) !== lo || atomicLoad(mergeLayout.$.parent[hi]!) !== hi) {
            atomicStore(mergeLayout.$.locks[hi]!, d.u32(0))
            atomicStore(mergeLayout.$.locks[lo]!, d.u32(0))
            continue
          }

          const cA = mergeLayout.$.components[lo]!
          const cB = mergeLayout.$.components[hi]!
          const countA2 = atomicLoad(cA.count)
          const countB2 = atomicLoad(cB.count)
          if (countB2 === d.u32(0)) {
            atomicStore(mergeLayout.$.locks[hi]!, d.u32(0))
            atomicStore(mergeLayout.$.locks[lo]!, d.u32(0))
            continue
          }
          const meanA2 = d.f32(atomicLoad(cA.sumGray)) / (d.f32(countA2) * d.f32(GRAY_SCALE))
          const meanB2 = d.f32(atomicLoad(cB.sumGray)) / (d.f32(countB2) * d.f32(GRAY_SCALE))
          const varianceA2 = max(
            d.f32(0),
            d.f32(atomicLoad(cA.sumGraySq)) / (d.f32(countA2) * d.f32(GRAY_SCALE * GRAY_SCALE)) - meanA2 * meanA2,
          )
          const varianceB2 = max(
            d.f32(0),
            d.f32(atomicLoad(cB.sumGraySq)) / (d.f32(countB2) * d.f32(GRAY_SCALE * GRAY_SCALE)) - meanB2 * meanB2,
          )
          const tolerance2 = min(
            p.maxTolerance,
            p.baseTolerance + p.z * sqrt(max(min(varianceA2, varianceB2), d.f32(0))),
          )
          if (abs(meanA2 - meanB2) > tolerance2) {
            atomicStore(mergeLayout.$.locks[hi]!, d.u32(0))
            atomicStore(mergeLayout.$.locks[lo]!, d.u32(0))
            continue
          }

          atomicAdd(cA.count, countB2)
          atomicAdd(cA.sumGray, atomicLoad(cB.sumGray))
          atomicAdd(cA.sumGraySq, atomicLoad(cB.sumGraySq))
          atomicAdd(cA.sumX, atomicLoad(cB.sumX))
          atomicAdd(cA.sumY, atomicLoad(cB.sumY))
          atomicMin(cA.minX, atomicLoad(cB.minX))
          atomicMax(cA.maxX, atomicLoad(cB.maxX))
          atomicMin(cA.minY, atomicLoad(cB.minY))
          atomicMax(cA.maxY, atomicLoad(cB.maxY))
          atomicStore(mergeLayout.$.parent[hi]!, lo)
          atomicStore(mergeLayout.$.locks[hi]!, d.u32(0))
          atomicStore(mergeLayout.$.locks[lo]!, d.u32(0))
        }
      }),
    })
    .$name('hex-debug-component-merge')
  const mergeBindGroup = root.createBindGroup(mergeLayout, {
    params: paramsBuffer,
    components: componentStatsBuffer,
    parent: parentBuffer,
    locks: locksBuffer,
  })

  const compactLayout = tgpu
    .bindGroupLayout({
      params: { uniform: DebugParams },
      components: { storage: d.arrayOf(ComponentStatsReadonly, blockCount), access: 'readonly' },
      roots: { storage: d.arrayOf(d.u32, blockCount), access: 'readonly' },
      grayBuffer: { storage: d.arrayOf(d.f32, blockCount), access: 'readonly' },
      counter: { storage: d.atomic(d.u32), access: 'mutable' },
      rows: { storage: d.arrayOf(ComponentRow, MAX_COMPONENTS), access: 'mutable' },
    })
    .$name('hex-debug-component-compact')
  const compactPipeline = root
    .createComputePipeline({
      compute: tgpu.computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [WORKGROUP_SIZE, 1, 1],
      })((input) => {
        'use gpu'
        const blockId = d.u32(input.gid.x)
        if (blockId >= d.u32(blockCount)) {
          return
        }
        const p = compactLayout.$.params
        if (compactLayout.$.roots[blockId] !== blockId) {
          return
        }
        const c = compactLayout.$.components[blockId]!
        if (c.count < p.minArea || c.count > p.maxArea) {
          return
        }

        const bboxWidth = c.maxX - c.minX + d.u32(1)
        const bboxHeight = c.maxY - c.minY + d.u32(1)
        const fillRatio = d.f32(c.count) / d.f32(max(bboxWidth * bboxHeight, d.u32(1)))
        let rejection = d.u32(0)
        let ringMean = d.f32(0)
        let ringVariance = d.f32(0)

        if (fillRatio * d.f32(100) < d.f32(p.minFillPercent)) {
          rejection = d.u32(1)
        } else {
          let ringCount = d.u32(0)
          let ringSum = d.u32(0)
          let ringSumSq = d.u32(0)
          const margin = p.ringMargin
          const left = c.minX > margin ? c.minX - margin : d.u32(0)
          const right = min(p.width - d.u32(1), c.maxX + margin)
          const top = c.minY > margin ? c.minY - margin : d.u32(0)
          const bottom = min(p.height - d.u32(1), c.maxY + margin)

          for (let x = left; x <= right; x = x + d.u32(1)) {
            const topIdx = top * p.width + x
            if (compactLayout.$.roots[topIdx] !== blockId) {
              const q = min(
                d.u32(127),
                max(d.u32(0), d.u32(round(compactLayout.$.grayBuffer[topIdx]! * d.f32(GRAY_SCALE)))),
              )
              ringCount = ringCount + d.u32(1)
              ringSum = ringSum + q
              ringSumSq = ringSumSq + q * q
            }
            const bottomIdx = bottom * p.width + x
            if (compactLayout.$.roots[bottomIdx] !== blockId) {
              const q = min(
                d.u32(127),
                max(d.u32(0), d.u32(round(compactLayout.$.grayBuffer[bottomIdx]! * d.f32(GRAY_SCALE)))),
              )
              ringCount = ringCount + d.u32(1)
              ringSum = ringSum + q
              ringSumSq = ringSumSq + q * q
            }
          }
          for (let y = top + d.u32(1); y < bottom; y = y + d.u32(1)) {
            const leftIdx = y * p.width + left
            if (compactLayout.$.roots[leftIdx] !== blockId) {
              const q = min(
                d.u32(127),
                max(d.u32(0), d.u32(round(compactLayout.$.grayBuffer[leftIdx]! * d.f32(GRAY_SCALE)))),
              )
              ringCount = ringCount + d.u32(1)
              ringSum = ringSum + q
              ringSumSq = ringSumSq + q * q
            }
            const rightIdx = y * p.width + right
            if (compactLayout.$.roots[rightIdx] !== blockId) {
              const q = min(
                d.u32(127),
                max(d.u32(0), d.u32(round(compactLayout.$.grayBuffer[rightIdx]! * d.f32(GRAY_SCALE)))),
              )
              ringCount = ringCount + d.u32(1)
              ringSum = ringSum + q
              ringSumSq = ringSumSq + q * q
            }
          }

          if (ringCount === d.u32(0)) {
            rejection = d.u32(4)
          } else {
            ringMean = d.f32(ringSum) / (d.f32(ringCount) * d.f32(GRAY_SCALE))
            const ringSecondMoment = d.f32(ringSumSq) / (d.f32(ringCount) * d.f32(GRAY_SCALE * GRAY_SCALE))
            ringVariance = max(d.f32(0), ringSecondMoment - ringMean * ringMean)
            const componentMean = d.f32(c.sumGray) / (d.f32(c.count) * d.f32(GRAY_SCALE))
            if (ringVariance > p.maxRingVariance) {
              rejection = d.u32(2)
            } else if (abs(componentMean - ringMean) < p.minContrast) {
              rejection = d.u32(3)
            }
          }
        }

        const index = atomicAdd(compactLayout.$.counter, d.u32(1))
        if (index >= d.u32(MAX_COMPONENTS)) {
          return
        }
        compactLayout.$.rows[index] = ComponentRow({
          label: blockId,
          count: c.count,
          sumGray: c.sumGray,
          sumGraySq: c.sumGraySq,
          sumX: c.sumX,
          sumY: c.sumY,
          minX: c.minX,
          maxX: c.maxX,
          minY: c.minY,
          maxY: c.maxY,
          rejection,
          fillRatio,
          ringMean,
          ringVariance,
        })
      }),
    })
    .$name('hex-debug-component-compact')
  const compactBindGroup = root.createBindGroup(compactLayout, {
    params: paramsBuffer,
    components: componentStatsBuffer,
    roots: rootsA,
    grayBuffer: gray.buffer,
    counter: componentCounter,
    rows: componentRowsBuffer,
  })

  const boxClearLayout = tgpu
    .bindGroupLayout({ mask: { storage: d.arrayOf(d.u32, blockCount), access: 'mutable' } })
    .$name('hex-debug-box-clear')
  const boxClearPipeline = root
    .createComputePipeline({
      compute: tgpu.computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [WORKGROUP_SIZE, 1, 1],
      })((input) => {
        'use gpu'
        const index = d.u32(input.gid.x)
        if (index < d.u32(blockCount)) {
          boxClearLayout.$.mask[index] = d.u32(0)
        }
      }),
    })
    .$name('hex-debug-box-clear')
  const boxDrawLayout = tgpu
    .bindGroupLayout({
      params: { uniform: BoxParams },
      rows: { storage: d.arrayOf(ComponentRow, MAX_COMPONENTS), access: 'readonly' },
      mask: { storage: d.arrayOf(d.u32, blockCount), access: 'mutable' },
    })
    .$name('hex-debug-box-draw')
  const boxDrawPipeline = root
    .createComputePipeline({
      compute: tgpu.computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [WORKGROUP_SIZE, 1, 1],
      })((input) => {
        'use gpu'
        const index = d.u32(input.gid.x)
        const p = boxDrawLayout.$.params
        if (index >= p.activeCount || index >= d.u32(MAX_COMPONENTS)) {
          return
        }
        const row = boxDrawLayout.$.rows[index]!
        if (row.rejection !== d.u32(0) && p.showRejections === d.u32(0)) {
          return
        }
        const code = select(d.u32(1), row.rejection + d.u32(1), row.rejection !== d.u32(0))
        const left = row.minX
        const right = row.maxX
        const top = row.minY
        const bottom = row.maxY
        if (row.rejection !== d.u32(0)) {
          for (let y = top; y <= bottom; y = y + d.u32(1)) {
            for (let x = left; x <= right; x = x + d.u32(1)) {
              if (x < p.width && y < p.height) {
                boxDrawLayout.$.mask[y * p.width + x] = code
              }
            }
          }
        } else {
          for (let x = left; x <= right; x = x + d.u32(1)) {
            if (x < p.width) {
              boxDrawLayout.$.mask[top * p.width + x] = code
              boxDrawLayout.$.mask[bottom * p.width + x] = code
            }
          }
          for (let y = top + d.u32(1); y < bottom; y = y + d.u32(1)) {
            if (y < p.height) {
              boxDrawLayout.$.mask[y * p.width + left] = code
              boxDrawLayout.$.mask[y * p.width + right] = code
            }
          }
        }
      }),
    })
    .$name('hex-debug-box-draw')
  const boxClearBindGroup = root.createBindGroup(boxClearLayout, { mask: boxMaskBuffer })
  const boxDrawBindGroup = root.createBindGroup(boxDrawLayout, {
    params: boxParamsBuffer,
    rows: componentRowsBuffer,
    mask: boxMaskBuffer,
  })

  const renderLayout = tgpu
    .bindGroupLayout({
      params: { uniform: RenderParams },
      grayBuffer: { storage: d.arrayOf(d.f32, blockCount), access: 'readonly' },
      labels: { storage: d.arrayOf(d.u32, blockCount), access: 'readonly' },
      doh: { storage: d.arrayOf(d.f32, blockCount), access: 'readonly' },
      dohMax: { storage: d.u32, access: 'readonly' },
      candidateMask: { storage: d.arrayOf(d.u32, blockCount), access: 'readonly' },
      boxMask: { storage: d.arrayOf(d.u32, blockCount), access: 'readonly' },
    })
    .$name('hex-debug-present')
  const linearToSrgb = (value: number): number => {
    'use gpu'
    return select(
      value * d.f32(12.92),
      d.f32(1.055) * pow(max(value, d.f32(0)), d.f32(1 / 2.4)) - d.f32(0.055),
      value > d.f32(0.0031308),
    )
  }

  const renderFragment = tgpu.fragmentFn({
    in: { position: d.builtin.position },
    out: d.vec4f,
  })((input) => {
    'use gpu'
    const p = renderLayout.$.params
    const x = d.u32(input.position.x)
    const y = d.u32(input.position.y)
    if (x >= p.width || y >= p.height) {
      return d.vec4f(0, 0, 0, 1)
    }
    const index = y * p.width + x
    if (p.mode === d.u32(0)) {
      const grayValue = linearToSrgb(renderLayout.$.grayBuffer[index]!)
      return d.vec4f(grayValue, grayValue, grayValue, d.f32(1))
    }
    if (p.mode === d.u32(1)) {
      const label = renderLayout.$.labels[index]!
      const hash = (label * d.u32(2654435761)) & d.u32(255)
      const color = d.vec3f(
        d.f32(hash & d.u32(127)) / d.f32(127),
        d.f32((hash >>> d.u32(2)) & d.u32(127)) / d.f32(127),
        d.f32((hash >>> d.u32(4)) & d.u32(127)) / d.f32(127),
      )
      return d.vec4f(color, d.f32(1))
    }
    if (p.mode === d.u32(2) || p.mode === d.u32(4)) {
      if (p.mode === d.u32(4) && renderLayout.$.candidateMask[index]! !== d.u32(0)) {
        return d.vec4f(d.f32(0.13), d.f32(0.95), d.f32(0.45), d.f32(1))
      }
      const response = renderLayout.$.doh[index]!
      const maxAbs = max(bitcastU32toF32(renderLayout.$.dohMax), d.f32(0.000001))
      const signed = max(d.f32(-1), min(d.f32(1), response / maxAbs))
      const magnitude = sqrt(abs(signed))
      if (signed >= d.f32(0)) {
        return d.vec4f(magnitude, d.f32(0.667) * magnitude * magnitude, d.f32(0.157) * (d.f32(1) - magnitude), d.f32(1))
      }
      return d.vec4f(d.f32(0.078) * (d.f32(1) - magnitude), d.f32(0.275) * magnitude, magnitude, d.f32(1))
    }
    const grayValue = linearToSrgb(renderLayout.$.grayBuffer[index]!)
    const mask = renderLayout.$.boxMask[index]!
    if (mask === d.u32(1)) {
      return d.vec4f(d.f32(0.13), d.f32(0.8), d.f32(0.35), d.f32(1))
    }
    if (mask === d.u32(2)) {
      return d.vec4f(d.f32(0.98), d.f32(0.45), d.f32(0.08), d.f32(1))
    }
    if (mask === d.u32(3)) {
      return d.vec4f(d.f32(0.94), d.f32(0.27), d.f32(0.27), d.f32(1))
    }
    if (mask === d.u32(4)) {
      return d.vec4f(d.f32(0.23), d.f32(0.51), d.f32(0.96), d.f32(1))
    }
    if (mask === d.u32(5)) {
      return d.vec4f(d.f32(0.92), d.f32(0.7), d.f32(0.03), d.f32(1))
    }
    return d.vec4f(grayValue, grayValue, grayValue, d.f32(1))
  })
  const renderPipeline = root
    .createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: renderFragment,
      targets: { format: presentationFormat },
    })
    .$name('hex-debug-present')
  const renderBindGroup = root.createBindGroup(renderLayout, {
    params: renderParamsBuffer,
    grayBuffer: gray.buffer,
    labels: rootsA,
    doh: dohResponseBuffer,
    dohMax: dohMaxBuffer,
    candidateMask: dohStage.candidateMaskBuffer,
    boxMask: boxMaskBuffer,
  })

  const jumpLayouts = createPointerJumpLayouts()
  const copyParentToRootsPipeline = createPointerJumpAtomicToLabelsPipeline(
    root,
    jumpLayouts.atomicToLabelsLayout,
    width,
    height,
  )
  const jumpStepPipeline = createPointerJumpStepPipeline(root, jumpLayouts.stepLayout, width, height)
  const copyParentToRootsBindGroup = root.createBindGroup(jumpLayouts.atomicToLabelsLayout, {
    atomicLabels: parentBuffer,
    dest: rootsA,
  })
  const jumpStepAB = root.createBindGroup(jumpLayouts.stepLayout, {
    readBuffer: rootsA,
    writeBuffer: rootsB,
  })
  const jumpStepBA = root.createBindGroup(jumpLayouts.stepLayout, {
    readBuffer: rootsB,
    writeBuffer: rootsA,
  })
  const jumpWgX = Math.ceil(width / 16)
  const jumpWgY = Math.ceil(height / 16)

  const dispatch = (
    enc: GPUCommandEncoder,
    pipeline: ReturnType<TgpuRoot['createComputePipeline']>,
    bindGroup: ReturnType<TgpuRoot['createBindGroup']>,
    count: number,
    label: string,
  ): void => {
    const pass = enc.beginComputePass({ label })
    pipeline
      .with(pass)
      .with(bindGroup)
      .dispatchWorkgroups(Math.ceil(Math.max(1, count) / WORKGROUP_SIZE))
    pass.end()
  }

  const resolveRoots = (enc: GPUCommandEncoder): void => {
    const copyPass = enc.beginComputePass({ label: 'hex-debug-copy-parent-roots' })
    copyParentToRootsPipeline.with(copyPass).with(copyParentToRootsBindGroup).dispatchWorkgroups(jumpWgX, jumpWgY)
    copyPass.end()
    for (let step = 0; step < ROOT_JUMP_ITERATIONS; step += 1) {
      const jumpPass = enc.beginComputePass({ label: 'hex-debug-jump-roots' })
      jumpStepPipeline
        .with(jumpPass)
        .with(step % 2 === 0 ? jumpStepAB : jumpStepBA)
        .dispatchWorkgroups(jumpWgX, jumpWgY)
      jumpPass.end()
    }
  }

  const encode = (
    enc: GPUCommandEncoder,
    video: HTMLVideoElement,
    tuning: HexDebugTuning,
    options?: { doh?: boolean },
  ): void => {
    currentTuning = { ...tuning }
    lastIterations = tuning.iterations
    writeParams()
    ingest.encodeIngest(enc, root, video)
    const grayPass = enc.beginComputePass({ label: 'hex-debug-gray' })
    gray.encodeCompute(grayPass)
    grayPass.end()
    dispatch(enc, blockPipeline, blockBindGroup, blockCount, 'hex-debug-blocks')
    dispatch(enc, componentInitPipeline, componentInitBindGroup, blockCount, 'hex-debug-components-init')
    for (let iteration = 0; iteration < tuning.iterations; iteration += 1) {
      dispatch(enc, mergePipeline, mergeBindGroup, blockCount, 'hex-debug-merge')
    }
    resolveRoots(enc)
    dispatch(enc, compactPipeline, compactBindGroup, blockCount, 'hex-debug-compact')
    if (options?.doh) {
      dohStage.encode(enc)
    }
  }

  const setNmsThreshold = (threshold: number): void => {
    dohStage.setNmsThreshold(threshold)
  }

  const render = (
    enc: GPUCommandEncoder,
    view: GPUTextureView,
    mode: 'gray' | 'components' | 'doh' | 'boxes' | 'nms',
    activeCount: number,
    showRejections: boolean,
  ): void => {
    const modeCode = mode === 'gray' ? 0 : mode === 'components' ? 1 : mode === 'doh' ? 2 : mode === 'nms' ? 4 : 3
    renderParamsBuffer.write({ width, height, mode: modeCode })
    if (mode === 'boxes') {
      boxParamsBuffer.write({
        width,
        height,
        activeCount: Math.min(Math.max(0, activeCount), MAX_COMPONENTS),
        showRejections: showRejections ? 1 : 0,
      })
      dispatch(enc, boxClearPipeline, boxClearBindGroup, blockCount, 'hex-debug-box-clear')
      const activeWg = Math.ceil(Math.max(1, Math.min(activeCount, MAX_COMPONENTS)) / WORKGROUP_SIZE)
      const boxPass = enc.beginComputePass({ label: 'hex-debug-box-draw' })
      boxDrawPipeline.with(boxPass).with(boxDrawBindGroup).dispatchWorkgroups(activeWg)
      boxPass.end()
    }
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view, clearValue: [0, 0, 0, 1], loadOp: 'clear', storeOp: 'store' }],
    })
    renderPipeline.with(pass).with(renderBindGroup).draw(3)
    pass.end()
  }

  const readComponents = async (): Promise<HexDebugFrame> => {
    const [counterRaw, rowsRaw] = await Promise.all([componentCounter.read(), componentRowsBuffer.read()])
    const countValue = Array.isArray(counterRaw) ? Number(counterRaw[0] ?? 0) : Number(counterRaw)
    const rows = (Array.isArray(rowsRaw) ? rowsRaw : []).slice(0, Math.min(countValue, MAX_COMPONENTS))
    const components = rows.map((row) => toComponent(row as d.Infer<typeof ComponentRow>))
    components.sort((first, second) => second.area - first.area)
    return {
      width,
      height,
      blockSize: 1,
      components,
      iterations: lastIterations,
      elapsedMs: 0,
    }
  }

  return {
    encode,
    readComponents,
    render,
    setNmsThreshold,
    destroy() {
      blockStatsBuffer.destroy()
      componentStatsBuffer.destroy()
      parentBuffer.destroy()
      locksBuffer.destroy()
      rootsA.destroy()
      rootsB.destroy()
      componentCounter.destroy()
      componentRowsBuffer.destroy()
      dohStage.destroy()
      boxMaskBuffer.destroy()
      boxParamsBuffer.destroy()
      renderParamsBuffer.destroy()
      paramsBuffer.destroy()
      gray.buffer.destroy()
      ingest.grayTex.destroy()
    },
  }
}
