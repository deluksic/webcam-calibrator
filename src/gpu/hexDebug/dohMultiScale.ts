import type { StorageFlag, TgpuBuffer, TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import { abs, atomicMax, atomicStore, bitcastF32toU32, bitcastU32toF32, floor, max } from 'typegpu/std'

/** Target scales, chosen to cover perspective variation of the small squares. */
export const DOH_SCALES = [1.5, 2.2, 3.2, 4.6] as const

const WORKGROUP_SIZE = 64

const NmsParams = d.struct({ threshold: d.f32 })

type ScaleKernels = {
  sigma: number
  radius: number
  g: Float32Array
  g1: Float32Array
  g2: Float32Array
}

function makeKernels(sigma: number): ScaleKernels {
  const radius = Math.ceil(3 * sigma)
  const length = radius * 2 + 1
  const g = new Float64Array(length)
  let sum = 0
  for (let i = 0; i < length; i += 1) {
    const x = i - radius
    const value = Math.exp(-(x * x) / (2 * sigma * sigma))
    g[i] = value
    sum += value
  }
  for (let i = 0; i < length; i += 1) {
    g[i] = (g[i] ?? 0) / sum
  }
  const g1 = new Float32Array(length)
  const g2 = new Float32Array(length)
  const sigma2 = sigma * sigma
  const sigma4 = sigma2 * sigma2
  for (let i = 0; i < length; i += 1) {
    const x = i - radius
    const base = g[i] ?? 0
    g1[i] = (-x / sigma2) * base
    g2[i] = ((x * x - sigma2) / sigma4) * base
  }
  return { sigma, radius, g: Float32Array.from(g), g1, g2 }
}

export type MultiScaleDohStage = {
  responseBuffer: TgpuBuffer<d.WgslArray<d.F32>> & StorageFlag
  maxBuffer: TgpuBuffer<d.Atomic<d.U32>> & StorageFlag
  candidateMaskBuffer: TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag
  setNmsThreshold(threshold: number): void
  encode(enc: GPUCommandEncoder): void
  destroy(): void
}

/**
 * Separable multi-scale determinant-of-Hessian response.
 *
 * For every scale:
 *   horizontal: g, g', g'' -> h0, h1, h2
 *   vertical:   h2 * g, h0 * g'', h1 * g' -> Lxx, Lyy, Lxy
 * The scale-normalized determinant is written to a per-scale response buffer.
 */
export function createMultiScaleDohStage(
  root: TgpuRoot,
  width: number,
  height: number,
  grayBuffer: TgpuBuffer<d.WgslArray<d.F32>> & StorageFlag,
): MultiScaleDohStage {
  const count = width * height
  const kernels = DOH_SCALES.map(makeKernels)
  const responseBuffer = root.createBuffer(d.arrayOf(d.f32, count)).$usage('storage')
  const maxBuffer = root.createBuffer(d.atomic(d.u32)).$usage('storage')
  const candidateMaskBuffer = root.createBuffer(d.arrayOf(d.u32, count)).$usage('storage')
  const nmsParamsBuffer = root.createBuffer(NmsParams).$usage('uniform')
  const h0 = root.createBuffer(d.arrayOf(d.f32, count)).$usage('storage')
  const h1 = root.createBuffer(d.arrayOf(d.f32, count)).$usage('storage')
  const h2 = root.createBuffer(d.arrayOf(d.f32, count)).$usage('storage')

  const horizontalLayout = tgpu
    .bindGroupLayout({
      gray: { storage: d.arrayOf(d.f32, count), access: 'readonly' },
      g: { storage: d.arrayOf(d.f32), access: 'readonly' },
      g1: { storage: d.arrayOf(d.f32), access: 'readonly' },
      g2: { storage: d.arrayOf(d.f32), access: 'readonly' },
      h0: { storage: d.arrayOf(d.f32, count), access: 'mutable' },
      h1: { storage: d.arrayOf(d.f32, count), access: 'mutable' },
      h2: { storage: d.arrayOf(d.f32, count), access: 'mutable' },
    })
    .$name('hex-doh-horizontal')

  const verticalLayout = tgpu
    .bindGroupLayout({
      h0: { storage: d.arrayOf(d.f32, count), access: 'readonly' },
      h1: { storage: d.arrayOf(d.f32, count), access: 'readonly' },
      h2: { storage: d.arrayOf(d.f32, count), access: 'readonly' },
      g: { storage: d.arrayOf(d.f32), access: 'readonly' },
      g1: { storage: d.arrayOf(d.f32), access: 'readonly' },
      g2: { storage: d.arrayOf(d.f32), access: 'readonly' },
      response: { storage: d.arrayOf(d.f32, count), access: 'mutable' },
    })
    .$name('hex-doh-vertical')

  const scales = kernels.map((kernel) => {
    const g = root.createBuffer(d.arrayOf(d.f32, kernel.g.length)).$usage('storage')
    const g1 = root.createBuffer(d.arrayOf(d.f32, kernel.g1.length)).$usage('storage')
    const g2 = root.createBuffer(d.arrayOf(d.f32, kernel.g2.length)).$usage('storage')
    g.write(kernel.g)
    g1.write(kernel.g1)
    g2.write(kernel.g2)
    const response = root.createBuffer(d.arrayOf(d.f32, count)).$usage('storage')
    const radius = kernel.radius
    const sigma = kernel.sigma

    const horizontalKernel = tgpu.computeFn({
      in: { gid: d.builtin.globalInvocationId },
      workgroupSize: [WORKGROUP_SIZE, 1, 1],
    })((input) => {
      'use gpu'
      const index = d.u32(input.gid.x)
      if (index >= d.u32(count)) {
        return
      }
      const y = d.u32(floor(d.f32(index) / d.f32(width)))
      const x = index - y * d.u32(width)
      let sum0 = d.f32(0)
      let sum1 = d.f32(0)
      let sum2 = d.f32(0)
      for (let offset = -radius; offset <= radius; offset += 1) {
        const sx = d.i32(x) + d.i32(offset)
        if (sx < d.i32(0) || sx >= d.i32(width)) {
          continue
        }
        const value = horizontalLayout.$.gray[d.u32(y) * d.u32(width) + d.u32(sx)]!
        const ki = d.u32(d.i32(offset) + d.i32(radius))
        sum0 = sum0 + value * horizontalLayout.$.g[ki]!
        sum1 = sum1 + value * horizontalLayout.$.g1[ki]!
        sum2 = sum2 + value * horizontalLayout.$.g2[ki]!
      }
      horizontalLayout.$.h0[index] = sum0
      horizontalLayout.$.h1[index] = sum1
      horizontalLayout.$.h2[index] = sum2
    })

    const verticalKernel = tgpu.computeFn({
      in: { gid: d.builtin.globalInvocationId },
      workgroupSize: [WORKGROUP_SIZE, 1, 1],
    })((input) => {
      'use gpu'
      const index = d.u32(input.gid.x)
      if (index >= d.u32(count)) {
        return
      }
      const y = d.u32(floor(d.f32(index) / d.f32(width)))
      const x = index - y * d.u32(width)
      let dxx = d.f32(0)
      let dyy = d.f32(0)
      let dxy = d.f32(0)
      for (let offset = -radius; offset <= radius; offset += 1) {
        const sy = d.i32(y) + d.i32(offset)
        if (sy < d.i32(0) || sy >= d.i32(height)) {
          continue
        }
        const si = d.u32(sy) * d.u32(width) + d.u32(x)
        const ki = d.u32(d.i32(offset) + d.i32(radius))
        const value0 = verticalLayout.$.h0[si]!
        const value1 = verticalLayout.$.h1[si]!
        const value2 = verticalLayout.$.h2[si]!
        dxx = dxx + value2 * verticalLayout.$.g[ki]!
        dyy = dyy + value0 * verticalLayout.$.g2[ki]!
        dxy = dxy + value1 * verticalLayout.$.g1[ki]!
      }
      const sigma4 = sigma * sigma * sigma * sigma
      verticalLayout.$.response[index] = d.f32(sigma4) * (dxx * dyy - dxy * dxy)
    })

    const horizontalPipeline = root.createComputePipeline({ compute: horizontalKernel })
    const verticalPipeline = root.createComputePipeline({ compute: verticalKernel })
    const horizontalBindGroup = root.createBindGroup(horizontalLayout, {
      gray: grayBuffer,
      g,
      g1,
      g2,
      h0,
      h1,
      h2,
    })
    const verticalBindGroup = root.createBindGroup(verticalLayout, {
      h0,
      h1,
      h2,
      g,
      g1,
      g2,
      response,
    })

    return {
      g,
      g1,
      g2,
      response,
      horizontalPipeline,
      verticalPipeline,
      horizontalBindGroup,
      verticalBindGroup,
    }
  })

  const combineLayout = tgpu
    .bindGroupLayout({
      r0: { storage: d.arrayOf(d.f32, count), access: 'readonly' },
      r1: { storage: d.arrayOf(d.f32, count), access: 'readonly' },
      r2: { storage: d.arrayOf(d.f32, count), access: 'readonly' },
      r3: { storage: d.arrayOf(d.f32, count), access: 'readonly' },
      best: { storage: d.arrayOf(d.f32, count), access: 'mutable' },
      maxResponse: { storage: d.atomic(d.u32), access: 'mutable' },
    })
    .$name('hex-doh-combine')

  const combinePipeline = root
    .createComputePipeline({
      compute: tgpu.computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [WORKGROUP_SIZE, 1, 1],
      })((input) => {
        'use gpu'
        const index = d.u32(input.gid.x)
        if (index >= d.u32(count)) {
          return
        }
        let best = combineLayout.$.r0[index]!
        let bestAbs = abs(best)
        const r1 = combineLayout.$.r1[index]!
        if (abs(r1) > bestAbs) {
          best = r1
          bestAbs = abs(r1)
        }
        const r2 = combineLayout.$.r2[index]!
        if (abs(r2) > bestAbs) {
          best = r2
          bestAbs = abs(r2)
        }
        const r3 = combineLayout.$.r3[index]!
        if (abs(r3) > bestAbs) {
          best = r3
          bestAbs = abs(r3)
        }
        combineLayout.$.best[index] = best
        atomicMax(combineLayout.$.maxResponse, bitcastF32toU32(bestAbs))
      }),
    })
    .$name('hex-doh-combine')

  const combineBindGroup = root.createBindGroup(combineLayout, {
    r0: scales[0]!.response,
    r1: scales[1]!.response,
    r2: scales[2]!.response,
    r3: scales[3]!.response,
    best: responseBuffer,
    maxResponse: maxBuffer,
  })

  const nmsLayout = tgpu
    .bindGroupLayout({
      current: { storage: d.arrayOf(d.f32, count), access: 'readonly' },
      below: { storage: d.arrayOf(d.f32, count), access: 'readonly' },
      above: { storage: d.arrayOf(d.f32, count), access: 'readonly' },
      maxBits: { storage: d.u32, access: 'readonly' },
      mask: { storage: d.arrayOf(d.u32, count), access: 'mutable' },
      params: { uniform: NmsParams },
    })
    .$name('hex-doh-nms')

  const nmsScales = scales.map((scale, scaleIndex) => {
    const hasBelow = scaleIndex > 0
    const hasAbove = scaleIndex < scales.length - 1
    const code = d.u32(scaleIndex + 1)
    const pipeline = root.createComputePipeline({
      compute: tgpu.computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [WORKGROUP_SIZE, 1, 1],
      })((input) => {
        'use gpu'
        const index = d.u32(input.gid.x)
        if (index >= d.u32(count)) {
          return
        }
        const y = d.u32(floor(d.f32(index) / d.f32(width)))
        const x = d.u32(index) - y * d.u32(width)
        if (x >= d.u32(width) || y >= d.u32(height)) {
          return
        }
        const response = nmsLayout.$.current[index]!
        const maxAbs = max(bitcastU32toF32(nmsLayout.$.maxBits), d.f32(0.000001))
        if (response <= d.f32(0) || response / maxAbs < nmsLayout.$.params.threshold) {
          return
        }
        for (let dy = -1; dy <= 1; dy += 1) {
          for (let dx = -1; dx <= 1; dx += 1) {
            if (dx === 0 && dy === 0) {
              continue
            }
            const nx = d.i32(x) + d.i32(dx)
            const ny = d.i32(y) + d.i32(dy)
            if (nx < d.i32(0) || nx >= d.i32(width) || ny < d.i32(0) || ny >= d.i32(height)) {
              continue
            }
            if (nmsLayout.$.current[d.u32(ny) * d.u32(width) + d.u32(nx)]! > response) {
              return
            }
          }
        }
        if (hasBelow) {
          for (let dy = -1; dy <= 1; dy += 1) {
            for (let dx = -1; dx <= 1; dx += 1) {
              const nx = d.i32(x) + d.i32(dx)
              const ny = d.i32(y) + d.i32(dy)
              if (nx < d.i32(0) || nx >= d.i32(width) || ny < d.i32(0) || ny >= d.i32(height)) {
                continue
              }
              if (nmsLayout.$.below[d.u32(ny) * d.u32(width) + d.u32(nx)]! > response) {
                return
              }
            }
          }
        }
        if (hasAbove) {
          for (let dy = -1; dy <= 1; dy += 1) {
            for (let dx = -1; dx <= 1; dx += 1) {
              const nx = d.i32(x) + d.i32(dx)
              const ny = d.i32(y) + d.i32(dy)
              if (nx < d.i32(0) || nx >= d.i32(width) || ny < d.i32(0) || ny >= d.i32(height)) {
                continue
              }
              if (nmsLayout.$.above[d.u32(ny) * d.u32(width) + d.u32(nx)]! > response) {
                return
              }
            }
          }
        }
        nmsLayout.$.mask[index] = code
      }),
    })
    const bindGroup = root.createBindGroup(nmsLayout, {
      current: scale.response,
      below: scales[Math.max(0, scaleIndex - 1)]!.response,
      above: scales[Math.min(scales.length - 1, scaleIndex + 1)]!.response,
      maxBits: maxBuffer,
      mask: candidateMaskBuffer,
      params: nmsParamsBuffer,
    })
    return { pipeline, bindGroup }
  })

  const maskClearLayout = tgpu
    .bindGroupLayout({ mask: { storage: d.arrayOf(d.u32, count), access: 'mutable' } })
    .$name('hex-doh-mask-clear')
  const maskClearPipeline = root
    .createComputePipeline({
      compute: tgpu.computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [WORKGROUP_SIZE, 1, 1],
      })((input) => {
        'use gpu'
        const index = d.u32(input.gid.x)
        if (index < d.u32(count)) {
          maskClearLayout.$.mask[index] = d.u32(0)
        }
      }),
    })
    .$name('hex-doh-mask-clear')
  const maskClearBindGroup = root.createBindGroup(maskClearLayout, { mask: candidateMaskBuffer })

  const clearLayout = tgpu
    .bindGroupLayout({ maxResponse: { storage: d.atomic(d.u32), access: 'mutable' } })
    .$name('hex-doh-clear')
  const clearPipeline = root
    .createComputePipeline({
      compute: tgpu.computeFn({
        in: { gid: d.builtin.globalInvocationId },
        workgroupSize: [1, 1, 1],
      })(() => {
        'use gpu'
        atomicStore(clearLayout.$.maxResponse, d.u32(0))
      }),
    })
    .$name('hex-doh-clear')
  const clearBindGroup = root.createBindGroup(clearLayout, { maxResponse: maxBuffer })

  const setNmsThreshold = (threshold: number): void => {
    nmsParamsBuffer.write({ threshold: Math.max(0, Math.min(1, threshold)) })
  }
  setNmsThreshold(0.08)

  const wg = Math.ceil(count / WORKGROUP_SIZE)

  const encode = (enc: GPUCommandEncoder): void => {
    const clearPass = enc.beginComputePass({ label: 'hex-doh-clear' })
    clearPipeline.with(clearPass).with(clearBindGroup).dispatchWorkgroups(1)
    clearPass.end()

    for (const scale of scales) {
      const horizontalPass = enc.beginComputePass({ label: 'hex-doh-horizontal' })
      scale.horizontalPipeline.with(horizontalPass).with(scale.horizontalBindGroup).dispatchWorkgroups(wg)
      horizontalPass.end()
      const verticalPass = enc.beginComputePass({ label: 'hex-doh-vertical' })
      scale.verticalPipeline.with(verticalPass).with(scale.verticalBindGroup).dispatchWorkgroups(wg)
      verticalPass.end()
    }

    const combinePass = enc.beginComputePass({ label: 'hex-doh-combine' })
    combinePipeline.with(combinePass).with(combineBindGroup).dispatchWorkgroups(wg)
    combinePass.end()

    const maskClearPass = enc.beginComputePass({ label: 'hex-doh-nms-clear' })
    maskClearPipeline.with(maskClearPass).with(maskClearBindGroup).dispatchWorkgroups(wg)
    maskClearPass.end()
    for (const scale of nmsScales) {
      const nmsPass = enc.beginComputePass({ label: 'hex-doh-nms' })
      scale.pipeline.with(nmsPass).with(scale.bindGroup).dispatchWorkgroups(wg)
      nmsPass.end()
    }
  }

  return {
    responseBuffer,
    maxBuffer,
    candidateMaskBuffer,
    setNmsThreshold,
    encode,
    destroy() {
      responseBuffer.destroy()
      maxBuffer.destroy()
      candidateMaskBuffer.destroy()
      nmsParamsBuffer.destroy()
      h0.destroy()
      h1.destroy()
      h2.destroy()
      for (const scale of scales) {
        scale.g.destroy()
        scale.g1.destroy()
        scale.g2.destroy()
        scale.response.destroy()
      }
    },
  }
}
