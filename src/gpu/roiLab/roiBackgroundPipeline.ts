import type { SampledFlag, StorageFlag, TgpuBuffer, TgpuSampler, TgpuTexture, TgpuRoot } from 'typegpu'
import { d, std, tgpu } from 'typegpu'
import { clamp, max, select } from 'typegpu/std'

/**
 * Low-frequency illumination estimate for the ROI.
 *
 * The low-resolution ONNX input is already a downsampled, per-frame-normalized
 * luma image. We blur it with a mask-aware normalized convolution, reconstruct
 * raw luma with the same frame mean/std (read from the GPU stats buffer, so no
 * readback is involved), and expose the result as a sampled texture. The
 * full-resolution component/histogram stages subtract it.
 */
const BLUR_RADIUS = 12
const MASK_WEIGHT_EPSILON = 0.001

const backgroundParams = d.struct({
  size: d.vec2f,
  maskSize: d.vec2f,
})

const weightLayout = tgpu.bindGroupLayout({
  inputBuffer: { storage: d.arrayOf(d.f32), access: 'readonly' },
  /** Frame mean and standard deviation, written by the GPU stats pass. */
  stats: { storage: d.arrayOf(d.f32), access: 'readonly' },
  maskTexture: { texture: d.texture2d() },
  sampler: { sampler: 'filtering' },
  params: { uniform: backgroundParams },
  output: { storage: d.arrayOf(d.vec2f), access: 'mutable' },
})

const blurLayout = tgpu.bindGroupLayout({
  input: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
  params: { uniform: backgroundParams },
  output: { storage: d.arrayOf(d.vec2f), access: 'mutable' },
})

const textureLayout = tgpu.bindGroupLayout({
  input: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
  stats: { storage: d.arrayOf(d.f32), access: 'readonly' },
  params: { uniform: backgroundParams },
  backgroundTexture: {
    storageTexture: d.textureStorage2d('rgba8unorm', 'write-only'),
  },
})

const weightPass = (x: number, y: number) => {
  'use gpu'
  const params = weightLayout.$.params
  const width = d.u32(params.size.x)
  const index = d.u32(y) * width + d.u32(x)
  const uv = (d.vec2f(x, y) + d.vec2f(0.5, 0.5)).div(params.size)
  const mask = std.textureSampleLevel(weightLayout.$.maskTexture, weightLayout.$.sampler, uv, d.f32(0)).r
  const z = weightLayout.$.inputBuffer[index]!
  const rawLuma = weightLayout.$.stats[d.u32(0)]! + weightLayout.$.stats[d.u32(1)]! * z
  weightLayout.$.output[index] = d.vec2f(rawLuma * mask, mask)
}

const horizontalBlur = (x: number, y: number) => {
  'use gpu'
  const params = blurLayout.$.params
  const width = d.i32(params.size.x)
  const index = d.u32(y) * d.u32(width) + d.u32(x)
  let sum = d.vec2f(0, 0)
  for (let offset = -BLUR_RADIUS; offset <= BLUR_RADIUS; offset += 1) {
    const sampleX = clamp(d.i32(x) + d.i32(offset), d.i32(0), width - d.i32(1))
    sum += blurLayout.$.input[d.u32(y) * d.u32(width) + d.u32(sampleX)]!
  }
  blurLayout.$.output[index] = d.vec2f(sum)
}

const verticalBlur = (x: number, y: number) => {
  'use gpu'
  const params = blurLayout.$.params
  const width = d.i32(params.size.x)
  const height = d.i32(params.size.y)
  const index = d.u32(y) * d.u32(width) + d.u32(x)
  let sum = d.vec2f(0, 0)
  for (let offset = -BLUR_RADIUS; offset <= BLUR_RADIUS; offset += 1) {
    const sampleY = clamp(d.i32(y) + d.i32(offset), d.i32(0), height - d.i32(1))
    sum += blurLayout.$.input[d.u32(sampleY) * d.u32(width) + d.u32(x)]!
  }
  blurLayout.$.output[index] = d.vec2f(sum)
}

const backgroundToTexture = (x: number, y: number) => {
  'use gpu'
  const params = textureLayout.$.params
  const width = d.u32(params.size.x)
  const value = textureLayout.$.input[d.u32(y) * width + d.u32(x)]!
  const weight = max(value.y, d.f32(MASK_WEIGHT_EPSILON))
  const background = select(textureLayout.$.stats[d.u32(0)]!, value.x / weight, value.y > d.f32(MASK_WEIGHT_EPSILON))
  std.textureStore(
    textureLayout.$.backgroundTexture,
    d.vec2u(x, y),
    d.vec4f(background, background, background, d.f32(1)),
  )
}

export type RoiBackgroundStage = {
  backgroundTexture: TgpuTexture & SampledFlag
  encode: () => void
  destroy: () => void
}

export type RoiBackgroundTimingName =
  | 'backgroundWeight'
  | 'backgroundHorizontal'
  | 'backgroundVertical'
  | 'backgroundTexture'

export type RoiBackgroundTimingCallback = (name: RoiBackgroundTimingName, ms: number) => void

export function createRoiBackgroundStage(
  root: TgpuRoot,
  modelWidth: number,
  modelHeight: number,
  inputBuffer: TgpuBuffer<d.WgslArray<d.F32>> & StorageFlag,
  maskTexture: TgpuTexture & SampledFlag,
  sampler: TgpuSampler,
  statsBuffer: TgpuBuffer<d.WgslArray<d.F32>> & StorageFlag,
  onGpuTiming?: RoiBackgroundTimingCallback,
): RoiBackgroundStage {
  const area = modelWidth * modelHeight
  const weightedBuffer = root.createBuffer(d.arrayOf(d.vec2f, area)).$usage('storage')
  const blurredBuffer = root.createBuffer(d.arrayOf(d.vec2f, area)).$usage('storage')
  const paramsBuffer = root.createUniform(backgroundParams, {
    size: d.vec2f(modelWidth, modelHeight),
    maskSize: d.vec2f(modelWidth / 2, modelHeight / 2),
  })
  const backgroundTexture = root
    .createTexture({
      size: [modelWidth, modelHeight],
      format: 'rgba8unorm',
    })
    .$usage('sampled', 'storage')

  const weightBindGroup = root.createBindGroup(weightLayout, {
    inputBuffer,
    stats: statsBuffer,
    maskTexture,
    sampler,
    params: paramsBuffer,
    output: weightedBuffer,
  })
  const horizontalBindGroup = root.createBindGroup(blurLayout, {
    input: weightedBuffer,
    params: paramsBuffer,
    output: blurredBuffer,
  })
  const verticalBindGroup = root.createBindGroup(blurLayout, {
    input: blurredBuffer,
    params: paramsBuffer,
    output: weightedBuffer,
  })
  const textureBindGroup = root.createBindGroup(textureLayout, {
    input: weightedBuffer,
    stats: statsBuffer,
    params: paramsBuffer,
    backgroundTexture,
  })

  const timed = (name: RoiBackgroundTimingName, pipeline: ReturnType<typeof root.createGuardedComputePipeline>) =>
    onGpuTiming === undefined
      ? pipeline
      : pipeline.withPerformanceCallback((start, end) => {
          onGpuTiming(name, Number(end - start) / 1e6)
        })

  const weightPipeline = timed('backgroundWeight', root.createGuardedComputePipeline(weightPass))
  const horizontalPipeline = timed('backgroundHorizontal', root.createGuardedComputePipeline(horizontalBlur))
  const verticalPipeline = timed('backgroundVertical', root.createGuardedComputePipeline(verticalBlur))
  const texturePipeline = timed('backgroundTexture', root.createGuardedComputePipeline(backgroundToTexture))

  return {
    backgroundTexture,
    encode() {
      weightPipeline.with(weightBindGroup).dispatchThreads(modelWidth, modelHeight)
      horizontalPipeline.with(horizontalBindGroup).dispatchThreads(modelWidth, modelHeight)
      verticalPipeline.with(verticalBindGroup).dispatchThreads(modelWidth, modelHeight)
      texturePipeline.with(textureBindGroup).dispatchThreads(modelWidth, modelHeight)
    },
    destroy() {
      weightedBuffer.destroy()
      blurredBuffer.destroy()
      backgroundTexture.destroy()
    },
  }
}
