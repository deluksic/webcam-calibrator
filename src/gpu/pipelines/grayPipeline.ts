// Gray pipeline: grayTex → grayBuffer
import type { ExtractBindGroupInputFromLayout, TgpuRoot } from 'typegpu'
import { tgpu, d, std } from 'typegpu'

export const grayTexToBufferLayout = tgpu.bindGroupLayout({
  grayTex: { texture: d.texture2d(d.f32), access: 'readonly' },
  grayBuffer: { storage: d.arrayOf(d.f32), access: 'mutable' },
})

export type GrayTexToBufferBindResources = ExtractBindGroupInputFromLayout<typeof grayTexToBufferLayout.entries>

/** Full-frame tile; match `computeDispatch2d` in cameraFrame. */
const FULL_FRAME_WG = 16

const { ceil } = Math

/** Allocates `grayBuffer`; binds `grayTex` (upstream). */
export function createGrayStage(
  root: TgpuRoot,
  width: number,
  height: number,
  grayTex: GrayTexToBufferBindResources['grayTex'],
) {
  const buffer = root.createBuffer(d.arrayOf(d.f32, width * height)).$usage('storage')
  const { pipeline, bindGroup } = createGrayPipeline(root, width, height, { grayTex, grayBuffer: buffer })
  const wgX = ceil(width / FULL_FRAME_WG)
  const wgY = ceil(height / FULL_FRAME_WG)
  const encodeCompute = (pass: GPUComputePassEncoder) => {
    pipeline.with(pass).with(bindGroup).dispatchWorkgroups(wgX, wgY)
  }
  return { buffer, encodeCompute }
}

export function createGrayPipeline(
  root: TgpuRoot,
  width: number,
  height: number,
  resources: GrayTexToBufferBindResources,
) {
  const grayKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [FULL_FRAME_WG, FULL_FRAME_WG, 1],
  })((input) => {
    'use gpu'
    if (input.gid.x >= width || input.gid.y >= height) {
      return
    }

    const color = std.textureLoad(grayTexToBufferLayout.$.grayTex, input.gid.xy, 0)
    const gray = color.g

    const idx = input.gid.y * width + input.gid.x
    grayTexToBufferLayout.$.grayBuffer[idx] = gray
  })

  const pipeline = root.createComputePipeline({ compute: grayKernel })
  const bindGroup = root.createBindGroup(grayTexToBufferLayout, resources)
  return { pipeline, bindGroup }
}
