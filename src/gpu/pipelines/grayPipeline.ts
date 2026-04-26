// Gray pipeline: grayTex → grayBuffer
import type { TgpuRoot } from 'typegpu'
import { tgpu, d, std } from 'typegpu'

export function createGrayPipeline(
  root: TgpuRoot,
  grayTexToBufferLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  const grayKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
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

  return root.createComputePipeline({ compute: grayKernel })
}
