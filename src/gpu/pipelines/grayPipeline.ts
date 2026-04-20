// Gray pipeline: grayTex → grayBuffer
import { tgpu, d, std } from 'typegpu'

import { WR, WG, WB } from '@/gpu/pipelines/constants'

export function createGrayPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  grayTexToBufferLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  const grayKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu'
    if (d.i32(input.gid.x) >= d.i32(width) || d.i32(input.gid.y) >= d.i32(height)) {
      return
    }

    const color = std.textureLoad(grayTexToBufferLayout.$.grayTex, input.gid.xy, 0)
    const gray = color.r * d.f32(WR) + color.g * d.f32(WG) + color.b * d.f32(WB)

    const idx = d.u32(d.i32(input.gid.y) * d.i32(width) + d.i32(input.gid.x))
    grayTexToBufferLayout.$.grayBuffer[idx] = gray
  })

  return root.createComputePipeline({ compute: grayKernel })
}
