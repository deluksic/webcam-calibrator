// Grayscale render pipeline: grayBuffer → canvas
import type { TgpuRoot } from 'typegpu'
import { tgpu, d } from 'typegpu'
import { common } from 'typegpu'

export function createGrayRenderPipeline(
  root: TgpuRoot,
  grayLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  _height: number,
  presentationFormat: GPUTextureFormat,
) {
  const grayFrag = tgpu.fragmentFn({
    in: { pos: d.builtin.position },
    out: d.vec4f,
  })((i) => {
    'use gpu'
    const px = d.i32(i.pos.x)
    const py = d.i32(i.pos.y)
    const idx = py * width + px
    const gray = grayLayout.$.grayBuffer[idx]
    return d.vec4f(gray, gray, gray, 1)
  })

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: grayFrag,
    targets: { format: presentationFormat },
  })
}
