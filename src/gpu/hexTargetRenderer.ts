import type { TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'
import { select } from 'typegpu/std'

import type { HexTargetLayout } from '@/lib/hexBlobTarget'

const MAX_TARGET_QUADS = 4096

const TargetUniform = d.struct({
  width: d.f32,
  height: d.f32,
})

const TargetQuadGpu = d.struct({
  left: d.f32,
  top: d.f32,
  right: d.f32,
  bottom: d.f32,
  color: d.u32,
})

const TargetQuadsSchema = d.arrayOf(TargetQuadGpu, MAX_TARGET_QUADS)

const targetRenderLayout = tgpu
  .bindGroupLayout({
    target: { uniform: TargetUniform },
    quads: { storage: TargetQuadsSchema, access: 'readonly' },
  })
  .$name('hex-target-render')

export type HexTargetRenderer = {
  resize(widthPx: number, heightPx: number): void
  setLayout(layout: HexTargetLayout): void
  render(): void
  destroy(): void
}

/** Draws integer-pixel target quads with no MSAA, blending, texture sampling, or depth. */
export function createHexTargetRenderer(
  root: TgpuRoot,
  canvas: HTMLCanvasElement,
  presentationFormat: GPUTextureFormat,
): HexTargetRenderer {
  const context = root.configureContext({ canvas, alphaMode: 'opaque' })
  const uniformBuffer = root.createBuffer(TargetUniform).$usage('uniform')
  const quadBuffer = root.createBuffer(TargetQuadsSchema).$usage('storage')
  const bindGroup = root.createBindGroup(targetRenderLayout, {
    target: uniformBuffer,
    quads: quadBuffer,
  })

  const vertex = tgpu.vertexFn({
    in: {
      vertexIndex: d.builtin.vertexIndex,
      instanceIndex: d.builtin.instanceIndex,
    },
    out: {
      clipPos: d.builtin.position,
      color: d.interpolate('flat', d.u32),
    },
  })(({ vertexIndex, instanceIndex }) => {
    'use gpu'
    const quad = targetRenderLayout.$.quads[instanceIndex]!
    const target = targetRenderLayout.$.target
    // Triangle list: TL, TR, BL, BL, TR, BR.
    const useRight = vertexIndex === d.u32(1) || vertexIndex === d.u32(4) || vertexIndex === d.u32(5)
    const useBottom = vertexIndex === d.u32(2) || vertexIndex === d.u32(3) || vertexIndex === d.u32(5)
    const px = select(quad.left, quad.right, useRight)
    const py = select(quad.top, quad.bottom, useBottom)
    return {
      clipPos: d.vec4f(
        (d.f32(2) * px) / target.width - d.f32(1),
        d.f32(1) - (d.f32(2) * py) / target.height,
        d.f32(0),
        d.f32(1),
      ),
      color: quad.color,
    }
  })

  const fragment = tgpu.fragmentFn({
    in: { color: d.interpolate('flat', d.u32) },
    out: d.vec4f,
  })(({ color }) => {
    'use gpu'
    return select(d.vec4f(0, 0, 0, 1), d.vec4f(1, 1, 1, 1), color === d.u32(1))
  })

  const pipeline = root
    .createRenderPipeline({
      vertex,
      fragment,
      targets: { format: presentationFormat },
      primitive: { topology: 'triangle-list', cullMode: 'none' },
    })
    .$name('hex-target-render')

  let drawCount = 0
  let widthPx = 0
  let heightPx = 0

  const resize = (width: number, height: number): void => {
    widthPx = Math.max(0, Math.floor(width))
    heightPx = Math.max(0, Math.floor(height))
    if (canvas.width !== widthPx) {
      canvas.width = widthPx
    }
    if (canvas.height !== heightPx) {
      canvas.height = heightPx
    }
  }

  const setLayout = (layout: HexTargetLayout): void => {
    widthPx = layout.widthPx
    heightPx = layout.heightPx
    uniformBuffer.write({ width: Math.max(1, widthPx), height: Math.max(1, heightPx) })
    const rows: Array<{
      left: number
      top: number
      right: number
      bottom: number
      color: number
    }> = []
    for (const quad of layout.quads.slice(0, MAX_TARGET_QUADS)) {
      rows.push({
        left: quad.left,
        top: quad.top,
        right: quad.right,
        bottom: quad.bottom,
        color: quad.color,
      })
    }
    while (rows.length < MAX_TARGET_QUADS) {
      rows.push({ left: 0, top: 0, right: 0, bottom: 0, color: 0 })
    }
    quadBuffer.write(rows)
    drawCount = Math.min(layout.quads.length, MAX_TARGET_QUADS)
  }

  const render = (): void => {
    if (widthPx < 1 || heightPx < 1) {
      return
    }
    const encoder = root.device.createCommandEncoder({ label: 'hex-target-render' })
    pipeline
      .with(encoder)
      .withColorAttachment({
        view: context.getCurrentTexture().createView(),
        loadOp: 'clear',
        storeOp: 'store',
        clearValue: [0.5, 0.5, 0.5, 1],
      })
      .with(bindGroup)
      .draw(6, Math.max(0, drawCount))
    root.device.queue.submit([encoder.finish()])
  }

  return {
    resize,
    setLayout,
    render,
    destroy() {
      uniformBuffer.destroy()
      quadBuffer.destroy()
    },
  }
}
