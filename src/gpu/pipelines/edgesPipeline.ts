// Edge render pipeline: filteredBuffer + sobelBuffer → edges canvas
// Colorizes edges by gradient direction using continuous HSV coloring.
import { tgpu, d, std } from 'typegpu'
import { common } from 'typegpu'
import { atan2, clamp, floor, length, max } from 'typegpu/std'

export function createEdgesPipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  edgesLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
  presentationFormat: GPUTextureFormat,
) {
  const frag = tgpu.fragmentFn({
    in: { uv: d.location(0, d.vec2f) },
    out: d.vec4f,
  })((i) => {
    'use gpu'
    const wi = d.i32(width)
    const hi = d.i32(height)
    const maxPx = d.f32(wi - d.i32(1))
    const maxPy = d.f32(hi - d.i32(1))
    const px = d.u32(floor(clamp(i.uv.x * d.f32(wi), d.f32(0), maxPx)))
    const py = d.u32(floor(clamp(i.uv.y * d.f32(hi), d.f32(0), maxPy)))
    const idx = py * d.u32(wi) + px

    const magVec = edgesLayout.$.filteredBuffer[idx]
    const mag = length(magVec)
    if (mag <= d.f32(0)) {
      return d.vec4f(d.f32(0.08), d.f32(0.08), d.f32(0.12), d.f32(1))
    }

    const g = edgesLayout.$.sobelBuffer[idx]
    const gm = length(g)
    const eps = d.f32(1e-6)
    const gxn = std.select(g.x / gm, d.f32(0), gm <= eps)
    const gyn = std.select(g.y / gm, d.f32(0), gm <= eps)

    // TODO: clean up this angle stuff, avoid if statements
    let angle = atan2(gyn, gxn) / d.f32(6.28318530718) + d.f32(0.5)
    if (angle < d.f32(0)) {
      angle = angle + d.f32(1)
    }
    if (angle >= d.f32(1)) {
      angle = angle - d.f32(1)
    }

    let hue = angle + d.f32(0.5)
    if (hue >= d.f32(1)) {
      hue = hue - d.f32(1)
    }

    const sat = d.f32(0.9)
    const val = max(mag, d.f32(0.2))
    const h = hue * d.f32(6)
    const sector = floor(h)
    const f = h - sector
    const p = val * (d.f32(1) - sat)
    const q = val * (d.f32(1) - f * sat)
    const t = val * (d.f32(1) - (d.f32(1) - f) * sat)

    const s = sector
    let r = d.f32(0)
    let gV = d.f32(0)
    let b = d.f32(0)
    if (s === d.f32(0)) {
      r = val
      gV = t
      b = p
    } else if (s === d.f32(1)) {
      r = q
      gV = val
      b = p
    } else if (s === d.f32(2)) {
      r = p
      gV = val
      b = t
    } else if (s === d.f32(3)) {
      r = p
      gV = q
      b = val
    } else if (s === d.f32(4)) {
      r = t
      gV = p
      b = val
    } else {
      r = val
      gV = p
      b = q
    }

    return d.vec4f(r, gV, b, d.f32(1))
  })

  return root.createRenderPipeline({
    vertex: common.fullScreenTriangle,
    fragment: frag,
    targets: { format: presentationFormat },
  })
}
