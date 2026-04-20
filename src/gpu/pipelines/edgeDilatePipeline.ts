// Merge edge responses only along the local edge tangent (perpendicular to Sobel gradient).
// Skips neighbors that lie mostly along the gradient normal so the mask does not thicken.
// Outputs gradient (vec2f) instead of scalar mask.
import { tgpu, d, std } from 'typegpu'
import { abs, length, sqrt } from 'typegpu/std'

import { EDGE_DILATE_THRESHOLD } from '@/gpu/pipelines/constants'

export function createEdgeDilatePipeline(
  root: Awaited<ReturnType<typeof tgpu.init>>,
  edgeDilateLayout: ReturnType<typeof tgpu.bindGroupLayout>,
  width: number,
  height: number,
) {
  const dilateKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [16, 16, 1],
  })((input) => {
    'use gpu'
    if (d.i32(input.gid.x) >= d.i32(width) || d.i32(input.gid.y) >= d.i32(height)) {
      return
    }

    const x = d.i32(input.gid.x)
    const y = d.i32(input.gid.y)
    const w = d.i32(width)
    const h = d.i32(height)
    const wU32 = d.u32(w)
    const idx = d.u32(y) * wU32 + d.u32(x)

    const srcG = edgeDilateLayout.$.src[idx]
    const srcMag = length(srcG)

    const g = edgeDilateLayout.$.grad[idx]
    const gm = length(g)
    const eps = d.f32(1e-6)

    // Normalize gradient for tangent computation.
    // If gm <= eps (suppressed), ngx/ngy fall back to 0, which means this pixel
    // will accept any aligned neighbor (correct: suppressed gaps have no direction).
    const ngx = std.select(g.x / gm, d.f32(0), gm <= eps)
    const ngy = std.select(g.y / gm, d.f32(0), gm <= eps)

    // Start with current pixel's gradient (may be zero if NMS suppressed).
    let bestGx = g.x
    let bestGy = g.y
    let bestMag = srcMag

    for (const iy of tgpu.unroll(std.range(3))) {
      for (const ix of tgpu.unroll(std.range(3))) {
        const dx = d.i32(ix) - d.i32(1)
        const dy = d.i32(iy) - d.i32(1)
        const isCenter = dx === d.i32(0) && dy === d.i32(0)
        const isDiagonal = abs(d.f32(dx)) === d.f32(1) && abs(d.f32(dy)) === d.f32(1)
        const isFar = abs(d.f32(dx)) > d.f32(2) || abs(d.f32(dy)) > d.f32(2)
        const skip = isCenter || isDiagonal || isFar

        if (!skip) {
          const nx2 = x + dx
          const ny2 = y + dy
          if (nx2 >= d.i32(0) && nx2 < w && ny2 >= d.i32(0) && ny2 < h) {
            const fx = d.f32(dx)
            const fy = d.f32(dy)
            const ulen = sqrt(fx * fx + fy * fy)
            const ux = fx / ulen
            const uy = fy / ulen
            const align = abs(ux * ngx + uy * ngy)
            if (align <= d.f32(EDGE_DILATE_THRESHOLD)) {
              const nIdx = d.u32(ny2) * wU32 + d.u32(nx2)
              const nm = length(edgeDilateLayout.$.src[nIdx])
              if (nm > bestMag) {
                bestMag = nm
                const ng = edgeDilateLayout.$.grad[nIdx]
                bestGx = ng.x
                bestGy = ng.y
              }
            }
          }
        }
      }
    }

    // If NMS output is zero (suppressed gap), adopt best aligned neighbor to close it.
    // If NMS output is non-zero (survived NMS), keep it — edges stay thin.
    const suppressed = gm <= eps
    const bestGx2 = std.select(bestGx, g.x, suppressed)
    const bestGy2 = std.select(bestGy, g.y, suppressed)
    edgeDilateLayout.$.dst[idx] = d.vec2f(bestGx2, bestGy2)
  })

  return root.createComputePipeline({ compute: dilateKernel })
}
