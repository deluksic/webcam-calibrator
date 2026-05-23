// Non-Maximum Suppression (NMS) pipeline: sobelBuffer → filteredBuffer
//
// Keeps only pixels that are local maxima along the edge tangent direction.
// Edge tangent = gradient direction rotated 90°: tangent = (-gy/gm, gx/gm).
// Uses bilinear interpolation for sub-pixel neighbor sampling.
//
// Output: filteredBuffer[i] = suppressed gradient (vec2f; zero if not a local max or below threshold).
// The edges display pipeline computes magnitude from length() on-the-fly.
import type { ExtractBindGroupInputFromLayout, TgpuRoot } from 'typegpu'
import { tgpu, d, std } from 'typegpu'
import { add, div, length, select, sub } from 'typegpu/std'

export const edgeFilterLayout = tgpu.bindGroupLayout({
  sobelBuffer: { storage: d.arrayOf(d.vec2f), access: 'readonly' },
  threshold: { uniform: d.f32 },
  filteredBuffer: { storage: d.arrayOf(d.vec2f), access: 'mutable' },
}).$name('nms-bgl')

export type EdgeFilterBindResources = ExtractBindGroupInputFromLayout<typeof edgeFilterLayout.entries>

/** Full-frame tile; match `computeDispatch2d` in cameraFrame. */
const FULL_FRAME_WG = 16

const { ceil } = Math

/** Allocates NMS `filteredBuffer` and edge-strength `threshold` uniform; reads `sobelBuffer` (upstream). */
export function createEdgeFilterStage(
  root: TgpuRoot,
  width: number,
  height: number,
  sobelBuffer: EdgeFilterBindResources['sobelBuffer'],
) {
  const thresholdBuffer = root.createBuffer(d.f32).$name('nms-threshold').$usage('uniform')
  const filteredBuffer = root.createBuffer(d.arrayOf(d.vec2f, width * height)).$name('nms-filtered').$usage('storage')
  const { pipeline, bindGroup } = createEdgeFilterPipeline(root, width, height, {
    sobelBuffer,
    threshold: thresholdBuffer,
    filteredBuffer,
  })
  const wgX = ceil(width / FULL_FRAME_WG)
  const wgY = ceil(height / FULL_FRAME_WG)
  const encodeCompute = (pass: GPUComputePassEncoder) => {
    pipeline.with(pass).with(bindGroup).dispatchWorkgroups(wgX, wgY)
  }
  return { filteredBuffer, thresholdBuffer, encodeCompute }
}

export function createEdgeFilterPipeline(
  root: TgpuRoot,
  width: number,
  height: number,
  resources: EdgeFilterBindResources,
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [FULL_FRAME_WG, FULL_FRAME_WG, 1],
  })((input) => {
    'use gpu'
    if (d.i32(input.gid.x) >= d.i32(width) || d.i32(input.gid.y) >= d.i32(height)) {
      return
    }

    const x = d.i32(input.gid.x)
    const y = d.i32(input.gid.y)
    const w = d.i32(width)
    const h = d.i32(height)
    const idx = d.u32(y) * d.u32(w) + d.u32(x)

    const g = edgeFilterLayout.$.sobelBuffer[idx]!
    const gm = length(g)
    const threshold = edgeFilterLayout.$.threshold

    if (gm < threshold) {
      edgeFilterLayout.$.filteredBuffer[idx] = d.vec2f(d.f32(0), d.f32(0))
      return
    }

    // Gradient direction (perpendicular to edge) for NMS — round to nearest int pixel
    const gDir = div(g, gm)
    const step = d.vec2i(select(sub(gDir, d.f32(0.5)), add(gDir, d.f32(0.5)), gDir >= d.vec2f(0)))
    const pos = d.vec2i(x, y)
    const posP = add(pos, step)
    const posN = sub(pos, step)

    // Collect neighbor magnitudes (0 if out of bounds)
    let gmPos = d.f32(0)
    if (posP.x >= d.i32(0) && posP.x < w && posP.y >= d.i32(0) && posP.y < h) {
      const idxP = d.u32(posP.y) * d.u32(w) + d.u32(posP.x)
      gmPos = length(edgeFilterLayout.$.sobelBuffer[idxP]!)
    }
    let gmNeg = d.f32(0)
    if (posN.x >= d.i32(0) && posN.x < w && posN.y >= d.i32(0) && posN.y < h) {
      const idxN = d.u32(posN.y) * d.u32(w) + d.u32(posN.x)
      gmNeg = length(edgeFilterLayout.$.sobelBuffer[idxN]!)
    }

    // Only suppress if neighbor is meaningfully stronger (not just slightly)
    const strongFactor = d.f32(1.05) // neighbor must be >5% stronger to suppress
    const suppressed = gmPos > gm * strongFactor || gmNeg > gm * strongFactor

    // Store suppressed gradient (edges pipeline derives magnitude via length())
    edgeFilterLayout.$.filteredBuffer[idx] = select(g, d.vec2f(d.f32(0), d.f32(0)), suppressed)
  })

  const pipeline = root.createComputePipeline({ compute: kernel }).$name('nms')
  const bindGroup = root.createBindGroup(edgeFilterLayout, resources)
  return { pipeline, bindGroup }
}
