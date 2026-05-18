// Per registered quad: line intersections → corner order → DLT homography → grid viz buffer.
import type { TgpuRoot } from 'typegpu'
import { tgpu, d, std } from 'typegpu'

import { COMPONENT_LABEL_INVALID } from '@/gpu/contour'
import { MAX_EDGES_PER_LABEL } from '@/gpu/lineFitThresholds'
import type {
  QuadCountBuffer,
  QuadPeakEdgeBuffer,
  QuadSourceLabelIdBuffer,
} from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import { MAX_QUADS } from '@/gpu/pipelines/edgeHistogramClusterPipeline'
import type { EdgeLineOutBuffer } from '@/gpu/pipelines/edgeLineFitPipeline'
import { EdgeLineEntry } from '@/gpu/pipelines/edgeLineFitPipeline'
import {
  DECODED_TAG_ID_UNKNOWN,
  GridDataSchema,
  QuadDataGpu,
  QuadDebug,
  type GridVizQuadBuffer,
} from '@/gpu/pipelines/gridVizPipeline'
import { Corners4, linesFromEdgeEntries, solveQuadCornersAndHomography } from '@/gpu/shaders/quadCornerOrder'
import { invalidGridHomography } from '@/gpu/shaders/homographyDlt'

const WORKGROUP_SIZE = 64

const EdgeLineArray4 = d.arrayOf(EdgeLineEntry, MAX_EDGES_PER_LABEL)

function createQuadCornerHomographyLayouts() {
  const layout = tgpu.bindGroupLayout({
    lineOut: { storage: d.arrayOf(EdgeLineEntry), access: 'readonly' },
    quadPeakEdge: { storage: d.arrayOf(d.u32), access: 'readonly' },
    quadSourceLabelId: { storage: d.arrayOf(d.u32), access: 'readonly' },
    quadCount: { storage: d.arrayOf(d.u32, 1), access: 'readonly' },
    quadData: { storage: GridDataSchema, access: 'mutable' },
  })
  return { layout }
}

function createQuadCornerHomographyPipeline(
  root: TgpuRoot,
  layout: ReturnType<typeof createQuadCornerHomographyLayouts>['layout'],
) {
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE, 1, 1],
  })((input) => {
    'use gpu'
    const quadId = d.u32(input.gid.x)
    if (quadId >= d.u32(MAX_QUADS)) {
      return
    }

    const nQuads = layout.$.quadCount[d.u32(0)]!
    const deadScreen = Corners4()
    for (const i of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      deadScreen[i] = d.vec2f(0, 0)
    }
    const emptyDebug = QuadDebug({
      failureCode: d.u32(0),
      edgePixelCount: d.f32(0),
      minR2: d.f32(0),
      intersectionCount: d.f32(0),
    })
    const emptyQuad = QuadDataGpu({
      homography: invalidGridHomography(),
      screenCorners: deadScreen,
      debug: emptyDebug,
      decodedTagId: d.u32(DECODED_TAG_ID_UNKNOWN),
      decodedRotation: d.u32(0),
    })

    if (quadId >= nQuads) {
      layout.$.quadData[quadId] = QuadDataGpu(emptyQuad)
      return
    }

    const labelId = layout.$.quadSourceLabelId[quadId]!
    if (labelId === d.u32(COMPONENT_LABEL_INVALID)) {
      layout.$.quadData[quadId] = QuadDataGpu(emptyQuad)
      return
    }

    const quadBase = quadId * d.u32(MAX_EDGES_PER_LABEL)
    const entries = EdgeLineArray4()
    let edgePixels = d.f32(0)

    for (const edgeId of tgpu.unroll(std.range(0, MAX_EDGES_PER_LABEL))) {
      const peakEdge = layout.$.quadPeakEdge[quadBase + edgeId]!
      if (peakEdge === d.u32(COMPONENT_LABEL_INVALID)) {
        layout.$.quadData[quadId] = QuadDataGpu({
          homography: invalidGridHomography(),
          screenCorners: deadScreen,
          debug: QuadDebug({
            failureCode: d.u32(1 << 4),
            edgePixelCount: edgePixels,
            minR2: d.f32(0),
            intersectionCount: d.f32(0),
          }),
          decodedTagId: d.u32(DECODED_TAG_ID_UNKNOWN),
          decodedRotation: d.u32(0),
        })
        return
      }
      const flatSlot = quadId * d.u32(MAX_EDGES_PER_LABEL) + edgeId
      const line = layout.$.lineOut[flatSlot]!
      if (line.valid === d.u32(0)) {
        layout.$.quadData[quadId] = QuadDataGpu({
          homography: invalidGridHomography(),
          screenCorners: deadScreen,
          debug: QuadDebug({
            failureCode: d.u32(1 << 4),
            edgePixelCount: edgePixels,
            minR2: d.f32(0),
            intersectionCount: d.f32(0),
          }),
          decodedTagId: d.u32(DECODED_TAG_ID_UNKNOWN),
          decodedRotation: d.u32(0),
        })
        return
      }
      entries[edgeId] = EdgeLineEntry(line)
      edgePixels = edgePixels + d.f32(line.inlierCount)
    }

    const lines = linesFromEdgeEntries(entries)
    const solved = solveQuadCornersAndHomography(lines)

    layout.$.quadData[quadId] = QuadDataGpu({
      homography: solved.homography,
      screenCorners: solved.corners,
      debug: QuadDebug({
        failureCode: solved.failureCode,
        edgePixelCount: edgePixels,
        minR2: d.f32(0),
        intersectionCount: d.f32(solved.intersectionCount),
      }),
      decodedTagId: d.u32(DECODED_TAG_ID_UNKNOWN),
      decodedRotation: d.u32(0),
    })
  })

  return root.createComputePipeline({ compute: kernel })
}

export function createQuadCornerHomographyStage(
  root: TgpuRoot,
  deps: {
    lineOut: EdgeLineOutBuffer
    quadPeakEdge: QuadPeakEdgeBuffer
    quadSourceLabelId: QuadSourceLabelIdBuffer
    quadCount: QuadCountBuffer
    quadDataBuffer: GridVizQuadBuffer
  },
) {
  const layouts = createQuadCornerHomographyLayouts()
  const pipeline = createQuadCornerHomographyPipeline(root, layouts.layout)

  const bindGroup = root.createBindGroup(layouts.layout, {
    lineOut: deps.lineOut,
    quadPeakEdge: deps.quadPeakEdge,
    quadSourceLabelId: deps.quadSourceLabelId,
    quadCount: deps.quadCount,
    quadData: deps.quadDataBuffer,
  })

  const wg = Math.ceil(MAX_QUADS / WORKGROUP_SIZE)

  const encodeCompute = (pass: GPUComputePassEncoder) => {
    pipeline.with(pass).with(bindGroup).dispatchWorkgroups(wg)
  }

  return { encodeCompute }
}
