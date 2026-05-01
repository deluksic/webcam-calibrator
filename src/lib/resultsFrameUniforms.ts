import { d } from 'typegpu'

import type {
  AxisResultsUniformGpuBuffer,
  MarkerResultsUniformGpuBuffer,
} from '@/gpu/resultsVizLayouts'
import { RESULTS_MARKER_DISK_RADIUS_PX } from '@/gpu/resultsVizLayouts'
import { WORLD_AXIS_HALF_LEN, buildClipHomogeneousMatrixFromCentroidBoard } from '@/lib/orbitOrthoMath'

/** Half stroke width (~1.5 px) for axis arrows in framebuffer pixel space. */
const axisPolylineHalfStrokePixels = 2 as const

const clipFromBoardScratch = new Float32Array(16)

function mat4x4fFromColMajor(m: Float32Array) {
  return d.mat4x4f(
    m[0]!,
    m[1]!,
    m[2]!,
    m[3]!,
    m[4]!,
    m[5]!,
    m[6]!,
    m[7]!,
    m[8]!,
    m[9]!,
    m[10]!,
    m[11]!,
    m[12]!,
    m[13]!,
    m[14]!,
    m[15]!,
  )
}

/**
 * Writes clip-from-board matrix for markers and axes. Marker disks use screen-space billboarded radius;
 * axes project board endpoints to NDC → pixel strokes then back to NDC inside the shader.
 */
export function writeResultsMarkerAndAxisUniforms(input: {
  aspectWidthOverHeight: number
  yawRad: number
  pitchRad: number
  baseOrthoExtentY: number
  orthoZoom: number
  viewportWidthPx: number
  viewportHeightPx: number
  pointCount: number
  markerUni: MarkerResultsUniformGpuBuffer
  axisUni: AxisResultsUniformGpuBuffer
}): void {
  const orthoExtentY = input.baseOrthoExtentY / input.orthoZoom
  buildClipHomogeneousMatrixFromCentroidBoard(
    input.aspectWidthOverHeight,
    input.yawRad,
    input.pitchRad,
    orthoExtentY,
    clipFromBoardScratch,
  )

  const clipMatrixCpu = mat4x4fFromColMajor(clipFromBoardScratch)
  const vw = Math.max(1, input.viewportWidthPx)
  const vh = Math.max(1, input.viewportHeightPx)
  const viewportHalfWidthPixels = vw * 0.5
  const viewportHalfHeightPixels = vh * 0.5
  input.markerUni.write({
    clipHomogeneousMatrixFromCentroidBoard: clipMatrixCpu,
    viewportHalfSizePixels: d.vec2f(viewportHalfWidthPixels, viewportHalfHeightPixels),
    markerDiskRadiusPixels: d.f32(RESULTS_MARKER_DISK_RADIUS_PX),
    displayedMarkerCount: d.u32(Math.max(0, Math.floor(input.pointCount))),
    orbitMarkerUniformReserve: d.vec2f(0, 0),
  })

  input.axisUni.write({
    clipHomogeneousMatrixFromCentroidBoard: clipMatrixCpu,
    viewportHalfSizePixels: d.vec2f(viewportHalfWidthPixels, viewportHalfHeightPixels),
    axisPolylineHalfStrokePixels: d.f32(axisPolylineHalfStrokePixels),
    worldAxisReachDistanceCentroidBoardUnits: d.f32(WORLD_AXIS_HALF_LEN),
  })
}
