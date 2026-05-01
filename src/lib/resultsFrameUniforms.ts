import { d } from 'typegpu'

import type {
  AxisResultsUniformGpuBuffer,
  MarkerResultsUniformGpuBuffer,
  TagQuadResultsUniformGpuBuffer,
} from '@/gpu/resultsVizLayouts'
import { RESULTS_MARKER_DISK_RADIUS_PX } from '@/gpu/resultsVizLayouts'
import type { Mat4Arg, Vec3Arg } from 'wgpu-matrix'

import { WORLD_AXIS_HALF_LEN, buildOrbitClipMatrix } from '@/lib/orbitOrthoMath'

/** Half stroke width (~1.5 px) for axis arrows in framebuffer pixel space. */
const axisPolylineHalfStrokePixels = 2 as const

const clipFromBoardScratch: Mat4Arg = Array.from({ length: 16 }, () => 0)

function mat4x4fFromColMajor(m: Mat4Arg) {
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
 * Writes the clip-from-board matrix into all three passes' uniforms (markers, axes, tag quads) plus
 * pass-specific fields. Marker disks use a screen-space billboarded radius; axes project board endpoints
 * to NDC → pixel strokes then back to NDC inside the shader; tag quads draw their bit pattern in the
 * fragment shader. RGB axes plant at board origin **`(0,0,0)`** (`axisOriginBoard`); orbit **`lookAt`**
 * is the same origin (board frame).
 */
export function writeResultsFrameUniforms(input: {
  aspectWidthOverHeight: number
  /** Unit direction from board origin toward the camera eye (board space). */
  orbitEyeDirUnit: Vec3Arg
  baseOrthoExtentY: number
  orthoZoom: number
  viewportWidthPx: number
  viewportHeightPx: number
  pointCount: number
  markerUni: MarkerResultsUniformGpuBuffer
  axisUni: AxisResultsUniformGpuBuffer
  tagQuadUni: TagQuadResultsUniformGpuBuffer
}): void {
  const orthoExtentY = input.baseOrthoExtentY / input.orthoZoom
  buildOrbitClipMatrix(
    { x: 0, y: 0, z: 0 },
    input.aspectWidthOverHeight,
    input.orbitEyeDirUnit,
    orthoExtentY,
    clipFromBoardScratch,
  )

  const clipMatrixCpu = mat4x4fFromColMajor(clipFromBoardScratch)
  const vw = Math.max(1, input.viewportWidthPx)
  const vh = Math.max(1, input.viewportHeightPx)
  const viewportHalfWidthPixels = vw * 0.5
  const viewportHalfHeightPixels = vh * 0.5
  input.markerUni.write({
    clipHomogeneousMatrixFromBoard: clipMatrixCpu,
    viewportHalfSizePixels: d.vec2f(viewportHalfWidthPixels, viewportHalfHeightPixels),
    markerDiskRadiusPixels: d.f32(RESULTS_MARKER_DISK_RADIUS_PX),
    displayedMarkerCount: d.u32(Math.max(0, Math.floor(input.pointCount))),
    orbitMarkerUniformReserve: d.vec2f(0, 0),
  })

  input.axisUni.write({
    clipHomogeneousMatrixFromBoard: clipMatrixCpu,
    viewportHalfSizePixels: d.vec2f(viewportHalfWidthPixels, viewportHalfHeightPixels),
    axisPolylineHalfStrokePixels: d.f32(axisPolylineHalfStrokePixels),
    worldAxisReachDistanceBoardUnits: d.f32(WORLD_AXIS_HALF_LEN),
    axisOriginBoard: d.vec3f(0, 0, 0),
  })

  input.tagQuadUni.write({
    clipHomogeneousMatrixFromBoard: clipMatrixCpu,
  })
}
