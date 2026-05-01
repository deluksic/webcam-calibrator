import type { ExtractBindGroupInputFromLayout, TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'

/** Triangulation subdiv for `circle()`; see `circleVertexCount`. */
export const MARKER_DISK_SUBDIV = 2 as const

/** Radius in framebuffer pixels (~CSS pixel at 1:1 backing store); disks are billboarded so size stays fixed on screen. */
export const RESULTS_MARKER_DISK_RADIUS_PX = 8 as const

export const MAX_RESULTS_MARKER_POINTS = 8192 as const

/** One calibration target in centroid-relative board space (storage buffer element). */
export const MarkerCenterCentroidBoard = d.struct({
  positionCentroidRelativeBoardUnits: d.vec3f,
})

/** CPU/GPU marker row (`createBuffer`/shader storage element type). */
export type MarkerCenterCentroidBoardRow = d.Infer<typeof MarkerCenterCentroidBoard>

const MarkerCentersSchema = d.arrayOf(MarkerCenterCentroidBoard, MAX_RESULTS_MARKER_POINTS)

/** Marker pass: full-view clip + viewport (for pixel-radius disks) + count. Disks billboard in clip using `circle()` × radius / half viewport. */
const MarkerUniformStruct = d.struct({
  clipHomogeneousMatrixFromCentroidBoard: d.mat4x4f,
  viewportHalfSizePixels: d.vec2f,
  markerDiskRadiusPixels: d.f32,
  displayedMarkerCount: d.u32,
  orbitMarkerUniformReserve: d.vec2f,
})

export const markersBindLayout = tgpu.bindGroupLayout({
  markerUniform: { uniform: MarkerUniformStruct },
  centers: {
    storage: MarkerCentersSchema,
    access: 'readonly',
  },
})

export type MarkersGpuBindValues = ExtractBindGroupInputFromLayout<typeof markersBindLayout.entries>

/** Same projection as markers; endpoints are reconstructed in-shader along ±board axes from the origin. */
const AxisUniformStruct = d.struct({
  clipHomogeneousMatrixFromCentroidBoard: d.mat4x4f,
  viewportHalfSizePixels: d.vec2f,
  /** Half stroking width (~1.5 ⇒ ~3 px line) for axis arrows, in framebuffer pixels. */
  axisPolylineHalfStrokePixels: d.f32,
  /** Each RGB arm runs from centroid-board origin ± this length along the corresponding axis (`WORLD_AXIS_HALF_LEN` on CPU). */
  worldAxisReachDistanceCentroidBoardUnits: d.f32,
})

export const axisBindLayout = tgpu.bindGroupLayout({
  axisUniform: { uniform: AxisUniformStruct },
})

export type AxisGpuBindValues = ExtractBindGroupInputFromLayout<typeof axisBindLayout.entries>

export function allocMarkerUni(root: TgpuRoot) {
  return root.createBuffer(MarkerUniformStruct).$usage('uniform')
}

export function allocMarkersCenters(root: TgpuRoot) {
  return root.createBuffer(MarkerCentersSchema).$usage('storage')
}

export function allocAxisUni(root: TgpuRoot) {
  return root.createBuffer(AxisUniformStruct).$usage('uniform')
}

export type MarkerResultsUniformGpuBuffer = ReturnType<typeof allocMarkerUni>
export type AxisResultsUniformGpuBuffer = ReturnType<typeof allocAxisUni>
