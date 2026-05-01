import type { ExtractBindGroupInputFromLayout, TgpuRoot } from 'typegpu'
import { d, tgpu } from 'typegpu'

/** Triangulation subdiv for `circle()`; see `circleVertexCount`. */
export const MARKER_DISK_SUBDIV = 2 as const

/** Radius in framebuffer pixels (~CSS pixel at 1:1 backing store); disks are billboarded so size stays fixed on screen. */
export const RESULTS_MARKER_DISK_RADIUS_PX = 8 as const

export const MAX_RESULTS_MARKER_POINTS = 8192 as const

/** One calibration target in board space (storage buffer element). */
export const MarkerCenter = d.struct({
  positionBoardUnits: d.vec3f,
})

/** CPU/GPU marker row (`createBuffer`/shader storage element type). */
export type MarkerCenterRow = d.Infer<typeof MarkerCenter>

const MarkerCentersSchema = d.arrayOf(MarkerCenter, MAX_RESULTS_MARKER_POINTS)

/** Marker pass: full-view clip + viewport (for pixel-radius disks) + count. Disks billboard in clip using `circle()` × radius / half viewport. */
const MarkerUniformStruct = d.struct({
  clipHomogeneousMatrixFromBoard: d.mat4x4f,
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

/** Same projection as markers; endpoints are reconstructed in-shader along ±board axes from `axisOriginBoard`. */
const AxisUniformStruct = d.struct({
  clipHomogeneousMatrixFromBoard: d.mat4x4f,
  viewportHalfSizePixels: d.vec2f,
  /** Half stroking width (~1.5 ⇒ ~3 px line) for axis arrows, in framebuffer pixels. */
  axisPolylineHalfStrokePixels: d.f32,
  /** Each RGB arm runs from `axisOriginBoard` ± this length along the corresponding axis (`WORLD_AXIS_HALF_LEN` on CPU). */
  worldAxisReachDistanceBoardUnits: d.f32,
  /** Origin where the RGB axes are planted (board space); fixed at **`(0,0,0)`** on CPU. */
  axisOriginBoard: d.vec3f,
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

export const MAX_RESULTS_TAG_QUADS = 2048 as const

/**
 * One calibrated tag quad on the GPU: four board-space corners (TL/TR/BL/BR) and the row-major
 * 6×6 interior bit pattern packed into 36 bits across two `u32`s (`.x` = bits 0..31, `.y` = 32..35).
 */
export const TagQuad = d.struct({
  corners: d.arrayOf(d.vec3f, 4),
  packedPattern: d.vec2u,
})

export type TagQuadRow = d.Infer<typeof TagQuad>

const TagQuadsSchema = d.arrayOf(TagQuad, MAX_RESULTS_TAG_QUADS)

const TagQuadUniformStruct = d.struct({
  clipHomogeneousMatrixFromBoard: d.mat4x4f,
})

export const tagQuadsBindLayout = tgpu.bindGroupLayout({
  tagQuadUniform: { uniform: TagQuadUniformStruct },
  tags: { storage: TagQuadsSchema, access: 'readonly' },
})

export type TagQuadsGpuBindValues = ExtractBindGroupInputFromLayout<typeof tagQuadsBindLayout.entries>

export function allocTagQuads(root: TgpuRoot) {
  return root.createBuffer(TagQuadsSchema).$usage('storage')
}

export function allocTagQuadUni(root: TgpuRoot) {
  return root.createBuffer(TagQuadUniformStruct).$usage('uniform')
}

export type TagQuadsResultsBuffer = ReturnType<typeof allocTagQuads>
export type TagQuadResultsUniformGpuBuffer = ReturnType<typeof allocTagQuadUni>
