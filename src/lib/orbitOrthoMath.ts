import { mat4, vec4 } from 'wgpu-matrix'

export function orbitEye(radius: number, yaw: number, pitch: number, out: Float32Array): void {
  const cosP = Math.cos(pitch)
  const sinP = Math.sin(pitch)
  const cosY = Math.cos(yaw)
  const sinY = Math.sin(yaw)
  out[0] = radius * cosP * sinY
  out[1] = radius * sinP
  out[2] = radius * cosP * cosY
}

const tmpView = new Float32Array(16)
const tmpProj = new Float32Array(16)
const eyeScratch = new Float32Array(3)
const boardTarget = new Float32Array([0, 0, 0])
const worldUp = new Float32Array([0, 1, 0])
const clipScratch4 = vec4.create()

const ORBIT_EYE_RADIUS = 15

/**
 * Orthographic clip transform (centroid-board → clip), same composition as phong-reflection:
 * `projection * view * vec4(board, 1)` with `wgpu-matrix` column-major layouts (matches `std.mul` / TypeGPU mat×vec).
 */
export function buildClipHomogeneousMatrixFromCentroidBoard(
  aspectWidthOverHeight: number,
  yawRad: number,
  pitchRad: number,
  orthoExtentY: number,
  mvpOut: Float32Array,
): void {
  orbitEye(ORBIT_EYE_RADIUS, yawRad, pitchRad, eyeScratch)
  mat4.lookAt(eyeScratch, boardTarget, worldUp, tmpView)
  const halfY = orthoExtentY * 0.5
  const halfX = halfY * aspectWidthOverHeight
  mat4.ortho(-halfX, halfX, -halfY, halfY, 0.1, 500, tmpProj)
  mat4.multiply(tmpProj, tmpView, mvpOut)
}

/** @deprecated Use `buildClipHomogeneousMatrixFromCentroidBoard`. */
export const buildResultsMvp = buildClipHomogeneousMatrixFromCentroidBoard

/** Project centroid-board XYZ → NDC XY in [-1, 1]² (ignores final Z). */
export function projectWorldToNdcXy(
  mvp: Float32Array,
  x: number,
  y: number,
  z: number,
  out: Float32Array,
): void {
  vec4.set(x, y, z, 1, clipScratch4)
  vec4.transformMat4(clipScratch4, mvp, clipScratch4)
  const w = Math.abs(clipScratch4[3]!) > 1e-12 ? clipScratch4[3]! : 1
  out[0] = clipScratch4[0]! / w
  out[1] = clipScratch4[1]! / w
}

export const WORLD_AXIS_HALF_LEN = 5
