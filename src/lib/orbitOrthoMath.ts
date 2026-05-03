import type { Mat4Arg, Vec3Arg } from 'wgpu-matrix'
import { mat4, vec3 } from 'wgpu-matrix'

import type { Point3 } from '@/lib/calibrationTypes'

/**
 * Reused mats/vecs for wgpu-matrix outputs so we do not allocate 16-float arrays (and a few vec3s)
 * on every drag step or every results frame.
 */
const viewMat: Mat4Arg = Array.from({ length: 16 }, () => 0)
const projMat: Mat4Arg = Array.from({ length: 16 }, () => 0)
/** Eye position in board space (then passed to lookAt). */
const orbitEyeWorldPos: Vec3Arg = [0, 0, 0]
const orbitLookTarget: Vec3Arg = [0, 0, 0]
const worldUp: Vec3Arg = [0, 1, 0]
const orbitPitchAxis: Vec3Arg = [0, 0, 0]
const orbitRotMat: Mat4Arg = Array.from({ length: 16 }, () => 0)

const ORBIT_EYE_RADIUS = 15

/** Unit direction from board origin toward the eye; turntable yaw around world +Y. `from` is not modified. */
export function applyOrbitYawWorldY(from: Vec3Arg, deltaYawRad: number, out: Vec3Arg): Vec3Arg {
  mat4.rotationY(deltaYawRad, orbitRotMat)
  vec3.transformMat4(from, orbitRotMat, out)
  vec3.normalize(out, out)
  return out
}

/**
 * Pitch in the vertical plane spanned by world up and the current eye ray: rotate around
 * `normalize(cross(worldUp, eyeDir))`. Near ±Y, copies/clamps `from` into `out` without rotating.
 * `from` is not modified.
 */
export function applyOrbitPitchVerticalPlane(from: Vec3Arg, deltaPitchRad: number, out: Vec3Arg): Vec3Arg {
  vec3.cross(worldUp, from, orbitPitchAxis)
  const axLen = vec3.len(orbitPitchAxis)
  if (axLen < 1e-6) {
    vec3.copy(from, out)
    vec3.normalize(out, out)
    return out
  }
  vec3.scale(orbitPitchAxis, 1 / axLen, orbitPitchAxis)
  mat4.axisRotation(orbitPitchAxis, deltaPitchRad, orbitRotMat)
  vec3.transformMat4(from, orbitRotMat, out)
  vec3.normalize(out, out)
  return out
}

/**
 * Orthographic clip transform (board → clip): `projection * view * vec4(board, 1)` with `wgpu-matrix`
 * column-major layouts (matches `std.mul` / TypeGPU mat×vec). The eye sits at fixed radius from
 * `lookAtBoard` along `eyeDirUnit` (typically **`lookAtBoard = (0,0,0)`**).
 */
export function buildOrbitClipMatrix(
  lookAtBoard: Point3,
  aspectWidthOverHeight: number,
  eyeDirUnit: Vec3Arg,
  orthoExtentY: number,
  mvpOut: Mat4Arg,
): void {
  orbitLookTarget[0] = lookAtBoard.x
  orbitLookTarget[1] = lookAtBoard.y
  orbitLookTarget[2] = lookAtBoard.z
  vec3.copy(eyeDirUnit, orbitEyeWorldPos)
  vec3.normalize(orbitEyeWorldPos, orbitEyeWorldPos)
  vec3.scale(orbitEyeWorldPos, ORBIT_EYE_RADIUS, orbitEyeWorldPos)
  vec3.add(orbitEyeWorldPos, orbitLookTarget, orbitEyeWorldPos)
  mat4.lookAt(orbitEyeWorldPos, orbitLookTarget, worldUp, viewMat)
  const halfY = orthoExtentY * 0.5
  const halfX = halfY * aspectWidthOverHeight
  mat4.ortho(-halfX, halfX, -halfY, halfY, 0.1, 500, projMat)
  mat4.multiply(projMat, viewMat, mvpOut)
}

export const WORLD_AXIS_HALF_LEN = 3
