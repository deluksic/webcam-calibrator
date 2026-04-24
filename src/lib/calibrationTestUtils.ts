/**
 * Shared row-major 3x3 utilities for synthetic Zhang / calibration tests (not used in app runtime).
 */
import type { Mat3 } from '@/lib/geometry'
import type { Mat3R } from '@/lib/zhangCalibration'

export function matMul3(
  a: readonly [number, number, number, number, number, number, number, number, number],
  b: Mat3,
): Mat3 {
  return [
    a[0]! * b[0]! + a[1]! * b[3]! + a[2]! * b[6]!,
    a[0]! * b[1]! + a[1]! * b[4]! + a[2]! * b[7]!,
    a[0]! * b[2]! + a[1]! * b[5]! + a[2]! * b[8]!,
    a[3]! * b[0]! + a[4]! * b[3]! + a[5]! * b[6]!,
    a[3]! * b[1]! + a[4]! * b[4]! + a[5]! * b[7]!,
    a[3]! * b[2]! + a[4]! * b[5]! + a[5]! * b[8]!,
    a[6]! * b[0]! + a[7]! * b[3]! + a[8]! * b[6]!,
    a[6]! * b[1]! + a[7]! * b[4]! + a[8]! * b[7]!,
    a[6]! * b[2]! + a[7]! * b[5]! + a[8]! * b[8]!,
  ]
}

/** Rotation about Y (right-handed), radians; row-major. */
export function rotY(rad: number): Mat3R {
  const c = Math.cos(rad)
  const s = Math.sin(rad)
  return [c, 0, s, 0, 1, 0, -s, 0, c]
}

/** Rotation about X (right-handed), radians; row-major. */
export function rotX(rad: number): Mat3R {
  const c = Math.cos(rad)
  const s = Math.sin(rad)
  return [1, 0, 0, 0, c, s, 0, -s, c]
}

/** Rotation about Z (right-handed), radians; row-major. */
export function rotZ(rad: number): Mat3R {
  const c = Math.cos(rad)
  const s = Math.sin(rad)
  return [c, s, 0, -s, c, 0, 0, 0, 1]
}

/** Compose two rotations: result = R1 * R2 */
export function composeRotations(R1: Mat3R, R2: Mat3R): Mat3R {
  const a: Mat3R = R1 as unknown as Mat3R
  const b: Mat3R = R2 as unknown as Mat3R
  return [
    a[0]! * b[0]! + a[1]! * b[3]! + a[2]! * b[6]!,
    a[0]! * b[1]! + a[1]! * b[4]! + a[2]! * b[7]!,
    a[0]! * b[2]! + a[1]! * b[5]! + a[2]! * b[8]!,
    a[3]! * b[0]! + a[4]! * b[3]! + a[5]! * b[6]!,
    a[3]! * b[1]! + a[4]! * b[4]! + a[5]! * b[7]!,
    a[3]! * b[2]! + a[4]! * b[5]! + a[5]! * b[8]!,
    a[6]! * b[0]! + a[7]! * b[3]! + a[8]! * b[6]!,
    a[6]! * b[1]! + a[7]! * b[4]! + a[8]! * b[7]!,
    a[6]! * b[2]! + a[7]! * b[5]! + a[8]! * b[8]!,
  ]
}

/** Plane (X,Y) -> image: H = K [r0 r1 t] with r0,r1 first two columns of R. */
export function homographyFromKT(K: [number, number, number, number, number, number, number, number, number], R: Mat3R, tx: number, ty: number, tz: number): Mat3 {
  const m: Mat3 = [R[0]!, R[1]!, tx, R[3]!, R[4]!, ty, R[6]!, R[7]!, tz] as const
  const h = matMul3(K, m)
  const w = h[8]!
  if (Math.abs(w) < 1e-15) {
    return h
  }
  return h.map((x) => x / w) as unknown as Mat3
}
