// Geometry helpers for homography-based extrinsic recovery (used in live overlay).

import type { CameraIntrinsics } from '@/lib/cameraModel'

export type Mat3 = [number, number, number, number, number, number, number, number, number]
export type Vec3 = { x: number; y: number; z: number }

function kInverse(k: CameraIntrinsics): Mat3 {
  return [1 / k.fx, 0, -k.cx / k.fx, 0, 1 / k.fy, -k.cy / k.fy, 0, 0, 1]
}

function matMul3(a: Mat3, b: Mat3): Mat3 {
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

function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[1]! * b[2]! - a[2]! * b[1]!, a[2]! * b[0]! - a[0]! * b[2]!, a[0]! * b[1]! - a[1]! * b[0]!]
}

function len3(v: [number, number, number]): number {
  return Math.hypot(v[0]!, v[1]!, v[2]!)
}

/**
 * M = K^{-1} H; R,t from first two columns; orthonormalize R.
 */
export function extrinsicsFromHomography(h: Mat3, k: CameraIntrinsics): { R: Mat3; t: Vec3 } | undefined {
  const kInv = kInverse(k)
  if (Math.abs(kInv[0]!) < 1e-15) {
    return undefined
  }
  const m = matMul3(kInv, h)
  const m0: [number, number, number] = [m[0]!, m[3]!, m[6]!]
  const m1: [number, number, number] = [m[1]!, m[4]!, m[7]!]
  const m2: [number, number, number] = [m[2]!, m[5]!, m[8]!]
  const l0 = len3(m0)
  const l1 = len3(m1)
  if (l0 < 1e-15 || l1 < 1e-15) {
    return undefined
  }
  const la = 2 / (l0 + l1)
  const r1: [number, number, number] = [la * m0[0]!, la * m0[1]!, la * m0[2]!]
  const r2: [number, number, number] = [la * m1[0]!, la * m1[1]!, la * m1[2]!]
  const r3v = cross(r1, r2)
  const t: Vec3 = { x: la * m2[0]!, y: la * m2[1]!, z: la * m2[2]! }
  const R: Mat3 = [r1[0]!, r2[0]!, r3v[0]!, r1[1]!, r2[1]!, r3v[1]!, r1[2]!, r2[2]!, r3v[2]!]
  return { R, t }
}

/** 3×3 rotation matrix (row-major) → Rodrigues vector. */
export function matrixToRvec(R: Mat3): [number, number, number] {
  const trace = R[0]! + R[4]! + R[8]!
  const cosTheta = (trace - 1) * 0.5
  if (cosTheta > 1 - 1e-12) {
    return [0, 0, 0]
  }
  const theta = Math.acos(Math.max(-1, Math.min(1, cosTheta)))
  const scale = theta / (2 * Math.sin(theta))
  return [(R[7]! - R[5]!) * scale, (R[2]! - R[6]!) * scale, (R[3]! - R[1]!) * scale]
}
