// WASM calibration wrapper using @deluksic/opencv-calibration-wasm.

import { calibrateCameraRO, projectPoints } from '@deluksic/opencv-calibration-wasm'
import type { Mat3 } from '@/lib/geometry'
import type { CameraIntrinsics, RationalDistortion8 } from '@/lib/cameraModel'
import type { CalibrationFrameObservation, LabeledPoint } from '@/lib/calibrationTypes'
import type { TargetLayout } from '@/lib/targetLayout'

export type Mat3 = [number, number, number, number, number, number, number, number, number]
export type Vec3 = { x: number; y: number; z: number }

// Rodrigues vector → 3×3 rotation matrix (row-major).
function rvecToMatrix(rvec: [number, number, number]): Mat3 {
  const theta = Math.hypot(rvec[0], rvec[1], rvec[2])
  if (theta < 1e-12) {
    return [1, 0, 0, 0, 1, 0, 0, 0, 1]
  }
  const ux = rvec[0] / theta
  const uy = rvec[1] / theta
  const uz = rvec[2] / theta
  const c = Math.cos(theta)
  const s = Math.sin(theta)
  const t = 1 - c
  return [
    t * ux * ux + c,
    t * ux * uy - s * uz,
    t * ux * uz + s * uy,
    t * ux * uy + s * uz,
    t * uy * uy + c,
    t * uy * uz - s * ux,
    t * ux * uz - s * uy,
    t * uy * uz + s * ux,
    t * uz * uz + c,
  ]
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
  return [
    (R[7]! - R[5]!) * scale,
    (R[2]! - R[6]!) * scale,
    (R[3]! - R[1]!) * scale,
  ]
}

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

function cameraIntrinsicsFromMatrix(
  cm: [[number, number, number], [number, number, number], [number, number, number]],
): CameraIntrinsics {
  return { fx: cm[0]![0]!, fy: cm[1]![1]!, cx: cm[0]![2]!, cy: cm[1]![2]! }
}

function padDistortion(coeffs: number[]): RationalDistortion8 {
  return [coeffs[0] ?? 0, coeffs[1] ?? 0, coeffs[2] ?? 0, coeffs[3] ?? 0, coeffs[4] ?? 0, coeffs[5] ?? 0, coeffs[6] ?? 0, coeffs[7] ?? 0]
}

export type CalibrationOk = {
  kind: 'ok'
  K: CameraIntrinsics
  distortion: RationalDistortion8
  homographies: { frameId: number; h: Mat3 }[]
  extrinsics: { frameId: number; R: Mat3; t: Vec3 }[]
  rmsPx: number
  perFrameRmsPx: Map<number, number>
}

export type CalibrationErr = { kind: 'error'; reason: 'too-few-views' | 'singular' | 'non-physical' }

export type CalibrationResult = CalibrationOk | CalibrationErr

/**
 * Build WASM input from current data model.
 * Only includes pointIds that appear in every frame (WASM requirement).
 */
function buildWasmInput(
  labeledPoints: LabeledPoint[],
  frames: CalibrationFrameObservation[],
): { objectPoints: [number, number, number][][]; imagePoints: [number, number][][]; sharedPointIds: number[] } | undefined {
  const framePointSets = frames.map((f) => new Set(f.framePoints.map((fp) => fp.pointId)))
  const sharedPointIds = [...framePointSets[0]!].filter((pid) => framePointSets.every((s) => s.has(pid)))

  if (sharedPointIds.length < 6) {
    return undefined
  }

  const objectPoints: [number, number, number][][] = frames.map(() =>
    sharedPointIds.map((pid) => {
      const lp = labeledPoints.find((l) => l.pointId === pid)
      return [lp!.plane.x, lp!.plane.y, 0]
    }),
  )

  const imagePoints: [number, number][][] = frames.map((f) =>
    sharedPointIds.map((pid) => {
      const fp = f.framePoints.find((p) => p.pointId === pid)!
      return [fp.imagePoint.x, fp.imagePoint.y]
    }),
  )

  return { objectPoints, imagePoints, sharedPointIds }
}

export async function solveCalibration(
  layout: TargetLayout,
  labeledPoints: LabeledPoint[],
  frames: readonly CalibrationFrameObservation[],
): Promise<CalibrationResult> {
  const imageSize = { width: 1280, height: 720 }

  const wasmInput = buildWasmInput(labeledPoints, [...frames] as CalibrationFrameObservation[])
  if (!wasmInput || wasmInput.objectPoints.length < 3) {
    return { kind: 'error', reason: 'too-few-views' }
  }

  const { objectPoints, imagePoints, sharedPointIds } = wasmInput

  // Call WASM calibrateCameraRO
  const wasmResult = await calibrateCameraRO({
    objectPoints,
    imagePoints,
    imageSize,
  })

  const K = cameraIntrinsicsFromMatrix(wasmResult.cameraMatrix)
  const distortion = padDistortion(wasmResult.distortionCoefficients)

  // Compute per-frame RMS via projectPoints
  const projected = await projectPoints({
    objectPoints,
    rvecs: wasmResult.rvecs,
    tvecs: wasmResult.tvecs,
    cameraMatrix: wasmResult.cameraMatrix,
    distortionCoefficients: wasmResult.distortionCoefficients,
  })

  const perFrameRmsPx = new Map<number, number>()
  let allSq: number[] = []
  for (let view = 0; view < frames.length; view++) {
    const fo = frames[view]!
    const proj = projected.projectedImagePoints[view]!
    let sq: number[] = []
    for (let p = 0; p < sharedPointIds.length; p++) {
      const dx = proj[p]![0]! - imagePoints[view]![p]![0]!
      const dy = proj[p]![1]! - imagePoints[view]![p]![1]!
      sq.push(dx * dx + dy * dy)
    }
    if (sq.length > 0) {
      perFrameRmsPx.set(fo.frameId, Math.sqrt(sq.reduce((a, b) => a + b, 0) / sq.length))
      allSq.push(...sq)
    }
  }
  const rmsPx = allSq.length > 0 ? Math.sqrt(allSq.reduce((a, b) => a + b, 0) / allSq.length) : 0

  // Convert extrinsics
  const extrinsics: { frameId: number; R: Mat3; t: Vec3 }[] = []
  for (let i = 0; i < wasmResult.rvecs.length; i++) {
    const rvec = wasmResult.rvecs[i]!
    const tvec = wasmResult.tvecs[i]!
    const frameId = frames[i]?.frameId ?? i
    extrinsics.push({
      frameId,
      R: rvecToMatrix(rvec),
      t: { x: tvec[0]!, y: tvec[1]!, z: tvec[2]! },
    })
  }

  return {
    kind: 'ok',
    K,
    distortion,
    homographies: [],
    extrinsics,
    rmsPx,
    perFrameRmsPx,
  }
}
