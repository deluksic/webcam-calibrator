/**
 * Levenberg-Marquardt Optimization for Camera Calibration
 *
 * Minimizes reprojection error iteratively.
 * Reference: opencv/modules/core/src/levmarq.cpp
 */

import type { Vec3 } from './vec3'
import type { Intrinsics } from './zhangCalibration'
import type { DistortionCoeffs } from './distortion'
import { distortPoint } from './distortion'
import { rodriguesToMatrix } from './rodrigues'
import { Mat3 } from './mat3'

/** Camera parameters for optimization */
export interface CameraParams {
  intrinsics: Intrinsics
  distortion: DistortionCoeffs
}

/** Per-view extrinsics */
export interface ViewParams {
  rvec: Vec3
  tvec: Vec3
}

/** Observation: 3D world point -> 2D image point */
export interface Observation {
  worldPt: Vec3
  imagePt: { u: number; v: number }
}

/** LM configuration */
export interface LmConfig {
  maxIter: number
  tau: number
  eps1: number
  eps2: number
  eps3: number
  delta: number
}

/** LM result */
export interface LmResult {
  converged: boolean
  iterations: number
  initialError: number
  finalError: number
}

/** Default LM configuration */
export const DEFAULT_LM_CONFIG: LmConfig = {
  maxIter: 100,
  tau: 1e-3,
  eps1: 1e-6,
  eps2: 1e-6,
  eps3: 1e-6,
  delta: 1e-8,
}

const NUM_INTRINSICS = 13 // fx, fy, cx, cy, skew, k1-k8

/**
 * Project point with given parameters.
 */
function projectWithParams(
  worldPt: Vec3,
  rvec: Vec3,
  tvec: Vec3,
  params: number[]
): { u: number; v: number } {
  const R = rodriguesToMatrix(rvec)

  // Extrinsics
  const Xc = Mat3.mulVec(R, worldPt)
  const x = (Xc.x + tvec.x) / (Xc.z + tvec.z)
  const y = (Xc.y + tvec.y) / (Xc.z + tvec.z)

  // Distortion
  let dx = x, dy = y
  if (Math.abs(params[5]!) > 1e-12 || Math.abs(params[6]!) > 1e-12) {
    const d = distortPoint({ x, y }, {
      k1: params[5]!, k2: params[6]!, p1: params[7]!, p2: params[8]!,
      k3: params[9]!, k4: params[10]!, k5: params[11]!, k6: params[12]!,
    })
    dx = d.x
    dy = d.y
  }

  // Project: u = fx*dx + skew*dy + cx, v = fy*dy + cy
  return {
    u: params[0]! * dx + params[4]! * dy + params[2]!,
    v: params[1]! * dy + params[3]!,
  }
}

/**
 * Compute total reprojection error.
 */
function computeError(
  observations: Observation[],
  viewObs: number[][],
  params: number[],
  viewParams: number[][]
): number {
  let sumSq = 0
  for (let i = 0; i < observations.length; i++) {
    const obs = observations[i]!
    const viewIdx = viewObs[i]![0]!
    const vp = viewParams[viewIdx]!
    const proj = projectWithParams(obs.worldPt,
      { x: vp[0]!, y: vp[1]!, z: vp[2]! },
      { x: vp[3]!, y: vp[4]!, z: vp[5]! },
      params)
    const dx = proj.u - obs.imagePt.u
    const dy = proj.v - obs.imagePt.v
    sumSq += dx * dx + dy * dy
  }
  return Math.sqrt(sumSq / observations.length)
}

/**
 * Compute gradient for parameter update.
 * Uses simplified gradient descent with adaptive step.
 */
function computeGradient(
  observations: Observation[],
  viewObs: number[][],
  params: number[],
  viewParams: number[][],
  delta: number
): number[] {
  const grad = new Array(NUM_INTRINSICS).fill(0)
  const err = computeError(observations, viewObs, params, viewParams)

  if (err < 1e-10) return grad

  for (let p = 0; p < NUM_INTRINSICS; p++) {
    const paramsPlus = [...params]
    paramsPlus[p] = (params[p] ?? 0) + delta
    const errPlus = computeError(observations, viewObs, paramsPlus, viewParams)
    grad[p] = (errPlus - err) / delta
  }

  return grad
}

/**
 * Levenberg-Marquardt optimization for camera calibration.
 */
export function levmarqCalibration(
  observations: Observation[],
  viewIndices: number[][], // [viewIdx, pointIdx] for each observation
  initialParams: CameraParams,
  initialViewParams: ViewParams[],
  config = DEFAULT_LM_CONFIG
): {
  params: CameraParams
  viewParams: ViewParams[]
  result: LmResult
} {
  const numViews = initialViewParams.length

  // Flatten initial intrinsics
  let params = [
    initialParams.intrinsics.fx, initialParams.intrinsics.fy,
    initialParams.intrinsics.cx, initialParams.intrinsics.cy, initialParams.intrinsics.skew,
    initialParams.distortion.k1, initialParams.distortion.k2,
    initialParams.distortion.p1, initialParams.distortion.p2,
    initialParams.distortion.k3, initialParams.distortion.k4,
    initialParams.distortion.k5, initialParams.distortion.k6,
  ]

  // Flatten view params
  let viewParams = initialViewParams.map(v => [v.rvec.x, v.rvec.y, v.rvec.z, v.tvec.x, v.tvec.y, v.tvec.z])

  let nu = config.tau
  let iter = 0
  let converged = false

  const initErr = computeError(observations, viewIndices, params, viewParams)
  let err = initErr

  for (iter = 0; iter < config.maxIter; iter++) {
    // Compute gradient
    const grad = computeGradient(observations, viewIndices, params, viewParams, config.delta)

    // Gradient norm
    let gradNorm = 0
    for (let i = 0; i < NUM_INTRINSICS; i++) {
      gradNorm += grad[i]! * grad[i]!
    }
    gradNorm = Math.sqrt(gradNorm)

    // Check gradient convergence
    if (gradNorm < config.eps1) {
      converged = true
      break
    }

    // Simple LM update: x_new = x - gradient / (grad_norm + nu)
    const newParams = [...params]
    for (let i = 0; i < NUM_INTRINSICS; i++) {
      const grad_i = grad[i] ?? 0
      newParams[i] = (params[i] ?? 0) - grad_i / (gradNorm + nu)
    }

    // Clamp to reasonable ranges
    newParams[0] = Math.max(10, (newParams[0] ?? 0)) // fx
    newParams[1] = Math.max(10, (newParams[1] ?? 0)) // fy
    newParams[2] = Math.max(0, (newParams[2] ?? 0)) // cx
    newParams[3] = Math.max(0, (newParams[3] ?? 0)) // cy

    const newErr = computeError(observations, viewIndices, newParams, viewParams)

    // Accept or reject step
    if (newErr < err) {
      params = newParams
      err = newErr
      nu = Math.max(1e-10, nu * 0.1)
    } else {
      nu = nu * 10
    }

    // Check error convergence
    if (Math.abs(err - newErr) < config.eps3 * err || err < config.eps3) {
      converged = true
      break
    }
  }

  return {
    params: {
      intrinsics: {
        fx: params[0]!, fy: params[1]!, cx: params[2]!, cy: params[3]!, skew: params[4]!,
      },
      distortion: {
        k1: params[5]!, k2: params[6]!, p1: params[7]!, p2: params[8]!,
        k3: params[9]!, k4: params[10]!, k5: params[11]!, k6: params[12]!,
      },
    },
    viewParams: viewParams.map(vp => ({
      rvec: { x: vp[0]!, y: vp[1]!, z: vp[2]! },
      tvec: { x: vp[3]!, y: vp[4]!, z: vp[5]! },
    })),
    result: {
      converged,
      iterations: iter,
      initialError: initErr,
      finalError: err,
    },
  }
}

/**
 * Simple gradient descent refinement for initial calibration.
 * More robust than full LM for initial refinement.
 */
export function refineCalibration(
  observations: Observation[],
  viewIndices: number[][],
  initialParams: CameraParams,
  initialViewParams: ViewParams[],
  maxIter = 50,
  stepSize = 0.1
): CameraParams {
  const numViews = initialViewParams.length

  // Flatten params
  let params = [
    initialParams.intrinsics.fx, initialParams.intrinsics.fy,
    initialParams.intrinsics.cx, initialParams.intrinsics.cy, initialParams.intrinsics.skew,
    initialParams.distortion.k1, initialParams.distortion.k2,
    initialParams.distortion.p1, initialParams.distortion.p2,
    initialParams.distortion.k3, initialParams.distortion.k4,
    initialParams.distortion.k5, initialParams.distortion.k6,
  ]

  let viewParams = initialViewParams.map(v => [v.rvec.x, v.rvec.y, v.rvec.z, v.tvec.x, v.tvec.y, v.tvec.z])

  const err = computeError(observations, viewIndices, params, viewParams)

  // Adaptive step size
  let alpha = stepSize

  for (let iter = 0; iter < maxIter; iter++) {
    const grad = computeGradient(observations, viewIndices, params, viewParams, 1e-6)

    // Update parameters
    const newParams = [...params]
    for (let i = 0; i < NUM_INTRINSICS; i++) {
      newParams[i] = (params[i] ?? 0) - alpha * (grad[i] ?? 0)
    }

    // Clamp intrinsic bounds
    newParams[0] = Math.max(100, (newParams[0] ?? 0))
    newParams[1] = Math.max(100, (newParams[1] ?? 0))
    newParams[2] = Math.max(0, Math.min(2000, (newParams[2] ?? 0)))
    newParams[3] = Math.max(0, Math.min(2000, (newParams[3] ?? 0)))

    const newErr = computeError(observations, viewIndices, newParams, viewParams)

    if (newErr < err) {
      params = newParams
      alpha *= 1.5 // Increase step on success
    } else {
      alpha *= 0.5 // Decrease step on failure
      if (alpha < 1e-10) break
    }

    if (Math.abs(err - newErr) < 1e-8 * err) break
  }

  return {
    intrinsics: {
      fx: params[0]!, fy: params[1]!, cx: params[2]!, cy: params[3]!, skew: params[4]!,
    },
    distortion: {
      k1: params[5]!, k2: params[6]!, p1: params[7]!, p2: params[8]!,
      k3: params[9]!, k4: params[10]!, k5: params[11]!, k6: params[12]!,
    },
  }
}
