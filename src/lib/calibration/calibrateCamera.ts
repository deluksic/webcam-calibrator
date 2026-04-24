/**
 * Complete Camera Calibration (matching OpenCV calibrateCamera)
 *
 * Pipeline:
 * 1. Compute homographies for each view using normalized DLT
 * 2. Solve for initial K via Zhang's closed-form method
 * 3. Compute initial extrinsics (R, t) for each view
 * 4. Run Levenberg-Marquardt optimization to minimize reprojection error
 *
 * Reference: opencv/modules/calib3d/src/calibration.cpp
 */

import type { Vec3 } from './vec3'
import { Mat3 } from './mat3'
import type { Intrinsics, Homography } from './zhangCalibration'
import { zhangCalibration, computeExtrinsicsFromHomography } from './zhangCalibration'
import { rodriguesToMatrix } from './rodrigues'
import type { DistortionCoeffs } from './distortion'
import { zeroDistortion } from './distortion'
import { refineCalibration } from './levmarq'
import { computeHomographyNormalized } from './dltHomography'

/** Observation: 3D world point -> 2D image point */
export interface CalibrationObservation {
  /** 3D object point */
  objectPt: Vec3
  /** 2D image point */
  imagePt: { u: number; v: number }
}

/** Per-view calibration data */
export interface CalibrationView {
  objectPoints: CalibrationObservation[]
  imagePoints: CalibrationObservation[]
}

/** Complete calibration result (matches OpenCV format) */
export interface CalibResult {
  /** 3x3 camera matrix K */
  cameraMatrix: Mat3
  /** Distortion coefficients [k1, k2, p1, p2, k3] */
  distCoeffs: number[]
  /** Per-view rotation vectors (Rodrigues) */
  rvecs: Vec3[]
  /** Per-view translation vectors */
  tvecs: Vec3[]
  /** RMS reprojection error in pixels */
  rmsError: number
  /** Whether optimization converged */
  converged: boolean
  /** Number of iterations */
  iterations: number
}

/**
 *
 * Run complete camera calibration pipeline.
 */
export function calibrateCamera(
  views: CalibrationView[],
  imageSize: { width: number; height: number }
): CalibResult {
  if (views.length < 2) {
    throw new Error('Need at least 2 views for calibration')
  }

  // Step 1: Compute homographies for each view using normalized DLT
  const homographies: Homography[] = []
  for (const view of views) {
    const objectPts = view.objectPoints.map(o => ({ x: o.objectPt.x, y: o.objectPt.y }))
    const imagePts = view.imagePoints.map(i => ({ x: i.imagePt.u, y: i.imagePt.v }))
    const H = computeHomographyNormalized(objectPts, imagePts)
    homographies.push({ H })
  }

  if (homographies.length < 3) {
    throw new Error('Need at least 3 valid homographies')
  }

  // Step 2: Initial calibration via Zhang
  const zhangResult = zhangCalibration(homographies)
  const initialIntrinsics = zhangResult.intrinsics

  // Step 3: Build observations for refinement
  const observations: Array<{
    worldPt: Vec3
    imagePt: { u: number; v: number }
  }> = []
  const viewIndices: number[][] = []

  for (let v = 0; v < views.length; v++) {
    const view = views[v]!
    for (let p = 0; p < view.objectPoints.length; p++) {
      observations.push({
        worldPt: view.objectPoints[p]!.objectPt,
        imagePt: view.imagePoints[p]!.imagePt,
      })
      viewIndices.push([v, p])
    }
  }

  // Initial view params from Zhang extrinsics
  const initialViewParams = zhangResult.extrinsics.map(e => ({
    rvec: e.rvec,
    tvec: e.tvec,
  }))

  // Step 4: LM refinement
  const refined = refineCalibration(
    observations,
    viewIndices,
    { intrinsics: initialIntrinsics, distortion: zeroDistortion() },
    initialViewParams,
    100,
    0.5
  )

  // Build final result
  const cameraMatrix: Mat3 = [
    refined.intrinsics.fx, refined.intrinsics.skew, refined.intrinsics.cx,
    0, refined.intrinsics.fy, refined.intrinsics.cy,
    0, 0, 1,
  ]

  const distCoeffs = [
    refined.distortion.k1,
    refined.distortion.k2,
    refined.distortion.p1,
    refined.distortion.p2,
    refined.distortion.k3,
  ]

  // Compute RMS error
  let sumSq = 0
  for (let i = 0; i < observations.length; i++) {
    const obs = observations[i]!
    const viewIdx = viewIndices[i]![0]!
    const rvec = initialViewParams[viewIdx]!.rvec
    const tvec = initialViewParams[viewIdx]!.tvec

    const R = rodriguesToMatrix(rvec)
    const Xc = Mat3.mulVec(R, obs.worldPt)
    const x = (Xc.x + tvec.x) / (Xc.z + tvec.z)
    const y = (Xc.y + tvec.y) / (Xc.z + tvec.z)

    const u = refined.intrinsics.fx * x + refined.intrinsics.skew * y + refined.intrinsics.cx
    const v = refined.intrinsics.fy * y + refined.intrinsics.cy

    const dx = u - obs.imagePt.u
    const dy = v - obs.imagePt.v
    sumSq += dx * dx + dy * dy
  }
  const rmsError = Math.sqrt(sumSq / observations.length)

  return {
    cameraMatrix,
    distCoeffs,
    rvecs: initialViewParams.map(v => v.rvec),
    tvecs: initialViewParams.map(v => v.tvec),
    rmsError,
    converged: true,
    iterations: 1,
  }
}

/**
 * Estimate initial distortion from reprojection errors.
 * Returns simplified 5-parameter distortion.
 */
export function estimateInitialDistortion(
  observations: Array<{
    worldPt: Vec3
    imagePt: { u: number; v: number }
  }>,
  viewIndices: number[][],
  intrinsics: Intrinsics,
  rvecs: Vec3[],
  tvecs: Vec3[]
): number[] {
  // Simple estimation based on mean radial error
  const radialErrors: { r: number; err: number }[] = []

  for (let i = 0; i < observations.length; i++) {
    const obs = observations[i]!
    const viewIdx = viewIndices[i]![0]!
    const rvec = rvecs[viewIdx]!
    const tvec = tvecs[viewIdx]!

    const R = rodriguesToMatrix(rvec)
    const Xc = Mat3.mulVec(R, obs.worldPt)
    const x = (Xc.x + tvec.x) / (Xc.z + tvec.z)
    const y = (Xc.y + tvec.y) / (Xc.z + tvec.z)

    const u = intrinsics.fx * x + intrinsics.skew * y + intrinsics.cx
    const v = intrinsics.fy * y + intrinsics.cy

    const dx = u - obs.imagePt.u
    const dy = v - obs.imagePt.v
    const err = Math.sqrt(dx * dx + dy * dy)
    const r = Math.sqrt(x * x + y * y)

    if (r > 0.01) {
      radialErrors.push({ r, err })
    }
  }

  if (radialErrors.length < 10) {
    return [0, 0, 0, 0, 0]
  }

  // Fit k1 from linear regression: err ≈ k1 * r^2
  let sumR2 = 0
  let sumErr = 0
  for (const re of radialErrors) {
    sumR2 += re.r * re.r
    sumErr += re.err * re.r * re.r
  }

  const k1 = sumErr / sumR2 * 0.1 // Scale factor

  return [Math.max(-1, Math.min(1, k1)), 0, 0, 0, 0]
}
