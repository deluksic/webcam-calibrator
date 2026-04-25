/**
 * Compare LM refinement with proper initialization
 * Key insight: extrinsics depend on K, so we need to recompute them
 */
import fs from 'fs'
import { computeHomography } from './dltHomography'
import { solveIntrinsicsFromHomographies, extrinsicsFromHomography } from '../zhangCalibration'
import type { Intrinsics } from '../zhangCalibration'
import type { DistortionCoeffs } from './distortion'
import { distortPoint } from './distortion'
import { refineCalibration, type Observation, type ViewParams } from './levmarq'
import { matrixToRodrigues, rodriguesToMatrix } from './rodrigues'

interface TestData {
  groundTruth: { K: number[][]; dist: number[][] }
  opencv: { rms: number; K: number[][]; dist: number[][] }
  views: Array<{
    objectPoints: number[][]
    imagePoints: number[][]
  }>
}

function colMajorFromRowMajor(R: number[]): number[] {
  return [R[0]!, R[3]!, R[6]!, R[1]!, R[4]!, R[7]!, R[2]!, R[5]!, R[8]!]
}

/**
 * Compute initial view parameters from homographies and K
 */
function computeInitialViewParams(homographies: number[][][], K: Intrinsics): ViewParams[] {
  const viewParams: ViewParams[] = []
  for (const h of homographies) {
    const ext = extrinsicsFromHomography(h, K)
    if (!ext) continue
    const RrowMajor = colMajorFromRowMajor([...ext.R])
    const rvec = matrixToRodrigues(RrowMajor)
    viewParams.push({ rvec, tvec: ext.t })
  }
  return viewParams
}

/**
 * Compute RMS error with given parameters
 */
function computeRms(
  observations: Observation[],
  viewIndices: number[][],
  viewParams: ViewParams[],
  K: Intrinsics,
  dist: DistortionCoeffs
): number {
  let sumSq = 0
  let n = 0

  for (let i = 0; i < observations.length; i++) {
    const obs = observations[i]!
    const viewIdx = viewIndices[i]![0]!
    const vp = viewParams[viewIdx]!
    const R = rodriguesToMatrix(vp.rvec)

    // Column-major rotation matrix
    const Rcol = colMajorFromRowMajor(R)

    // Transform to camera frame
    const Xc = Rcol[0]! * obs.worldPt.x + Rcol[1]! * obs.worldPt.y + Rcol[2]! * obs.worldPt.z + vp.tvec.x
    const Yc = Rcol[3]! * obs.worldPt.x + Rcol[4]! * obs.worldPt.y + Rcol[5]! * obs.worldPt.z + vp.tvec.y
    const Zc = Rcol[6]! * obs.worldPt.x + Rcol[7]! * obs.worldPt.y + Rcol[8]! * obs.worldPt.z + vp.tvec.z

    const x = Xc / Zc
    const y = Yc / Zc

    const distorted = distortPoint({ x, y }, dist)
    const u = K.fx * distorted.x + K.skew * distorted.y + K.cx
    const v = K.fy * distorted.y + K.cy

    const du = u - obs.imagePt.u
    const dv = v - obs.imagePt.v
    sumSq += du * du + dv * dv
    n++
  }

  return n > 0 ? Math.sqrt(sumSq / n) : 0
}

function main() {
  const data: TestData = JSON.parse(fs.readFileSync('/tmp/multi_view_data.json', 'utf-8'))

  // Build observations
  const observations: Observation[] = []
  const viewIndices: number[][] = []

  for (let v = 0; v < data.views.length; v++) {
    const view = data.views[v]!
    for (let i = 0; i < view.objectPoints.length; i++) {
      observations.push({
        worldPt: { x: view.objectPoints[i]![0], y: view.objectPoints[i]![1], z: 0 },
        imagePt: { u: view.imagePoints[i]![0], v: view.imagePoints[i]![1] },
      })
      viewIndices.push([v])
    }
  }

  // Compute homographies
  const homographies: number[][][] = []
  for (const view of data.views) {
    const srcPoints = view.objectPoints.map(p => ({ x: p[0], y: p[1] }))
    const dstPoints = view.imagePoints.map(p => ({ x: p[0], y: p[1] }))
    homographies.push(computeHomography(srcPoints, dstPoints))
  }

  // Ground truth
  const gtK: Intrinsics = {
    fx: data.groundTruth.K[0][0],
    fy: data.groundTruth.K[1][1],
    cx: data.groundTruth.K[0][2],
    cy: data.groundTruth.K[1][2],
    skew: 0,
  }
  const gtDist: DistortionCoeffs = {
    k1: data.groundTruth.dist[0][0],
    k2: data.groundTruth.dist[0][1],
    p1: data.groundTruth.dist[0][2],
    p2: data.groundTruth.dist[0][3],
    k3: data.groundTruth.dist[0][4],
    k4: 0, k5: 0, k6: 0,
  }

  // OpenCV K
  const opencvK: Intrinsics = {
    fx: data.opencv.K[0][0],
    fy: data.opencv.K[1][1],
    cx: data.opencv.K[0][2],
    cy: data.opencv.K[1][2],
    skew: 0,
  }
  const opencvDist: DistortionCoeffs = {
    k1: data.opencv.dist[0][0],
    k2: data.opencv.dist[0][1],
    p1: data.opencv.dist[0][2],
    p2: data.opencv.dist[0][3],
    k3: data.opencv.dist[0][4],
    k4: 0, k5: 0, k6: 0,
  }

  // Zhang K
  const zhangK = solveIntrinsicsFromHomographies(homographies)!

  // Zero distortion
  const zeroDist: DistortionCoeffs = {
    k1: 0, k2: 0, p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0
  }

  console.log("=== Ground Truth ===")
  console.log("K:", gtK)
  console.log("dist:", gtDist)
  console.log("RMS:", computeRms(observations, viewIndices, computeInitialViewParams(homographies, gtK), gtK, gtDist).toFixed(4))

  console.log("\n=== OpenCV calibrateCamera ===")
  console.log("K:", opencvK)
  console.log("dist:", opencvDist)
  console.log("RMS:", data.opencv.rms.toFixed(4))

  console.log("\n=== Zhang Closed-Form ===")
  console.log("K:", zhangK)
  console.log("RMS (zhang K, zero dist, zhang extrinsics):", computeRms(observations, viewIndices, computeInitialViewParams(homographies, zhangK), zhangK, zeroDist).toFixed(4))

  // Test 1: Zhang K + LM refinement (extrinsics from Zhang K)
  console.log("\n=== Test 1: Zhang K + LM ===")
  const zhangParams = computeInitialViewParams(homographies, zhangK)
  const result1 = refineCalibration(
    observations, viewIndices,
    { intrinsics: zhangK, distortion: zeroDist },
    zhangParams,
    100, 1e-3
  )
  console.log("Initial RMS:", result1.initialError.toFixed(4))
  console.log("Final RMS:", result1.finalError.toFixed(4))
  console.log("K:", result1.intrinsics)

  // Test 2: OpenCV K + LM refinement (extrinsics from OpenCV K)
  console.log("\n=== Test 2: OpenCV K + LM ===")
  const opencvParams = computeInitialViewParams(homographies, opencvK)
  const result2 = refineCalibration(
    observations, viewIndices,
    { intrinsics: opencvK, distortion: zeroDist },
    opencvParams,
    100, 1e-3
  )
  console.log("Initial RMS:", result2.initialError.toFixed(4))
  console.log("Final RMS:", result2.finalError.toFixed(4))
  console.log("K:", result2.intrinsics)

  // Test 3: OpenCV K + OpenCV dist + LM refinement (extrinsics from OpenCV K)
  console.log("\n=== Test 3: OpenCV K + OpenCV dist + LM ===")
  const result3 = refineCalibration(
    observations, viewIndices,
    { intrinsics: opencvK, distortion: opencvDist },
    opencvParams,
    100, 1e-3
  )
  console.log("Initial RMS:", result3.initialError.toFixed(4))
  console.log("Final RMS:", result3.finalError.toFixed(4))
  console.log("K:", result3.intrinsics)
  console.log("dist:", result3.distortion)

  // Test 4: Ground truth parameters (verify our RMS computation)
  console.log("\n=== Test 4: Ground Truth Parameters ===")
  const gtParams = computeInitialViewParams(homographies, gtK)
  console.log("RMS with GT K and GT dist:", computeRms(observations, viewIndices, gtParams, gtK, gtDist).toFixed(4))
  console.log("RMS with GT K and zero dist:", computeRms(observations, viewIndices, gtParams, gtK, zeroDist).toFixed(4))

  console.log("\n=== Summary ===")
  console.log("Ground Truth RMS:", computeRms(observations, viewIndices, gtParams, gtK, gtDist).toFixed(4))
  console.log("OpenCV RMS:", data.opencv.rms.toFixed(4))
  console.log("Zhang + LM:", result1.finalError.toFixed(4))
  console.log("OpenCV K + LM:", result2.finalError.toFixed(4))
  console.log("OpenCV K + OpenCV dist + LM:", result3.finalError.toFixed(4))
}

main()
