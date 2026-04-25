/**
 * Compare our full pipeline (Zhang + LM) with OpenCV calibrateCamera
 * Fixed: handles rotation matrix format correctly
 */
import fs from 'fs'
import { computeHomography } from './dltHomography'
import { solveIntrinsicsFromHomographies, extrinsicsFromHomography } from '../zhangCalibration'
import type { Intrinsics } from '../zhangCalibration'
import type { DistortionCoeffs } from '../distortion'
import { distortPoint } from '../distortion'
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

/**
 * Convert column-major rotation matrix to column-major flat array.
 * The column-major format is: [c0x, c1x, c2x, c0y, c1y, c2y, c0z, c1z, c2z]
 */
function colMajorFromRowMajor(R: number[]): number[] {
  return [
    R[0]!, R[3]!, R[6]!,  // column 0: R00, R10, R20
    R[1]!, R[4]!, R[7]!,  // column 1: R01, R11, R21
    R[2]!, R[5]!, R[8]!,  // column 2: R02, R12, R22
  ]
}

function main() {
  const data: TestData = JSON.parse(fs.readFileSync('/tmp/multi_view_data.json', 'utf-8'))

  console.log("=== Ground Truth ===")
  console.log("K:", data.groundTruth.K)
  console.log("dist:", data.groundTruth.dist)

  console.log("\n=== OpenCV calibrateCamera Result ===")
  console.log("RMS:", data.opencv.rms.toFixed(4))
  console.log("K:", data.opencv.K)
  console.log("dist:", data.opencv.dist)

  // Compute homographies
  const homographies: number[][][] = []
  for (const view of data.views) {
    const srcPoints = view.objectPoints.map(p => ({ x: p[0], y: p[1] }))
    const dstPoints = view.imagePoints.map(p => ({ x: p[0], y: p[1] }))
    homographies.push(computeHomography(srcPoints, dstPoints))
  }

  // Zhang closed-form K
  console.log("\n=== Zhang Closed-Form ===")
  const initialK = solveIntrinsicsFromHomographies(homographies)
  if (!initialK) {
    console.log("Failed to compute intrinsics!")
    return
  }
  console.log("Zhang K:", initialK)

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

  // Compute initial extrinsics from homographies
  const initialViewParams: ViewParams[] = []
  for (let v = 0; v < homographies.length; v++) {
    const ext = extrinsicsFromHomography(homographies[v]!, initialK)
    if (!ext) continue

    // extrinsicsFromHomography returns R in column-major format
    // matrixToRodrigues expects row-major input, returns row-major output
    // So we need to convert R from column-major to row-major, call matrixToRodrigues, then back
    const RrowMajor = colMajorFromRowMajor([...ext.R])
    const rvec = matrixToRodrigues(RrowMajor)

    initialViewParams.push({
      rvec,
      tvec: ext.t,
    })
  }

  console.log("\n=== Initial Extrinsics (from homographies) ===")
  console.log("First view rvec:", initialViewParams[0]?.rvec)
  console.log("First view tvec:", initialViewParams[0]?.tvec)

  // Initial distortion (zero)
  const initialDist: DistortionCoeffs = {
    k1: 0, k2: 0, p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0
  }

  // Run LM refinement
  console.log("\n=== LM Refinement ===")
  const result = refineCalibration(
    observations,
    viewIndices,
    { intrinsics: initialK, distortion: initialDist },
    initialViewParams,
    100,  // maxIter
    1e-3  // damping
  )

  console.log("\n=== Refined Parameters ===")
  console.log("Refined K:", result.intrinsics)
  console.log("Refined dist:", result.distortion)

  // Compare with OpenCV
  const opencvK = data.opencv.K
  console.log("\n=== Comparison ===")
  console.log("\n--- K ---")
  console.log(`fx: ours=${result.intrinsics.fx.toFixed(2)}, opencv=${opencvK[0][0].toFixed(2)}, err=${((result.intrinsics.fx - opencvK[0][0]) / opencvK[0][0] * 100).toFixed(2)}%`)
  console.log(`fy: ours=${result.intrinsics.fy.toFixed(2)}, opencv=${opencvK[1][1].toFixed(2)}, err=${((result.intrinsics.fy - opencvK[1][1]) / opencvK[1][1] * 100).toFixed(2)}%`)
  console.log(`cx: ours=${result.intrinsics.cx.toFixed(2)}, opencv=${opencvK[0][2].toFixed(2)}, err=${((result.intrinsics.cx - opencvK[0][2]) / opencvK[0][2] * 100).toFixed(2)}%`)
  console.log(`cy: ours=${result.intrinsics.cy.toFixed(2)}, opencv=${opencvK[1][2].toFixed(2)}, err=${((result.intrinsics.cy - opencvK[1][2]) / opencvK[1][2] * 100).toFixed(2)}%`)

  console.log("\n--- Distortion ---")
  console.log(`k1: ours=${result.distortion.k1.toFixed(6)}, opencv=${data.opencv.dist[0][0].toFixed(6)}`)
  console.log(`k2: ours=${result.distortion.k2.toFixed(6)}, opencv=${data.opencv.dist[0][1].toFixed(6)}`)
  console.log(`p1: ours=${result.distortion.p1.toFixed(6)}, opencv=${data.opencv.dist[0][2].toFixed(6)}`)
  console.log(`p2: ours=${result.distortion.p2.toFixed(6)}, opencv=${data.opencv.dist[0][3].toFixed(6)}`)
  console.log(`k3: ours=${result.distortion.k3.toFixed(6)}, opencv=${data.opencv.dist[0][4].toFixed(6)}`)

  console.log("\n--- Summary ---")
  console.log(`OpenCV RMS: ${data.opencv.rms.toFixed(4)} pixels`)
}

main()
