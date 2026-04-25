/**
 * Tests for LM calibration refinement
 */
import { describe, it, expect } from 'vitest'
import { refineCalibration, type Observation, type ViewParams, type CameraParams } from './levmarq'

describe('refineCalibration', () => {
  it('should modify parameters from initial values', () => {
    // Create simple test data
    const observations: Observation[] = [
      { worldPt: { x: -1, y: -1, z: 5 }, imagePt: { u: 800, v: 800 } },
      { worldPt: { x: 1, y: -1, z: 5 }, imagePt: { u: 1200, v: 800 } },
      { worldPt: { x: -1, y: 1, z: 5 }, imagePt: { u: 800, v: 1200 } },
      { worldPt: { x: 1, y: 1, z: 5 }, imagePt: { u: 1200, v: 1200 } },
      { worldPt: { x: 0, y: 0, z: 5 }, imagePt: { u: 1000, v: 1000 } },
    ]

    const viewIndices: number[][] = observations.map(() => [0])

    const viewParams: ViewParams[] = [
      { rvec: { x: 0, y: 0, z: 0 }, tvec: { x: 0, y: 0, z: 0 } }
    ]

    // Initial params with error in focal length
    const initialParams: CameraParams = {
      intrinsics: { fx: 1100, fy: 1100, cx: 1000, cy: 1000, skew: 0 },
      distortion: { k1: 0, k2: 0, p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0 }
    }

    const refined = refineCalibration(observations, viewIndices, initialParams, viewParams, 100, 0.1)

    // Parameters should have been updated
    expect(refined.intrinsics.fx).not.toBe(1100)
    expect(refined.intrinsics.fy).not.toBe(1100)
  })

  it('should handle multiple views', () => {
    const observations: Observation[] = [
      // View 0
      { worldPt: { x: -1, y: -1, z: 5 }, imagePt: { u: 800, v: 800 } },
      { worldPt: { x: 1, y: -1, z: 5 }, imagePt: { u: 1200, v: 800 } },
      // View 1 (offset - different tvec)
      { worldPt: { x: -1, y: -1, z: 5 }, imagePt: { u: 700, v: 700 } },
      { worldPt: { x: 1, y: -1, z: 5 }, imagePt: { u: 1100, v: 700 } },
    ]

    const viewIndices: number[][] = [[0], [0], [1], [1]]

    const viewParams: ViewParams[] = [
      { rvec: { x: 0, y: 0, z: 0 }, tvec: { x: 0, y: 0, z: 0 } },
      { rvec: { x: 0, y: 0, z: 0 }, tvec: { x: 0.1, y: 0.1, z: 0 } },
    ]

    const initialParams: CameraParams = {
      intrinsics: { fx: 1000, fy: 1000, cx: 1000, cy: 1000, skew: 0 },
      distortion: { k1: 0, k2: 0, p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0 }
    }

    const refined = refineCalibration(observations, viewIndices, initialParams, viewParams, 50, 0.1)

    // Should have modified parameters
    expect(refined.intrinsics.fx).toBeDefined()
    expect(refined.intrinsics.fy).toBeDefined()
  })

  it('should return CameraParams type', () => {
    const observations: Observation[] = [
      { worldPt: { x: 0, y: 0, z: 5 }, imagePt: { u: 1000, v: 1000 } },
    ]

    const viewIndices: number[][] = [[0]]
    const viewParams: ViewParams[] = [
      { rvec: { x: 0, y: 0, z: 0 }, tvec: { x: 0, y: 0, z: 0 } }
    ]

    const initialParams: CameraParams = {
      intrinsics: { fx: 1000, fy: 1000, cx: 1000, cy: 1000, skew: 0 },
      distortion: { k1: 0, k2: 0, p1: 0, p2: 0, k3: 0, k4: 0, k5: 0, k6: 0 }
    }

    const result = refineCalibration(observations, viewIndices, initialParams, viewParams)

    // Should have all expected fields
    expect(result.intrinsics.fx).toBeDefined()
    expect(result.intrinsics.fy).toBeDefined()
    expect(result.intrinsics.cx).toBeDefined()
    expect(result.intrinsics.cy).toBeDefined()
    expect(result.distortion.k1).toBeDefined()
    expect(result.distortion.k2).toBeDefined()
  })
})
