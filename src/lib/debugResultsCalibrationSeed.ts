import { zeroRationalDistortion8 } from '@/lib/cameraModel'
import type { CalibrationOk } from '@/workers/calibration.worker'

/** Flip to false when finished debugging the Results 3D view. */
export const DEBUG_SEED_RESULTS_CALIBRATION = true

/** Minimal plausible `CalibrationOk` with several 3D object points around the origin. */
export function debugSeedCalibrationOk(): CalibrationOk {
  return {
    kind: 'ok',
    K: { fx: 800, fy: 800, cx: 640, cy: 360 },
    distortion: zeroRationalDistortion8(),
    extrinsics: [
      {
        frameId: 0,
        R: [1, 0, 0, 0, 1, 0, 0, 0, 1],
        t: { x: 0, y: 0, z: 12 },
      },
    ],
    updatedTargetPoints: [
      { pointId: 10000, position: { x: 0, y: 0, z: 0 } },
      { pointId: 10001, position: { x: 3.5, y: -1.2, z: 0.08 } },
      { pointId: 10002, position: { x: -2.8, y: 2.4, z: -0.12 } },
      { pointId: 10003, position: { x: 1.2, y: 3.6, z: 0.15 } },
      { pointId: 20000, position: { x: -3.2, y: -2.7, z: 0.05 } },
      { pointId: 20001, position: { x: 2.1, y: 1.9, z: -0.2 } },
      { pointId: 20002, position: { x: 0.4, y: -3.5, z: 0.11 } },
    ],
    rmsPx: 0.35,
    perFrameRmsPx: [
      [0, 0.35],
    ],
  }
}
