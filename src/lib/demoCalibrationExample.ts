import { createCalibrationLibraryEntry } from '@/lib/calibrationLibraryTypes'
import { DEFAULT_TARGET_CONFIG } from '@/lib/hexTarget/config'
import { defaultHexTargetModel } from '@/lib/hexTarget/target'
import type { Mat3 } from '@/lib/poseMath'
import type { CalibrationOk } from '@/workers/calibration.worker'

const IDENTITY: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1]

function identity(): Mat3 {
  return [...IDENTITY]
}

/** Example solve used as the default Saved calibration entry so Results is not empty on first visit. */
export function demoCalibrationOk(): CalibrationOk {
  const points = defaultHexTargetModel().objectPoints.map((point) => ({ ...point }))
  return {
    kind: 'ok',
    K: {
      fx: 638.7490234375,
      fy: 639.4755249023438,
      cx: 326.4219055175781,
      cy: 237.4363555908203,
    },
    distortion: [
      -0.12911082804203033, 0.9649162292480469, 0.006222105585038662, 0.0006739901727996767,
      -1.9532757997512817, 0, 0, 0,
    ],
    extrinsics: [
      { frameId: 1, R: identity(), t: { x: -1.0, y: -1.4, z: 15.6 } },
      { frameId: 2, R: identity(), t: { x: -1.8, y: 1.3, z: 14.0 } },
      { frameId: 3, R: identity(), t: { x: 0.1, y: 1.4, z: 12.3 } },
      { frameId: 4, R: identity(), t: { x: 1.8, y: -1.5, z: 13.0 } },
    ],
    updatedTargetPoints: points,
    rmsPx: 0.2491298397112407,
    perFrameRmsPx: [
      [1, 0.401848439978948],
      [2, 0.1963474012530714],
      [3, 0.1639799454629092],
      [4, 0.28330208140597835],
    ],
    imageSize: { width: 640, height: 480 },
    cameraId: 'demo',
    targetConfig: DEFAULT_TARGET_CONFIG,
  }
}

/** Single built-in library row; safe to remove like any other saved run. */
export function createDemoCalibrationLibraryEntry() {
  const ok = demoCalibrationOk()
  return createCalibrationLibraryEntry(
    ok,
    {
      validSolveFrameCount: Math.max(4, ok.extrinsics.length),
      videoWidth: 640,
      videoHeight: 480,
    },
    'Demo calibration',
  )
}
